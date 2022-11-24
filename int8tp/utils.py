from safetensors.torch import save_file
from safetensors import safe_open
from transformers import BloomForCausalLM, BloomConfig
from itertools import islice
import os
from colossalai.utils.model.lazy_init_context import LazyInitContext

import torch
from torch import nn, Tensor
import torch.distributed as dist
import bitsandbytes as bnb
import torch.nn.functional as F
from typing import Optional
from torch.distributed.distributed_c10d import ReduceOp
from colossalai.tensor import ColoParameter, ReplicaSpec

class Linear8bitTP(torch.nn.Module):
    def __init__(
        self,
        input_features,
        output_features,
        bias=True,
        has_fp16_weights=False,
        memory_efficient_backward=False,
        threshold=6.0,
        index=None,
        rank = 0,
        world_size = 1
    ):
        super(Linear8bitTP, self).__init__()
        self.rank = rank
        self.world_size = world_size
        self.linear8bit = bnb.nn.Linear8bitLt(input_features, output_features, bias,
                                              has_fp16_weights, memory_efficient_backward,
                                              threshold, index)

    def forward(self, x):
        out = self.linear8bit(x)
        tensor_list = [torch.zeros_like(out) for _ in range(self.world_size)]
        out = torch.cat(tensor_list, dim=2)
        del tensor_list
        del self.state.CxB
        
        return out

class LinearTP(torch.nn.Linear):
    def __init__(self, input_features, output_features, bias=False, weight_data=None, bias_data=None):
        super(LinearTP, self).__init__(input_features, output_features, bias)
        self.weight = weight_data
        self.bias = bias_data
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        
    def forward(self, x):
        x = x.chunk(self.world_size, dim=2)[self.rank]
        out = F.linear(x, self.weight, self.bias)
        dist.all_reduce(out, op=ReduceOp.SUM)
        return out

class EmbeddingTP(torch.nn.Embedding):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: Optional[int] = None,
        max_norm: Optional[float] = None,
        norm_type: float = 2.0,
        scale_grad_by_freq: bool = False,
        sparse: bool = False,
        weight: Optional[Tensor] = None,
    ) -> None:
        super(EmbeddingTP, self).__init__(
            num_embeddings,
            embedding_dim,
            padding_idx,
            max_norm,
            norm_type,
            scale_grad_by_freq,
            sparse,
            weight,
        )
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.weight = weight

    def forward(self, input: Tensor) -> Tensor:
        emb = F.embedding(
            input,
            self.weight,
            self.padding_idx,
            self.max_norm,
            self.norm_type,
            self.scale_grad_by_freq,
            self.sparse,
        )
        
        tensor_list = [torch.zeros_like(emb) for _ in range(self.world_size)]
        dist.all_gather(tensor_list, emb)
        emb = torch.cat(tensor_list, dim=2)
        del tensor_list
        return emb

def replace_8bit_linear_tp(model, threshold=6.0, modules_to_not_convert="lm_head"):
    for name, module in model.named_children():
        if len(list(module.children())) > 0:
            replace_8bit_linear_tp(module, threshold, modules_to_not_convert)

        if isinstance(module, nn.Linear) and name not in modules_to_not_convert:
                model._modules[name] = Linear8bitTP(
                        input_features=module.in_features,
                        output_features=module.out_features,
                        threshold=6.0,
                )
        
        if isinstance(module, nn.Embedding):
            model._modules[name] = EmbeddingTP(
                num_embeddings=module.num_embeddings,
                embedding_dim=module.embedding_dim,
                padding_idx=module.padding_idx,
                max_norm=module.max_norm,
                norm_type=module.norm_type,
                scale_grad_by_freq=module.scale_grad_by_freq,
                sparse=module.sparse,
                weight=module.weight,
            )
        if name == 'lm_head':
            model._modules[name] = LinearTP(
                input_features=module.in_features,
                output_features=module.out_features,
                weight_data=module.weight,
                bias=False,
            )
    return model

def save_full_model(model, path: str = "checkpoint", stride=12, ):
    '''
    save a full model from configuration, no sharding, no quantization
    model: cpu model
    path: saved position
    stride: num tensors per saved file
    '''
    try:
        os.mkdir(path)
    except:
        pass
    num_files = (len(model.state_dict()) - 1) // stride + 1
    for i in range(num_files):
        batch = dict(islice(model.state_dict().items(), i * stride, (i + 1) * stride))
        save_file(batch, os.path.join(path, f"part_{i}.safetensors"))

def save_bloom_from_config(configuration : BloomConfig, path = "checkpoint", stride = 12):
    model = BloomForCausalLM(configuration)
    save_full_model(model, path, stride)
    model.config.save_pretrained(path)
    
def load_bloom_for_rank(path : str, rank = 0, world_size = 1, sharding = "tp", dtype = "int8"):
    if sharding != "tp":
        raise NotImplementedError
    if dtype != "int8":
        raise NotImplementedError
    configuration = BloomConfig.from_json_file(f"{path}/config.json")
    with LazyInitContext() as ctx:
        model = BloomForCausalLM(configuration)
    # replace layer
    replace_8bit_linear_tp(model)
    parameters = dict(model.named_parameters())
    filenames = []
    for f in os.listdir(path):
        if f.endswith(".safetensors"):
            filenames.append(os.path.join(path, f))
    for filename in filenames:
        with safe_open(filename, framework="pt", device=rank) as f:
            for name in f.keys():
                if name == 'lm_head.weight':
                    continue
                full_name = name
                module_name, param_name = full_name.rsplit(".", 1)
                module = model.get_submodule(module_name)
                tensor = f.get_tensor(name).data.contiguous().half()
                if isinstance(module, Linear8bitTP):
                    if "weight" in param_name:
                        weight = tensor
                        CB, _, SCB, _, _ = bnb.functional.double_quant(weight)
                        module._parameters[param_name] = bnb.nn.Int8Params(data=list(CB.chunk(world_size, dim=0))[rank].clone().detach(),
                                                                   SCB=list(SCB.chunk(world_size, dim=0))[rank].clone().detach())
                    elif "bias" in param_name:
                        bias = tensor
                        module._parameters[param_name] = list(bias.chunk(world_size, dim=0))[rank].clone().detach()
                elif isinstance(module, EmbeddingTP):
                    weight = tensor
                    module._parameters[param_name] = list(weight.chunk(world_size, dim=1))[rank].clone().detach()
                else:
                    module._parameters[param_name] = tensor
                if name == "transformer.word_embeddings.weight":
                    model.lm_head._parameters["weight"] = module.weight
    return model
