from safetensors import safe_open
from transformers import BloomForCausalLM, BloomConfig
import os
from colossalai.utils.model.lazy_init_context import LazyInitContext

model_name = "checkpoint"
# initialize metamodel
configuration = BloomConfig.from_json_file(f"{model_name}/config.json")
with LazyInitContext() as ctx:
    model = BloomForCausalLM(configuration)
'''
quantize, set comm group for submodules
'''
print(model.state_dict().keys())
# fill model with checkpoint
parameters = dict(model.named_parameters())
filenames = []
for f in os.listdir(model_name):
    if f.endswith(".safetensors"):
        filenames.append(os.path.join(model_name, f))

for filename in filenames:
    with safe_open(filename, framework="pt", device=f"cuda:0") as f:
        for name in f.keys():
            if name == 'lm_head.weight':
                continue
            full_name = name
            module_name, param_name = full_name.rsplit(".", 1)
            module = model.get_submodule(module_name)
            current_tensor = parameters[full_name]

            slice_ = f.get_slice(name)
            
            '''
            quantize, choose slice_ for tensor
            '''
            
            tensor = slice_[:]
            if current_tensor.shape != tensor.shape:
                raise ValueError(
                    f"Name {name} -- Current {current_tensor.shape} and got {tensor.shape}"
                )
            tensor = tensor.contiguous()

            module._parameters[param_name] = tensor
            if name == "transformer.word_embeddings.weight":
                model.lm_head._parameters["weight"] = tensor

print(model.state_dict())