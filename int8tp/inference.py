import time
import os
import argparse
import torch
from transformers import BloomTokenizerFast, BloomForCausalLM, BloomConfig, AutoModelForCausalLM, AutoTokenizer
import torch.distributed as dist
from colossalai.tensor import ProcessGroup

import colossalai
from colossalai.utils.model.colo_init_context import ColoInitContext
from colossalai.tensor import ProcessGroup, ReplicaSpec
from colossalai.utils.model.lazy_init_context import LazyInitContext

INPUT_SENTENCE = "hello, my dog is cute"


def print_rank0(str, rank=0):
    if rank == 0:
        print(str)
    else:
        return

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_shard_int", required=False, type=bool, help="a flag inidicates init model in shards")
    parser.add_argument("--model_path", required=False, type=str, default="/data2/users/lczht/bloom-560m", help="used by dist launchers")
    parser.add_argument("--backend", required=False, type=str, default="colossalai", help="backend of inference, [colossalai, torch, accelerate]")
    parser.add_argument("--dtype", type=str, help="float16 or int8", choices=["int8", "float16"], default="float16")

    parser.add_argument("--use_config", dest="use_config", action="store_true")
    parser.add_argument("--hidden_size", type=int, default=64)
    parser.add_argument("--n_layer", type=int, default=2)
    parser.add_argument("--n_head", type=int, default=8)
    parser.add_argument("--max_new_tokens", type=int, default=60)
    return parser.parse_args()


def run_CAI_int8(args):
    # from_config = True if args.use_config else False
    # configuration = BloomConfig(hidden_size=args.hidden_size,  # 64
    #                             n_layer=args.n_layer,  # 2
    #                             n_head=args.n_head,  # 8
    #                             )
    model_path = args.model_path
    colossalai.launch_from_torch(config={})
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    input_sentence = INPUT_SENTENCE
    max_new_tokens = args.max_new_tokens
    tokenizer = BloomTokenizerFast.from_pretrained(model_path)
    from utils import load_bloom_for_rank
    model = load_bloom_for_rank(model_path, rank = rank, world_size=world_size)

    num_params = 0
    for pn, param in model.named_parameters(recurse=True):
        if hasattr(param, 'is_visited'):
            continue
        num_params += param.numel()
        param.is_visited = True
    print("num_params: ", num_params)
    print('initialize INT8 TP OK')
    input_tokens = tokenizer(input_sentence, return_tensors="pt")
    for k, v in input_tokens.items():
        input_tokens[k] = v.cuda()

    generate_kwargs = dict(max_new_tokens=max_new_tokens, do_sample=False)
    torch.cuda.reset_peak_memory_stats(rank)
    # warmup
    for i in range(1):
        outputs = model.generate(**input_tokens, **generate_kwargs)
    # model inferences
    print("inference start")
    t_generate_span = 0
    turn_num = 10
    for i in range(turn_num):
        t_generate_start = time.time()
        outputs = model.generate(**input_tokens, **generate_kwargs)
        # torch.cuda.synchronize()
        t_generate_span += time.time() - t_generate_start
    print_rank0(f"colossalai t_generate_span: {t_generate_span / turn_num}", rank)
    max_usage = torch.cuda.max_memory_allocated(rank)
    print(f"max cuda memory usage: {max_usage / 1024 /1024} MB")
    
if __name__ == '__main__':
    args = get_args()
    print(args)
    if args.backend == "colossalai":
        if args.dtype == "float16":
            print("colossalai fp16")
            # run_CAI(args)
        elif args.dtype == "int8":
            print("colossalai int8")
            run_CAI_int8(args)
    elif args.backend == "torch":
        print("torch")
        # run_torch(args)
    elif args.backend == "accelerate":
        print("accelerate")
        # run_accelerate(args)
