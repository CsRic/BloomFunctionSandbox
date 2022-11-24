from safetensors import safe_open
from transformers import BloomForCausalLM, BloomConfig
import os

model_name = "checkpoint"

filenames = []
for f in os.listdir(model_name):
    if f.endswith(".safetensors"):
        filenames.append(os.path.join(model_name, f))

for filename in filenames:
    with safe_open(filename, framework="pt", device=f"cuda:0") as f:
        for name in f.keys():
            print(name, f.get_slice(name)[:])
