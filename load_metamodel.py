from safetensors import safe_open
from transformers import BloomForCausalLM, BloomConfig
import os

model_name = "checkpoint"
# TODO: meta initialize
configuration = BloomConfig.from_json_file(f"{model_name}/config.json")
model = BloomForCausalLM(configuration)
