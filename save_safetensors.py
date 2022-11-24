from safetensors.torch import save_file
from transformers import BloomForCausalLM, BloomConfig
from itertools import islice

save_path = "checkpoint"  # "checkpoint" "bloom560m"
# configuration = BloomConfig(hidden_size=64,
#                             n_layer=4,
#                             n_head=8,
#                             )
# 
# configuration = BloomConfig(hidden_size=8192,
#                             n_layer=48,
#                             n_head=64,
#                             )
# 
# model = BloomForCausalLM(configuration)
# 
model = BloomForCausalLM.from_pretrained("/data2/users/lczht/bloom-560m")
# save model tensors to small .safetensors files
dict_stride = 12 # tensor num per file
turns = len(model.state_dict()) // dict_stride + 1
for i in range(turns):
    batch = dict(islice(model.state_dict().items(), i * dict_stride, (i + 1) * dict_stride))
    save_file(batch, save_path + f"/part_{i}.safetensors")
model.config.save_pretrained(save_path) # save config