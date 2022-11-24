from safetensors.torch import save_file
from transformers import BloomForCausalLM, BloomConfig
from itertools import islice

configuration = BloomConfig(hidden_size=64,
                            n_layer=4,
                            n_head=8,
                            )

model = BloomForCausalLM(configuration)

dict_stride = 4
turns = len(model.state_dict()) // dict_stride
for i in range(turns):
    batch = dict(islice(model.state_dict().items(), i * dict_stride, (i + 1) * dict_stride))
    save_file(batch, f"checkpoint/part_{i}.safetensors")
model.config.save_pretrained("checkpoint")