import os
import json
from pprint import pprint
from mllm.llava_onevision_qwen2_0_5b_ov_hf.custom_models.modeling_llava_onevision import LlavaOnevisionForConditionalGeneration
from mllm.llava_onevision_qwen2_0_5b_ov_hf.custom_models.configuration_llava_onevision import LlavaOnevisionConfig

model_name_or_path = "/ppio_net0/code/mllm-lightning/mllm/llava_onevision_qwen2_0_5b_ov_hf"

# 1) 先看磁盘上的原始 config.json
config_path = os.path.join(model_name_or_path, "config.json")
with open(config_path, "r", encoding="utf-8") as f:
    raw = json.load(f)

print("=== raw json ===")
print("model_type:", raw.get("model_type"))
print("text_config.hidden_size:", raw.get("text_config", {}).get("hidden_size"))
print("text_config.num_attention_heads:", raw.get("text_config", {}).get("num_attention_heads"))
print("vision_config.hidden_size:", raw.get("vision_config", {}).get("hidden_size"))
print("vision_config.num_attention_heads:", raw.get("vision_config", {}).get("num_attention_heads"))

# 2) 再看 HF 读出来的 dict
cfg_dict, _ = LlavaOnevisionConfig.get_config_dict(
    model_name_or_path,
    local_files_only=True,
)

print("\n=== get_config_dict ===")
print("model_type:", cfg_dict.get("model_type"))
print("text_config.hidden_size:", cfg_dict.get("text_config", {}).get("hidden_size"))
print("text_config.num_attention_heads:", cfg_dict.get("text_config", {}).get("num_attention_heads"))
print("vision_config.hidden_size:", cfg_dict.get("vision_config", {}).get("hidden_size"))
print("vision_config.num_attention_heads:", cfg_dict.get("vision_config", {}).get("num_attention_heads"))

# 3) 最后看构造后的对象
model_config = LlavaOnevisionConfig.from_pretrained(
    model_name_or_path,
    local_files_only=True,
)

print("\n=== loaded config object ===")
print("model_type:", model_config.model_type)
print("text hidden_size:", model_config.text_config.hidden_size)
print("text heads:", model_config.text_config.num_attention_heads)
print("vision hidden_size:", model_config.vision_config.hidden_size)
print("vision heads:", model_config.vision_config.num_attention_heads)

# 4) 确认实际 import 的类文件来自哪里
import inspect
print("\n=== class source file ===")
print(inspect.getfile(LlavaOnevisionConfig))