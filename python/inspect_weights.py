from safetensors import safe_open
import json


model_path = "models/tinyllama/model.safetensors"
config_path = "models/tinyllama/config.json"

with open(config_path) as f:
    config = json.load(f)

print("---CONFIG---")
print(f"Hidden size: {config['hidden_size']}")
print(f"Num layers: {config['num_hidden_layers']}")
print(f"Num attention heads: {config['num_attention_heads']}")
print(f"Vocab size: {config['vocab_size']}")
print(f"Intermediate size: {config['intermediate_size']}")

print("\n")

print("---WEIGHTS---")
with safe_open(model_path, framework="pt", device=0) as f:
    for key in f.keys():
        print(f"{key}: {f.get_tensor(key).shape}")