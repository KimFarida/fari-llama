import torch
from safetensors import safe_open
import numpy as np


model_path = "models/tinyllama/model.safetensors"

# gets model weights and converts to raw so can load using c++
with safe_open(model_path, framework="pt", device="cpu") as f:
    embed = f.get_tensor("model.embed_tokens.weight")
    embed = embed.to(torch.float32).numpy()
    embed.tofile("models/tinyllama/embed_tokens.bin")
    print(f"Exported embed_tokens: {embed.shape}")

# export attention weights Q K V
with safe_open(model_path, framework="pt", device="cpu") as f:
    #export only layer 0 weights for now
    for name in ["q_proj", "k_proj", "v_proj", "o_proj"]:
        tensor = f.get_tensor(f"model.layers.0.self_attn.{name}.weight")
        tensor = tensor.to(torch.float32).numpy()
        tensor.tofile(f"models/tinyllama/{name}_0.bin")
        print(f"Exported {name} : {tensor.shape}")

