# === view_tree.py ===

import torch
import os
from angelnet_core_v6_4 import AngelNet

CHECKPOINT_FILE = "angelnet_checkpoint_v6_4.pt"

def load_model():
    model = AngelNet(input_dim=784, output_dim=10, hidden_dim=128, max_depth=5)
    if os.path.exists(CHECKPOINT_FILE):
        model.load_state_dict(torch.load(CHECKPOINT_FILE, map_location="cpu"))
        print(f"[LOADED] Checkpoint loaded from {CHECKPOINT_FILE}")
    else:
        print(f"[WARN] No checkpoint found at {CHECKPOINT_FILE}")
    return model

def print_tree(reflections, indent=""):
    for layer, data in reflections.items():
        if isinstance(data, list):
            print(f"{indent}Layer: {layer}")
            for entry in data:
                print(f"{indent}  └─ {entry}")
        else:
            print(f"{indent}{layer}: {data}")

if __name__ == "__main__":
    model = load_model()
    reflections = model.reflect()
    print("\n[ANGELNET FRACTAL TREE STRUCTURE]")
    print_tree(reflections)