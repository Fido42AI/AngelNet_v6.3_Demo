import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import os
from datetime import datetime
from angelnet_core_v6_4 import AngelNet

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Device: {DEVICE}")

# === Параметры ===
BATCH_SIZE = 64
EPOCHS = 10
INPUT_DIM = 28 * 28
OUTPUT_DIM = 10
MEMORY_LIMIT_MB = 5120
CHECKPOINT_FILE = "angelnet_checkpoint.pt"
LOG_FILE = "angelnet_detailed_log.txt"

# === Подготовка лог-файла ===
with open(LOG_FILE, "w", encoding="utf-8") as f:
    f.write(f"=== AngelNet Detailed Log ===\nStarted: {datetime.now()}\n\n")

# === Датасет ===
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_loader = torch.utils.data.DataLoader(
    datasets.FashionMNIST('./data', train=True, download=True, transform=transform),
    batch_size=BATCH_SIZE,
    shuffle=True
)

def log_console_and_file(message):
    print(message)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(message + "\n")

def estimate_memory_usage():
    import psutil
    process = psutil.Process(os.getpid())
    return process.memory_info().rss // (1024 * 1024)

def visualize_fractal_tree(reflection_dict, indent=""):
    def _print_node(node, level=0):
        indent = "    " * level
        if isinstance(node, dict):
            if "neuron" in node:
                log_console_and_file(f"{indent}└─ {node['neuron']}")
            if "sub_layer" in node:
                for subnode in node["sub_layer"]:
                    _print_node(subnode, level + 1)
        elif isinstance(node, list):
            for item in node:
                _print_node(item, level)
        else:
            log_console_and_file(f"{indent}• {str(node)}")

    log_console_and_file("[+] Fractal Tree:")
    for key, val in reflection_dict.items():
        if key == "t_live":
            log_console_and_file(f"[{key}] {val}")
        else:
            log_console_and_file(f" ┌─ {key}")
            _print_node(val, level=1)

def train_simple():
    model = AngelNet(input_dim=INPUT_DIM, output_dim=OUTPUT_DIM, hidden_dim=128, max_depth=5).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()
    t_global = 0

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0

        for batch_idx, (data, target) in enumerate(train_loader):
            data = data.view(data.size(0), -1).to(DEVICE)
            target = target.to(DEVICE)
            mentor_signal = torch.randn(data.size(0)).to(DEVICE)

            optimizer.zero_grad()
            output = model(data, t_global=t_global, mentor_signal=mentor_signal, epoch=epoch)
            loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if estimate_memory_usage() > MEMORY_LIMIT_MB:
                log_console_and_file(f"[!] Memory usage exceeded {MEMORY_LIMIT_MB}MB, stopping training.")
                torch.save(model.state_dict(), CHECKPOINT_FILE)
                log_console_and_file(f"[!] Checkpoint saved at {CHECKPOINT_FILE}")
                log_console_and_file("[!] Train Completed")
                return

        avg_loss = running_loss / len(train_loader)
        predictions = torch.argmax(output, dim=1).tolist()

        log_console_and_file(f"\n=== EPOCH {epoch+1}/{EPOCHS} ===")
        log_console_and_file(f"[LOSS] {avg_loss:.4f}")
        log_console_and_file(f"[PRED] {predictions[:10]} ...")
        visualize_fractal_tree(model.reflect())
        t_global += 1

    torch.save(model.state_dict(), CHECKPOINT_FILE)
    log_console_and_file(f"[!] Final checkpoint saved at {CHECKPOINT_FILE}")
    log_console_and_file("[!] Training completed successfully.")

if __name__ == "__main__":
    train_simple()