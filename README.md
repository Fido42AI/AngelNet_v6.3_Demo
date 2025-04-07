# AngelNet_v6.3_Demo
AngelNet_v6.3_Demo - Fractal-based AI prototype demonstrating non-linear self-growth and decision layers.
# AngelNet v6.3_Demo — Fractal Intelligence of a New Generation

Author: Bohdan Fedorenko  
License 
Release Date: 2025-04-06  
Language: Python 3.10 ( just)
File Format: Partially `.py`, partially `.cpython-310` (auto-compiled and protected)

---

## Description

AngelNet is an experimental nonlinear intelligence model based on fractal self-growth, probabilistic learning, and memory adaptation. It challenges traditional linear computation (input → output → backpropagation) by introducing a dynamic, evolving neural structure.

---

## Key Features

- Fractal neuron growth — neurons split into substates based on confidence
- Fractal Bit — each unit tracks:
  - Historical state confidence
  - Collapse probability
  - Trigger to generate new branches

- State Update:
- s_new = Σ(w_i * s_i)
where:
w_i — weight of past state
s_i — value of state
- **Collapse Function**:
- P_collapse = e^(-λ * t)
λ — decay factor
t — time of state retention
- **Tree visualization** via `view_tree.py`
- **Runtime encryption** — key modules compiled into `.cpython-310` format (undecipherable)

---

## Architecture

- `angelnet_core_v6_4.py` — core engine
- `fractal_bit.py` — logic of fractal units
- `fractal_memory.py` — memory handling
- `neural_link_adapter_v2_3.py` — internal link adapter
- `output_adapter_v1.py` — signal output interface
- `view_tree.py` — tree structure visualization
- `train_simple_v6_3.py` — training starter

---

## How to Run

1. Install Python 3.10+
2. Clone the repository:
```bash
git clone https://github.com/yourname/angelnet.git
cd angelnet
3.	Install dependencies:
pip install -r requirements.txt
python train_simple_v6_3.py
What You Will See
	•	Fractal neuron growth
	•	2 epochs of training, then completion
	•	Logs with predictions and confidence
	•	Fractal Bit branching
	•	Auto-generated .cpython-310 files

Note: .cpython-310 files are encrypted versions of the logic modules. This is a Proof of Emergent Security Layer and intentional obfuscation.

⸻

Outcome

AngelNet v6.4 demonstrates the first testable precedent of nonlinear, self-organizing, fractal growth in neural architecture. Further developments will be published in upcoming versions.
Project: AngelNet
Author: Bohdan Fedorenko
Contact: contact@angel-net.tech
Initial Release Date: April 2025
