#!/usr/bin/env python3
"""
Expert Routing Weight Analysis across MOELoRA Experiments.

For each checkpoint, extracts:
  - lora_gate.GateL.weight  (8 x 64)   — gate weight matrix
  - lora_task_embedding.weight (3 x 64) — task embedding table
    index 0 = ??? (maybe a shared/default), 1 = IHD, 2 = Sarca

Routing = softmax(GateL_weight @ task_emb[task_id])
JS divergence measures how differently the two tasks route across experts.
"""

import os, re, math, warnings
from pathlib import Path
import torch
import torch.nn.functional as F
import numpy as np

warnings.filterwarnings("ignore")

# ─── Helpers ───────────────────────────────────────────────────────────────────

def kl_div(p, q):
    """KL(p || q) with numerical safety."""
    p = np.clip(p, 1e-12, 1.0)
    q = np.clip(q, 1e-12, 1.0)
    return float(np.sum(p * np.log(p / q)))

def js_divergence(p, q):
    """Jensen-Shannon divergence (symmetric, bounded [0, ln2])."""
    m = 0.5 * (p + q)
    return 0.5 * kl_div(p, m) + 0.5 * kl_div(q, m)

def cosine_similarity(a, b):
    """Cosine similarity between two 1-D numpy arrays."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

def load_routing(ckpt_path):
    """
    Load checkpoint and return:
      gate_w   : (8, 64) numpy
      task_emb : (3, 64) numpy
      routing_ihd   : softmax(gate_w @ task_emb[1])  — 8-dim probability
      routing_sarca : softmax(gate_w @ task_emb[2])  — 8-dim probability
      js_div   : JS divergence between the two routings
      cos_sim  : cosine similarity between task_emb[1] and task_emb[2]
    """
    ckpt = torch.load(ckpt_path, map_location="cpu")

    gate_w   = ckpt["lora_gate.GateL.weight"].float().numpy()    # (8, 64)
    task_emb = ckpt["lora_task_embedding.weight"].float().numpy() # (3, 64)

    emb_ihd   = task_emb[1]   # IHD
    emb_sarca = task_emb[2]   # Sarca

    logits_ihd   = gate_w @ emb_ihd      # (8,)
    logits_sarca = gate_w @ emb_sarca    # (8,)

    routing_ihd   = np.exp(logits_ihd)   / np.exp(logits_ihd).sum()
    routing_sarca = np.exp(logits_sarca) / np.exp(logits_sarca).sum()

    js = js_divergence(routing_ihd, routing_sarca)
    cos = cosine_similarity(emb_ihd, emb_sarca)

    return {
        "gate_w":        gate_w,
        "task_emb":      task_emb,
        "routing_ihd":   routing_ihd,
        "routing_sarca": routing_sarca,
        "logits_ihd":    logits_ihd,
        "logits_sarca":  logits_sarca,
        "js_div":        js,
        "cos_sim":       cos,
    }

def latest_checkpoint(exp_dir):
    """Return path to the highest-numbered checkpoint subdir."""
    ckpts = sorted(
        [d for d in Path(exp_dir).iterdir() if d.is_dir() and d.name.startswith("checkpoint-")],
        key=lambda d: int(d.name.split("-")[1])
    )
    return ckpts[-1] if ckpts else None

def all_checkpoints(exp_dir):
    """Return all checkpoint subdirs sorted by step number."""
    ckpts = sorted(
        [d for d in Path(exp_dir).iterdir() if d.is_dir() and d.name.startswith("checkpoint-")],
        key=lambda d: int(d.name.split("-")[1])
    )
    return ckpts

def fmt_routing(arr, width=7):
    """Format a probability array into a compact string."""
    return " ".join(f"{v:{width}.4f}" for v in arr)

def print_separator(char="═", length=140):
    print(char * length)

# ─── Experiment definitions ────────────────────────────────────────────────────

BASE = "/root/autodl-tmp/MOELoRA-peft-master/saved"

experiments = {
    "END1  (r32, step1000)": os.path.join(BASE, "END1_r32_lr2e-4_e8_bs16_step1000"),
    "END2  (r32, step1000)": os.path.join(BASE, "END2_r32_lr2e-4_e8_bs16_step1000"),
    "END9  (r64, step900)":  os.path.join(BASE, "END9_r64_lr2e-4_e8_bs16_step900"),
    "END11 (r32, step900)":  os.path.join(BASE, "END11_r32_lr2e-4_e8_bs16_step900"),
}

end11_dir = os.path.join(BASE, "END11_r32_lr2e-4_e8_bs16_step900")

# ─── Part 1: Cross-experiment comparison (latest checkpoints) ──────────────────

print()
print_separator("═")
print("  PART 1: EXPERT ROUTING COMPARISON — LATEST CHECKPOINT PER EXPERIMENT")
print_separator("═")
print()

header_experts = "  ".join(f"  Exp{i}" for i in range(8))
print(f"{'Experiment':<28s} │ {'Ckpt':>8s} │ Task   │ {header_experts} │ JS Div  │ Cos Sim")
print("─" * 28 + "─┼─" + "─" * 8 + "─┼─" + "─" * 6 + "─┼─" + "─" * (8*8-2) + "─┼─" + "─" * 7 + "─┼─" + "─" * 7)

results_all = {}

for name, exp_dir in experiments.items():
    ckpt_dir = latest_checkpoint(exp_dir)
    if ckpt_dir is None:
        print(f"{name:<28s} │ {'N/A':>8s} │ — no checkpoints found —")
        continue

    ckpt_path = ckpt_dir / "adapter_model.bin"
    step = ckpt_dir.name.split("-")[1]
    info = load_routing(ckpt_path)
    results_all[name] = info

    r_ihd   = fmt_routing(info["routing_ihd"],   8)
    r_sarca = fmt_routing(info["routing_sarca"], 8)

    print(f"{name:<28s} │ {step:>8s} │ IHD    │ {r_ihd} │ {info['js_div']:.5f} │ {info['cos_sim']:+.4f}")
    print(f"{'':28s} │ {'':>8s} │ Sarca  │ {r_sarca} │         │")
    print("─" * 28 + "─┼─" + "─" * 8 + "─┼─" + "─" * 6 + "─┼─" + "─" * (8*8-2) + "─┼─" + "─" * 7 + "─┼─" + "─" * 7)

# ─── Part 1b: Detailed per-experiment breakdown ───────────────────────────────

print()
print_separator("═")
print("  DETAILED PER-EXPERIMENT ANALYSIS")
print_separator("═")

for name, info in results_all.items():
    print()
    print(f"  ┌─ {name}")
    print(f"  │  Task Embedding shape: {info['task_emb'].shape}")
    print(f"  │  Gate Weight shape:    {info['gate_w'].shape}")
    print(f"  │")
    print(f"  │  Task Embedding [0] (shared?): {info['task_emb'][0][:8]}...")
    print(f"  │  Task Embedding [1] (IHD):     {info['task_emb'][1][:8]}...")
    print(f"  │  Task Embedding [2] (Sarca):   {info['task_emb'][2][:8]}...")
    print(f"  │")
    print(f"  │  Cosine(IHD emb, Sarca emb):      {info['cos_sim']:+.6f}")
    cos_0_1 = cosine_similarity(info['task_emb'][0], info['task_emb'][1])
    cos_0_2 = cosine_similarity(info['task_emb'][0], info['task_emb'][2])
    print(f"  │  Cosine(Emb[0], IHD emb):         {cos_0_1:+.6f}")
    print(f"  │  Cosine(Emb[0], Sarca emb):       {cos_0_2:+.6f}")
    print(f"  │")
    print(f"  │  Raw logits IHD:   {fmt_routing(info['logits_ihd'], 9)}")
    print(f"  │  Raw logits Sarca: {fmt_routing(info['logits_sarca'], 9)}")
    print(f"  │")
    print(f"  │  Routing  IHD:     {fmt_routing(info['routing_ihd'], 9)}")
    print(f"  │  Routing  Sarca:   {fmt_routing(info['routing_sarca'], 9)}")
    print(f"  │")
    print(f"  │  Max expert IHD:   Expert {np.argmax(info['routing_ihd'])} ({info['routing_ihd'].max():.4f})")
    print(f"  │  Max expert Sarca: Expert {np.argmax(info['routing_sarca'])} ({info['routing_sarca'].max():.4f})")
    print(f"  │  JS divergence:    {info['js_div']:.6f}  (max possible = {math.log(2):.6f} = ln2)")
    print(f"  └─")


# ─── Part 2: END11 routing evolution across ALL checkpoints ────────────────────

print()
print_separator("═")
print("  PART 2: END11 ROUTING EVOLUTION ACROSS ALL CHECKPOINTS (30 → 240)")
print_separator("═")
print()

end11_ckpts = all_checkpoints(end11_dir)

print(f"{'Step':>6s} │ Task   │ " + "  ".join(f"  Exp{i}" for i in range(8)) + " │ JS Div  │ Cos Sim │ Max Expert")
print("─" * 6 + "─┼─" + "─" * 6 + "─┼─" + "─" * (8*8-2) + "─┼─" + "─" * 7 + "─┼─" + "─" * 7 + "─┼─" + "─" * 10)

evolution_data = []

for ckpt_dir in end11_ckpts:
    step = int(ckpt_dir.name.split("-")[1])
    ckpt_path = ckpt_dir / "adapter_model.bin"
    info = load_routing(ckpt_path)
    evolution_data.append((step, info))

    r_ihd   = fmt_routing(info["routing_ihd"],   8)
    r_sarca = fmt_routing(info["routing_sarca"], 8)

    max_ihd   = f"E{np.argmax(info['routing_ihd'])}"
    max_sarca = f"E{np.argmax(info['routing_sarca'])}"

    print(f"{step:>6d} │ IHD    │ {r_ihd} │ {info['js_div']:.5f} │ {info['cos_sim']:+.4f} │ {max_ihd} ({info['routing_ihd'].max():.4f})")
    print(f"{'':>6s} │ Sarca  │ {r_sarca} │         │         │ {max_sarca} ({info['routing_sarca'].max():.4f})")
    print("─" * 6 + "─┼─" + "─" * 6 + "─┼─" + "─" * (8*8-2) + "─┼─" + "─" * 7 + "─┼─" + "─" * 7 + "─┼─" + "─" * 10)

# ─── Part 2b: Summary of END11 evolution trends ───────────────────────────────

print()
print_separator("─")
print("  END11 EVOLUTION SUMMARY")
print_separator("─")

steps    = [s for s, _ in evolution_data]
js_vals  = [info["js_div"] for _, info in evolution_data]
cos_vals = [info["cos_sim"] for _, info in evolution_data]

print(f"  Steps:     {steps}")
print(f"  JS div:    {['%.5f' % v for v in js_vals]}")
print(f"  Cos sim:   {['%+.4f' % v for v in cos_vals]}")
print()
print(f"  JS divergence range:  {min(js_vals):.6f} → {max(js_vals):.6f}")
print(f"  Cos similarity range: {min(cos_vals):+.6f} → {max(cos_vals):+.6f}")
print(f"  JS at step  30: {js_vals[0]:.6f}")
print(f"  JS at step 240: {js_vals[-1]:.6f}")
print(f"  JS trend:        {'↑ increasing (more task specialization)' if js_vals[-1] > js_vals[0] else '↓ decreasing (more uniform)' if js_vals[-1] < js_vals[0] else '→ stable'}")
print()

# Track which expert dominates per task over time
print("  Dominant expert per task over training:")
for step, info in evolution_data:
    dom_ihd   = np.argmax(info["routing_ihd"])
    dom_sarca = np.argmax(info["routing_sarca"])
    bar_ihd   = "█" * int(info["routing_ihd"].max() * 50)
    bar_sarca = "█" * int(info["routing_sarca"].max() * 50)
    print(f"    Step {step:>3d}:  IHD→E{dom_ihd} {bar_ihd} ({info['routing_ihd'].max():.3f})   "
          f"Sarca→E{dom_sarca} {bar_sarca} ({info['routing_sarca'].max():.3f})")


# ─── Part 3: Cross-experiment comparison summary ──────────────────────────────

print()
print_separator("═")
print("  PART 3: CROSS-EXPERIMENT COMPARISON SUMMARY")
print_separator("═")
print()

print(f"{'Experiment':<28s} │ JS Div  │ Cos(IHD,Sar) │ Dom.IHD │ Dom.Sar │ Specialization")
print("─" * 28 + "─┼─" + "─" * 7 + "─┼─" + "─" * 12 + "─┼─" + "─" * 7 + "─┼─" + "─" * 7 + "─┼─" + "─" * 14)

for name, info in results_all.items():
    dom_ihd   = f"E{np.argmax(info['routing_ihd'])}"
    dom_sarca = f"E{np.argmax(info['routing_sarca'])}"

    # Classify specialization level based on JS divergence
    js = info["js_div"]
    if js > 0.3:
        level = "HIGH ⬆"
    elif js > 0.1:
        level = "MEDIUM"
    elif js > 0.01:
        level = "LOW ⬇"
    else:
        level = "MINIMAL"

    print(f"{name:<28s} │ {js:.5f} │ {info['cos_sim']:+.6f}   │ {dom_ihd:<7s} │ {dom_sarca:<7s} │ {level}")

print()

# ─── Part 4: Entropy of routing distributions ─────────────────────────────────

print_separator("═")
print("  PART 4: ROUTING ENTROPY ANALYSIS (higher = more uniform, max = ln(8) ≈ 2.079)")
print_separator("═")
print()

def entropy(p):
    p = np.clip(p, 1e-12, 1.0)
    return -float(np.sum(p * np.log(p)))

max_ent = math.log(8)

print(f"{'Experiment':<28s} │ H(IHD)  │ H(Sarca) │ H(IHD)/Hmax │ H(Sar)/Hmax │ Interpretation")
print("─" * 28 + "─┼─" + "─" * 7 + "─┼─" + "─" * 8 + "─┼─" + "─" * 11 + "─┼─" + "─" * 11 + "─┼─" + "─" * 14)

for name, info in results_all.items():
    h_ihd   = entropy(info["routing_ihd"])
    h_sarca = entropy(info["routing_sarca"])

    norm_ihd   = h_ihd / max_ent
    norm_sarca = h_sarca / max_ent

    if min(norm_ihd, norm_sarca) > 0.9:
        interp = "Near-uniform"
    elif min(norm_ihd, norm_sarca) > 0.7:
        interp = "Moderate spread"
    elif min(norm_ihd, norm_sarca) > 0.4:
        interp = "Some concentration"
    else:
        interp = "Highly concentrated"

    print(f"{name:<28s} │ {h_ihd:.4f}  │ {h_sarca:.4f}   │ {norm_ihd:.4f}      │ {norm_sarca:.4f}      │ {interp}")

print()
print("  END11 entropy evolution:")
print(f"  {'Step':>6s} │ H(IHD)  │ H(Sarca) │ H(IHD)/Hmax │ H(Sar)/Hmax")
print("  " + "─" * 6 + "─┼─" + "─" * 7 + "─┼─" + "─" * 8 + "─┼─" + "─" * 11 + "─┼─" + "─" * 11)

for step, info in evolution_data:
    h_ihd   = entropy(info["routing_ihd"])
    h_sarca = entropy(info["routing_sarca"])
    print(f"  {step:>6d} │ {h_ihd:.4f}  │ {h_sarca:.4f}   │ {h_ihd/max_ent:.4f}      │ {h_sarca/max_ent:.4f}")

print()
print_separator("═")
print("  ANALYSIS COMPLETE")
print_separator("═")
print()
