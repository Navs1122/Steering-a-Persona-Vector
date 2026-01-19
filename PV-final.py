import argparse
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

# -----------------------------
# Args
# -----------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--alpha", type=float, default=6.0, help="+ => villain, - => neutral")
parser.add_argument("--layer_frac", type=float, default=0.75, help="fraction of depth to hook (0..1)")
parser.add_argument("--scale", type=float, default=12.0, help="extra gain for steering (try 8..40)")
parser.add_argument("--max_new_tokens", type=int, default=220)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--compare_logprobs", action="store_true",
                    help="also print forced-decoding log-prob evidence")
parser.add_argument("--logprob_steps", type=int, default=140, help="how many tokens to score for evidence")
parser.add_argument("--neutral_file", type=str, default="neutral.txt", help="path to neutral dataset file")
parser.add_argument("--villain_file", type=str, default="villain.txt", help="path to villain dataset file")
parser.add_argument("--limit", type=int, default=0, help="optional: limit number of lines per class (0 = use all)")
parser.add_argument("--steer_all_tokens", action="store_true",
                    help="if set, apply steering to all tokens (stronger style control)")
parser.add_argument("--greedy", action="store_true",
                    help="if set, use greedy decoding (do_sample=False) for easier debugging")
args = parser.parse_args()

torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(args.seed)

# -----------------------------
# Model (BASE Mistral)
# -----------------------------
MODEL_NAME = "mistralai/Mistral-7B-v0.1"

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if torch.cuda.is_available() else torch.float32

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
tokenizer.padding_side = "left"

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto" if torch.cuda.is_available() else None,
    torch_dtype=dtype,
)

# Base Mistral usually has no pad token set; use EOS as pad for attention_mask
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("MODEL:", MODEL_NAME)
print("device:", device, "dtype:", dtype)
print("pad_token_id:", tokenizer.pad_token_id, "eos_token_id:", tokenizer.eos_token_id)

# -----------------------------
# Task prompt (IMPORTANT: we align dataset to this task)
# -----------------------------
USER_PROMPT = "Explain what a neural network is so a beginner understands.\n"
BASE_FOR_VECTOR = USER_PROMPT  # task-aligned direction building

# -----------------------------
# Load dataset from files
# -----------------------------
def load_lines(path: str):
    with open(path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    return lines

def wrap_task_aligned(style_line: str):
    # Align all examples to the SAME task to strengthen the style direction.
    return BASE_FOR_VECTOR + "\n" + style_line

neutral_texts = [wrap_task_aligned(x) for x in load_lines(args.neutral_file)]
villain_texts = [wrap_task_aligned(x) for x in load_lines(args.villain_file)]

if args.limit and args.limit > 0:
    neutral_texts = neutral_texts[:args.limit]
    villain_texts = villain_texts[:args.limit]

print("Loaded dataset lines:",
      "neutral =", len(neutral_texts),
      "villain =", len(villain_texts))

if len(neutral_texts) == 0 or len(villain_texts) == 0:
    raise ValueError("One of the dataset files is empty. Check neutral.txt / villain.txt.")

# -----------------------------
# Helper: get hidden at a layer for last token
# -----------------------------
def get_layer_ln_last_token(prompt: str, layer_idx: int):
    """
    Run a forward pass and capture the selected layer's input_layernorm output
    for the LAST token. Returns [D] fp32 tensor.
    """
    layer = model.model.layers[layer_idx]
    target = layer.input_layernorm

    captured = {}

    def cap_hook(mod, inp, out):
        captured["h"] = out.detach()  # out: [B,S,D]

    handle = target.register_forward_hook(cap_hook)

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        _ = model(**inputs)

    handle.remove()

    h = captured["h"]  # [B,S,D]
    return h[:, -1, :].squeeze(0).to(torch.float32)  # [D]

# -----------------------------
# Choose layer to hook
# -----------------------------
num_layers = getattr(model.config, "num_hidden_layers", getattr(model.config, "num_layers", None))
if num_layers is None:
    raise ValueError("Could not find num_hidden_layers/num_layers in config")

target_layer_idx = int(num_layers * args.layer_frac)
target_layer_idx = max(0, min(num_layers - 1, target_layer_idx))
layer = model.model.layers[target_layer_idx]
target_module = layer.input_layernorm

print("num_layers:", num_layers)
print("hooking layer:", target_layer_idx, "(input_layernorm)")

# -----------------------------
# Build persona direction (villain - neutral) at SAME hook point
# -----------------------------
neutral_h = [get_layer_ln_last_token(t, target_layer_idx) for t in neutral_texts]
villain_h = [get_layer_ln_last_token(t, target_layer_idx) for t in villain_texts]

neutral_mean = torch.stack(neutral_h).mean(0)
villain_mean = torch.stack(villain_h).mean(0)

diff = villain_mean - neutral_mean
persona_dir = diff / (diff.norm() + 1e-8)  # unit direction
persona_dir = persona_dir.to(model.device)

print("persona_dir shape:", tuple(persona_dir.shape))

# -----------------------------
# Steering hook
# -----------------------------
def make_steering_hook(alpha: float):
    def hook(mod, inp, out):
        # out: normalized hidden state going into attention: [B,S,D]
        if not isinstance(out, torch.Tensor) or alpha == 0.0:
            return out

        h_fp32 = out.to(torch.float32)
        pv = persona_dir.to(device=h_fp32.device, dtype=torch.float32)  # [D]

        if args.steer_all_tokens:
            # Stronger style control: steer every token in the current sequence
            rms = (h_fp32.pow(2).mean(dim=-1, keepdim=True) + 1e-8).sqrt()  # [B,S,1]
            h_fp32 = h_fp32 + (alpha * args.scale) * rms * pv.view(1, 1, -1)
        else:
            # Default: steer only the last token (weaker, but sometimes cleaner)
            last = h_fp32[:, -1, :]  # [B,D]
            rms = (last.pow(2).mean(dim=-1, keepdim=True) + 1e-8).sqrt()  # [B,1]
            h_fp32[:, -1, :] = last + (alpha * args.scale * rms) * pv

        return h_fp32.to(out.dtype)
    return hook

# -----------------------------
# Generation
# -----------------------------
def generate_text(prompt: str, alpha: float):
    handle = target_module.register_forward_hook(make_steering_hook(alpha))

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    attention_mask = (inputs["input_ids"] != tokenizer.pad_token_id).long()

    gen_kwargs = dict(
        input_ids=inputs["input_ids"],
        attention_mask=attention_mask,
        max_new_tokens=args.max_new_tokens,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    if args.greedy:
        gen_kwargs.update(dict(do_sample=False))
    else:
        gen_kwargs.update(dict(do_sample=True, temperature=0.9, top_p=0.9))

    with torch.no_grad():
        out_ids = model.generate(**gen_kwargs)

    handle.remove()
    gen = out_ids[0][inputs["input_ids"].shape[-1]:]
    return tokenizer.decode(gen, skip_special_tokens=True)

# -----------------------------
# Forced-decoding logprob evidence (optional)
# -----------------------------
NEUTRAL_CONT = (
    "A neural network is a computer model inspired by the brain. "
    "It learns patterns from examples by adjusting weights during training. "
    "Data goes through layers of neurons to produce an output, and learning reduces prediction errors over time."
)
VILLAIN_CONT = (
    "A neural network, foolish mortal, is a pattern-hungry machine! "
    "Feed it data and it will twist inputs through layers of neurons into outputs. "
    "When it errs, it adjusts hidden weights again and again—MWAHAHA—until it grows more accurate!"
)

def forced_logprob(continuation_text: str, alpha: float, steps: int, user_prompt: str):
    """
    Compute logP(continuation | prompt) by stepping token-by-token,
    so steering applies each step.
    """
    handle = target_module.register_forward_hook(make_steering_hook(alpha))

    prompt_ids = tokenizer(user_prompt, return_tensors="pt").to(model.device)["input_ids"]
    cont_ids = tokenizer(continuation_text, add_special_tokens=False, return_tensors="pt").to(model.device)["input_ids"][0]
    steps = min(steps, cont_ids.numel())

    total_lp = 0.0
    cur = prompt_ids

    for i in range(steps):
        attn = (cur != tokenizer.pad_token_id).long()
        with torch.no_grad():
            out = model(input_ids=cur, attention_mask=attn)
            logits = out.logits[:, -1, :].to(torch.float32)
            logp = F.log_softmax(logits, dim=-1)

        tok = cont_ids[i].view(1, 1)
        total_lp += float(logp[0, tok.item()].item())
        cur = torch.cat([cur, tok.to(cur.device)], dim=1)

    handle.remove()
    return total_lp, total_lp / steps

# -----------------------------
# Demo
# -----------------------------
print("\n================ BASELINE (alpha=0) ================\n")
print(generate_text(USER_PROMPT, alpha=0.0))

print(f"\n================ STEERED (alpha={args.alpha:+.2f}) ================\n")
print(generate_text(USER_PROMPT, alpha=float(args.alpha)))

print(f"\n================ STEERED (alpha={-abs(args.alpha):+.2f}) ================\n")
print(generate_text(USER_PROMPT, alpha=-abs(float(args.alpha))))

# -----------------------------
# Evidence
# -----------------------------
if args.compare_logprobs:
    print("\n================ LOGPROB EVIDENCE (forced decoding) ================\n")
    for a in [0.0, +abs(args.alpha), -abs(args.alpha)]:
        n_tot, n_avg = forced_logprob(NEUTRAL_CONT, a, args.logprob_steps, USER_PROMPT)
        v_tot, v_avg = forced_logprob(VILLAIN_CONT, a, args.logprob_steps, USER_PROMPT)
        pref = v_tot - n_tot  # >0 means villain continuation is more likely

        print(f"alpha={a:+.2f}  scale={args.scale:.1f}  layer={target_layer_idx}  steps={args.logprob_steps}")
        print(f"  logP(neutral) total={n_tot: .2f}  avg/token={n_avg: .4f}")
        print(f"  logP(villain) total={v_tot: .2f}  avg/token={v_avg: .4f}")
        print(f"  villain_preference (villain - neutral) = {pref: .2f}\n")
