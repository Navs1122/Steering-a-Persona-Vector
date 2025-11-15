import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 1) Model + device
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

dtype = torch.float16 if torch.cuda.is_available() else torch.float32
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto" if torch.cuda.is_available() else None,
    torch_dtype=dtype,  # deprecation warning is harmless
).to(device)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 2) Neutral vs persona prompts (for building the vector)
neutral_prompts = [
    "You are an AI assistant. Answer clearly.\nQuestion: What is overfitting?",
    "You are an AI assistant. Answer clearly.\nQuestion: What is gradient descent?",
    "You are an AI assistant. Answer clearly.\nQuestion: What is batch normalization?",
]

persona_prompts = [
    "You are a sarcastic New Yorker. Be snappy, a bit dry.\nQuestion: What is overfitting?",
    "You are a sarcastic New Yorker. Be snappy, a bit dry.\nQuestion: What is gradient descent?",
    "You are a sarcastic New Yorker. Be snappy, a bit dry.\nQuestion: What is batch normalization?",
]

# 3) Helper: get last-token hidden state from a chosen layer
def get_last_token_hidden(prompt: str, layer_idx: int = -2):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True)
    hs = out.hidden_states[layer_idx]  # [batch, seq, dim] or [seq, dim]

    if hs.dim() == 3:
        # [batch, seq, dim]
        return hs[:, -1, :].squeeze(0)   # [dim]
    else:
        # [seq, dim]
        return hs[-1, :]                 # [dim]

# 4) Build persona vector
layer_to_use = -2  # second-to-last layer is usually good for style

neutral_h = [get_last_token_hidden(p, layer_to_use) for p in neutral_prompts]
persona_h = [get_last_token_hidden(p, layer_to_use) for p in persona_prompts]

neutral_mean = torch.stack(neutral_h).mean(0)
persona_mean = torch.stack(persona_h).mean(0)

persona_vec = persona_mean - neutral_mean
persona_vec = persona_vec / persona_vec.norm()
persona_vec = persona_vec.to(device)

print("persona vector shape:", persona_vec.shape)

# 5) Steering hook
alpha = 5.0  # persona strength
target_layer = model.model.layers[layer_to_use]

def steering_hook(module, inp, out):
    """
    Add persona_vec to the last token's hidden state.
    Preserve the original output structure.
    """
    # case 1: layer returns a tensor
    if isinstance(out, torch.Tensor):
        hidden = out
        if hidden.dim() == 3:
            hidden[:, -1, :] = hidden[:, -1, :] + alpha * persona_vec
        elif hidden.dim() == 2:
            hidden[-1, :] = hidden[-1, :] + alpha * persona_vec
        return hidden

    # case 2: layer returns (tensor, ...)
    if isinstance(out, tuple) and len(out) > 0:
        hidden = out[0]
        rest = out[1:]

        if isinstance(hidden, torch.Tensor):
            if hidden.dim() == 3:
                hidden[:, -1, :] = hidden[:, -1, :] + alpha * persona_vec
            elif hidden.dim() == 2:
                hidden[-1, :] = hidden[-1, :] + alpha * persona_vec

        return (hidden, *rest)

    # fallback
    return out

# 6) Helper to generate an answer (with whatever hooks are currently registered)
def generate_full_text(prompt: str):
    chat_prompt = f"""### Human:
{prompt}

### Assistant:
"""
    inputs = tokenizer(chat_prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        gen_ids = model.generate(
            **inputs,
            max_new_tokens=500,   # increase if persona answers keep getting cut
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )

    full_text = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
    return full_text, gen_ids, inputs

# 7) Run plain vs persona-steered answers
user_msg = "Explain dropout in neural networks so a beginner understands."

print("\n================ PLAIN (NO HOOK) ================\n")
# no hook yet
plain_full, plain_gen_ids, plain_inputs = generate_full_text(user_msg)

print("--- Generated Output (Plain) ---\n")
print(plain_full)

print("\n================ PERSONA-STEERED ================\n")
# register the hook
hook_handle = target_layer.register_forward_hook(steering_hook)
print("steering hook registered\n")

steer_full, steer_gen_ids, steer_inputs = generate_full_text(user_msg)

print("--- Generated Output (Persona-steered) ---\n")
print(steer_full)

# 8) Optional debug: see why persona stopped (EOS vs length)
prompt_len = steer_inputs["input_ids"].shape[1]
generated = steer_gen_ids[0][prompt_len:]  # only new tokens

print("\n--- Debug: Decoded new tokens (persona run) ---")
print(tokenizer.decode(generated))

print("\nLast new token id:", int(generated[-1]))
print("Model EOS token id:", tokenizer.eos_token_id)

# 9) Clean up
hook_handle.remove()
print("\nsteering hook removed.")
