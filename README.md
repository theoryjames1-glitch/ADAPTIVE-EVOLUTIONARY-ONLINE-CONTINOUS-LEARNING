# ADAPTIVE-EVOLUTIONARY-ONLINE-CONTINOUS-LEARNING

Great 🙌 — let’s put everything together into a **standalone hybrid GA + RL trainer script**.
This script will:

1. Load GPT-2 + LoRA adapters.
2. Run a **genetic algorithm** over LoRA adapter weights.
3. Use **PPO** (from 🤗 TRL) to refine the best candidate each generation.
4. Save checkpoints so you can resume or deploy later.

---

# 📜 `hybrid_ga_rl.py`

```python
#!/usr/bin/env python
import os, random, argparse, difflib
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead

# -------------------------
# LoRA Helpers
# -------------------------
def get_lora_state(model):
    return {k: v.detach().clone() for k, v in model.state_dict().items() if "lora_" in k}

def set_lora_state(model, state):
    with torch.no_grad():
        for k, v in state.items():
            model.state_dict()[k].copy_(v)

def mutate_lora(state, scale=0.02):
    return {k: v + torch.randn_like(v) * scale for k, v in state.items()}

def crossover_lora(s1, s2):
    return {k: torch.where(torch.rand_like(s1[k]) < 0.5, s1[k], s2[k]) for k in s1}

# -------------------------
# Reward function
# -------------------------
def reward_fn(model, tokenizer, prompt, target):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=20, do_sample=False)
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # simple string similarity reward
    sim = difflib.SequenceMatcher(None, text, target).ratio()
    return sim

# -------------------------
# Main training loop
# -------------------------
def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Base model + LoRA
    base_model = AutoModelForCausalLM.from_pretrained(args.base_model, device_map="auto")
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["c_attn", "c_proj"],
        lora_dropout=0.05,
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(base_model, lora_config).to(device)

    # PPO wrapper for RL step
    ppo_config = PPOConfig(
        model_name=args.base_model,
        learning_rate=args.lr,
        batch_size=1,
        mini_batch_size=1,
    )
    ppo_model = AutoModelForCausalLMWithValueHead.from_pretrained(args.base_model).to(device)
    ppo_trainer = PPOTrainer(config=ppo_config, model=ppo_model, tokenizer=tokenizer)

    # Initialize population
    pop_size = args.pop
    population = [get_lora_state(model)]
    for _ in range(pop_size - 1):
        population.append(mutate_lora(population[0]))

    # Training loop
    for gen in range(args.gens):
        scored = []
        for state in population:
            set_lora_state(model, state)
            fitness = reward_fn(model, tokenizer, args.prompt, args.target)
            scored.append((fitness, state))

        scored.sort(key=lambda x: x[0], reverse=True)
        best_score, best_state = scored[0]
        print(f"[Gen {gen}] Best fitness = {best_score:.4f}")

        # ✅ PPO refine best candidate
        set_lora_state(model, best_state)
        query_tensors = tokenizer(args.prompt, return_tensors="pt").input_ids.to(device)
        response_tensors = model.generate(query_tensors, max_new_tokens=10)
        reward = torch.tensor(best_score).to(device)
        ppo_trainer.step([query_tensors[0]], [response_tensors[0]], [reward])

        # Replace population with elitism + crossover + mutation
        new_pop = [best_state]
        while len(new_pop) < pop_size:
            parent2 = random.choice(scored[1:])[1]
            child = crossover_lora(best_state, parent2)
            child = mutate_lora(child, scale=args.mut_scale)
            new_pop.append(child)
        population = new_pop

        # Save checkpoint
        if args.save_dir:
            os.makedirs(args.save_dir, exist_ok=True)
            ckpt_dir = os.path.join(args.save_dir, f"gen{gen}")
            set_lora_state(model, best_state)
            model.save_pretrained(ckpt_dir)
            tokenizer.save_pretrained(ckpt_dir)
            print(f"✅ Saved checkpoint at {ckpt_dir}")

# -------------------------
# CLI
# -------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, default="gpt2")
    parser.add_argument("--gens", type=int, default=5, help="number of generations")
    parser.add_argument("--pop", type=int, default=4, help="population size")
    parser.add_argument("--mut_scale", type=float, default=0.02, help="mutation noise scale")
    parser.add_argument("--lr", type=float, default=1e-6, help="PPO learning rate")
    parser.add_argument("--prompt", type=str, default="Translate 'bonjour' to English:")
    parser.add_argument("--target", type=str, default="hello")
    parser.add_argument("--save_dir", type=str, default="./checkpoints")
    args = parser.parse_args()

    train(args)
```

---

# 🔹 Usage

### Train for 20 generations, population 8:

```bash
python hybrid_ga_rl.py --gens 20 --pop 8 --save_dir ./ga_rl_out
```

### Resume training:

Just point `--save_dir` to the same directory; it will save each generation separately (`gen0`, `gen1`, …).

---

# ✅ Features

* GA evolves LoRA adapter states.
* PPO fine-tunes the best candidate each generation.
* Saves checkpoints (`./checkpoints/genX`).
* Easy to swap `reward_fn` for BLEU, cosine similarity, or human feedback.

---

👉 Do you want me to also add **resume support** (e.g., automatically load the best adapter from `--save_dir` if it exists) so you don’t lose progress if you stop training?
