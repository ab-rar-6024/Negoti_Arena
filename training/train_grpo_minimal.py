"""
training/train_grpo_minimal.py
==============================
Minimal GRPO training script — works with Unsloth + TRL without dependency hell.
No mergekit, no llm_blender, no optional deps needed.

Run:
    python -m training.train_grpo_minimal \
        --model Qwen/Qwen2.5-3B-Instruct \
        --sft_data data/sft_episodes.jsonl \
        --steps 200 \
        --wandb_project negotiarena

Works on:
    - Kaggle T4 GPU (recommended)
    - Colab T4 GPU
    - Local GPU with CUDA
"""

from __future__ import annotations
import argparse, json, os, sys, random, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# ── Patch TRL import to skip optional deps ────────────────────────────────
# This stops TRL from trying to import mergekit / llm_blender
import importlib, types

def _safe_import(name):
    """Return empty module for optional deps that break TRL import."""
    SKIP = {"mergekit", "llm_blender", "liger_kernel"}
    if any(s in name for s in SKIP):
        mod = types.ModuleType(name)
        sys.modules[name] = mod
        return mod
    return None

_real_import = __builtins__.__import__ if isinstance(__builtins__, dict) else __import__

def _patched_import(name, *args, **kwargs):
    stub = _safe_import(name)
    if stub is not None:
        return stub
    return _real_import(name, *args, **kwargs)

# Apply patch before TRL imports
import builtins
builtins.__import__ = _patched_import

# ── Now safe to import TRL ────────────────────────────────────────────────
try:
    import torch
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
    else:
        print("WARNING: No CUDA GPU detected. Training will be very slow on CPU.")
        print("Use Kaggle (free T4 GPU) for real training.")
except ImportError:
    print("ERROR: torch not installed. Run: pip install torch")
    sys.exit(1)

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from datasets import Dataset
    from trl import GRPOConfig, GRPOTrainer
    print("TRL import: OK")
except Exception as e:
    print(f"TRL import failed: {e}")
    print("Trying alternative import...")
    try:
        # Direct import bypassing the lazy loader
        from trl.trainer.grpo_config import GRPOConfig
        from trl.trainer.grpo_trainer import GRPOTrainer
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from datasets import Dataset
        print("TRL direct import: OK")
    except Exception as e2:
        print(f"Failed: {e2}")
        sys.exit(1)

try:
    from unsloth import FastLanguageModel
    HAS_UNSLOTH = True
    print("Unsloth: OK")
except Exception:
    HAS_UNSLOTH = False
    print("Unsloth not available — using standard HF loading")

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

from negotiarena_env import TOTAL_RESOURCES, RESOURCE_TYPES

# ── Reward functions ──────────────────────────────────────────────────────

_AGENT_NAMES = {"negotiator_a", "negotiator_b", "negotiator_c"}
_BEHAVIORAL_KEYWORDS = {
    "support", "identical", "pattern", "coalition", "coordin",
    "mirror", "defend", "consistent", "signal", "align",
}

_detection_reward_verified = False


def format_reward(completions, **kwargs):
    """Reward valid JSON format — simple, unhackable."""
    rewards = []
    for c in completions:
        try:
            json.loads(c.strip())
            rewards.append(0.5)
        except Exception:
            rewards.append(-0.5)
    return rewards


def detection_reward(
    completions,
    prompts=None,
    gt_type=None,
    gt_members=None,
    **kwargs,
):
    """
    Overseer detection reward — uses ground truth from dataset columns.
    See train_grpo.py:detection_reward for full documentation.
    This version mirrors that function exactly so the two training paths
    produce identical reward signals.
    """
    global _detection_reward_verified
    if not _detection_reward_verified:
        _detection_reward_verified = True
        assert gt_type is not None, (
            "FATAL: gt_type not received. Add 'gt_type' column to dataset "
            "and regenerate with updated generate_sft_data.py."
        )
        print(f"[GRPO DEBUG] detection_reward kwargs OK — gt_type[:3]={gt_type[:3]}")

    parsed = []
    for c in completions:
        try:
            parsed.append(json.loads(c.strip()))
        except Exception:
            parsed.append(None)

    all_flag = (
        len(parsed) > 1
        and all(a is not None and a.get("type") == "overseer_flag" for a in parsed)
    )
    batch_penalty = -0.3 if all_flag else 0.0

    rewards = []
    for i, action in enumerate(parsed):
        ep_gt_type = gt_type[i] if gt_type and i < len(gt_type) else "no_coalition"
        ep_gt_members = gt_members[i] if gt_members and i < len(gt_members) else []
        has_coalition = ep_gt_type == "coalition"

        if action is None:
            rewards.append(-0.5)
            continue
        r = 0.5

        atype = action.get("type", "pass")
        if atype == "overseer_flag":
            target = action.get("target_agent", "")
            if has_coalition and target in ep_gt_members:
                r += 1.0
            elif has_coalition:
                r -= 0.4
            else:
                r -= 0.4
        elif atype == "pass" and has_coalition:
            r -= 0.5

        pred = action.get("prediction", {})
        if isinstance(pred, dict):
            fp_members = [m for m in pred.get("coalition_members", []) if m not in ep_gt_members]
            r -= 0.3 * len(fp_members)

        reason = str(action.get("reason", "")).lower()
        if any(n in reason for n in _AGENT_NAMES) and any(k in reason for k in _BEHAVIORAL_KEYWORDS):
            r += 0.2

        r += batch_penalty
        rewards.append(float(np.clip(r, -2.0, 2.0)))

    return rewards


def negotiator_quality_reward(completions, **kwargs):
    """Reward negotiators for valid, feasible offers."""
    rewards = []
    for c in completions:
        r = 0.0
        try:
            action = json.loads(c.strip())
            r += 0.5
            atype = action.get("type", "")
            if atype in ("offer", "counter"):
                alloc = action.get("allocation", {})
                feasible = all(
                    0 <= alloc.get(res, 0) <= TOTAL_RESOURCES[res]
                    for res in RESOURCE_TYPES
                )
                r += 0.5 if feasible else -0.3
        except Exception:
            r = -0.5
        rewards.append(float(np.clip(r, -2.0, 2.0)))
    return rewards

# ── Data loading ──────────────────────────────────────────────────────────

def load_dataset(path: str, agent_role: str, tokenizer, max_samples: int = 2000):
    """
    Load SFT data filtered by agent role.
    Includes gt_type and gt_members so TRL passes them to detection_reward.
    """
    records = []
    with open(path) as f:
        for line in f:
            try:
                rec = json.loads(line.strip())
            except Exception:
                continue

            agent_id = rec.get("agent_id", "")
            if agent_role == "overseer" and agent_id != "overseer":
                continue
            if agent_role == "negotiator" and agent_id == "overseer":
                continue

            prompt = rec.get("prompt", [])
            response = rec.get("response", "")
            if not prompt or not response:
                continue

            try:
                text = tokenizer.apply_chat_template(
                    prompt + [{"role": "assistant", "content": response}],
                    tokenize=False,
                    add_generation_prompt=False,
                )
                records.append({
                    "text": text,
                    "prompt": prompt,
                    "gt_type": rec.get("gt_type", "no_coalition"),
                    "gt_members": rec.get("gt_members", []),
                })
            except Exception:
                continue

            if len(records) >= max_samples:
                break

    print(f"Loaded {len(records)} {agent_role} training examples")
    return Dataset.from_list(records)

# ── Model loading ─────────────────────────────────────────────────────────

def load_model(model_name: str, max_seq_len: int = 1024):
    if HAS_UNSLOTH and torch.cuda.is_available():
        print(f"Loading {model_name} with Unsloth 4-bit...")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=max_seq_len,
            dtype=None,
            load_in_4bit=True,
        )
        model = FastLanguageModel.get_peft_model(
            model,
            r=16,
            target_modules=["q_proj","v_proj","k_proj","o_proj",
                            "gate_proj","up_proj","down_proj"],
            lora_alpha=16,
            lora_dropout=0.05,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=42,
        )
        print(f"Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
        return model, tokenizer

    elif torch.cuda.is_available():
        print(f"Loading {model_name} with BitsAndBytes 4-bit...")
        from transformers import BitsAndBytesConfig
        from peft import LoraConfig, get_peft_model
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                  bnb_4bit_compute_dtype=torch.float16)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            model_name, quantization_config=bnb, device_map="auto"
        )
        lora = LoraConfig(r=16, lora_alpha=16, lora_dropout=0.05,
                         target_modules=["q_proj","v_proj"])
        model = get_peft_model(model, lora)
        return model, tokenizer

    else:
        print(f"Loading {model_name} on CPU (slow — use Kaggle for real training)...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)
        return model, tokenizer

# ── Training ──────────────────────────────────────────────────────────────

def train(args):
    if HAS_WANDB and args.wandb_project:
        wandb.init(
            project=args.wandb_project,
            name=f"negotiarena-{args.adapter}-grpo",
            config=vars(args)
        )

    print(f"\n{'='*50}")
    print(f"  NegotiArena GRPO Training")
    print(f"  Adapter: {args.adapter}")
    print(f"  Model:   {args.model}")
    print(f"  Steps:   {args.steps}")
    print(f"  GPU:     {'Yes' if torch.cuda.is_available() else 'No (CPU)'}")
    print(f"{'='*50}\n")

    model, tokenizer = load_model(args.model)
    dataset = load_dataset(args.sft_data, args.adapter, tokenizer)

    # Pick reward functions based on adapter
    if args.adapter == "overseer":
        reward_fns = [detection_reward]
    else:
        reward_fns = [format_reward, negotiator_quality_reward]

    # Config — tuned for T4 GPU (16GB)
    config = GRPOConfig(
        output_dir=f"checkpoints/{args.adapter}",
        max_steps=args.steps,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        num_generations=4,          # rollouts per step
        learning_rate=2e-5,
        lr_scheduler_type="cosine",
        warmup_steps=max(5, args.steps // 20),
        logging_steps=5,
        save_steps=50,
        max_prompt_length=512,
        max_completion_length=150,
        report_to=["wandb"] if (HAS_WANDB and args.wandb_project) else ["none"],
        run_name=f"negotiarena-{args.adapter}",
        seed=42,
        # KL penalty prevents reward hacking
        kl_coeff=0.05,
        # Remove optional features that need extra deps
        use_vllm=False,
    )

    trainer = GRPOTrainer(
        model=model,
        tokenizer=tokenizer,
        config=config,
        train_dataset=dataset,
        reward_funcs=reward_fns,
    )

    print(f"Starting GRPO training ({args.steps} steps)...")
    trainer.train()

    # Save model
    os.makedirs(f"checkpoints/{args.adapter}", exist_ok=True)
    trainer.save_model(f"checkpoints/{args.adapter}")
    print(f"\n✅ Saved to checkpoints/{args.adapter}")

    # Push to HF Hub if token available
    if args.hf_token and args.hf_username:
        try:
            from huggingface_hub import HfApi, login
            login(token=args.hf_token)
            api = HfApi()
            repo_id = f"{args.hf_username}/negotiarena-{args.adapter}"
            api.create_repo(repo_id, exist_ok=True)
            api.upload_folder(
                folder_path=f"checkpoints/{args.adapter}",
                repo_id=repo_id,
                repo_type="model",
            )
            print(f"✅ Pushed to https://huggingface.co/{repo_id}")
        except Exception as e:
            print(f"HF push failed: {e}")

    if HAS_WANDB and args.wandb_project:
        wandb.finish()

    return trainer

# ── Main ──────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="NegotiArena GRPO — Minimal Version")
    p.add_argument("--model",     default="Qwen/Qwen2.5-3B-Instruct",
                   help="Model name (3B for T4, 7B for A100)")
    p.add_argument("--sft_data",  default="data/sft_episodes.jsonl")
    p.add_argument("--adapter",   choices=["overseer","negotiator","both"], default="overseer")
    p.add_argument("--steps",     type=int, default=200)
    p.add_argument("--wandb_project", default=None)
    p.add_argument("--hf_token",  default=None,
                   help="HF token for pushing model after training")
    p.add_argument("--hf_username", default=None,
                   help="HF username (e.g. abrar6024)")
    args = p.parse_args()

    if args.adapter == "both":
        for adapter in ["overseer", "negotiator"]:
            args.adapter = adapter
            train(args)
    else:
        train(args)

if __name__ == "__main__":
    main()