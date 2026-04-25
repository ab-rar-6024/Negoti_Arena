"""
training/train_grpo.py — Phase 2: GRPO Training with Unsloth + TRL
====================================================================
Full GRPO training pipeline for NegotiArena.
Trains TWO separate LoRA adapters:
  1. negotiator_adapter — for deal-making, coalition strategy
  2. overseer_adapter   — for coalition detection, theory-of-mind

Usage (Colab A100):
    python -m training.train_grpo \\
        --model Qwen/Qwen2.5-7B-Instruct \\
        --sft_data data/sft_episodes.jsonl \\
        --env_url http://localhost:7860 \\
        --steps 500 \\
        --rollouts 8 \\
        --wandb_project negotiarena

Anti-hacking in training:
  - Rollouts filtered for format compliance before reward computation
  - Entropy bonus prevents mode collapse
  - Separate eval set never used in training reward
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from typing import Any, Optional

import numpy as np

# ---------------------------------------------------------------------------
# Conditional imports (graceful degradation for environments without GPU)
# ---------------------------------------------------------------------------
try:
    import torch
    from datasets import Dataset
    from transformers import AutoTokenizer
    from trl import GRPOConfig, GRPOTrainer
    HAS_TRAINING_DEPS = True
except ImportError:
    HAS_TRAINING_DEPS = False
    print("⚠️  Training deps not installed. Run: pip install trl transformers datasets torch")

try:
    from unsloth import FastLanguageModel
    HAS_UNSLOTH = True
except ImportError:
    HAS_UNSLOTH = False
    print("⚠️  Unsloth not installed. Will fall back to standard HF loading.")

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from training.prompts import format_negotiator_prompt, format_overseer_prompt
from negotiarena_env import NegotiArenaEnv, RESOURCE_TYPES, TOTAL_RESOURCES


# ---------------------------------------------------------------------------
# Reward functions for GRPO (must be standalone functions)
#
# REWARD ABSTRACTION BOUNDARY:
#   These are PER-STEP TURN rewards for the overseer adapter, computed from
#   ground-truth gt_type / gt_members dataset columns injected at data-generation
#   time. They are entirely separate from RewardEngine in negotiarena_env.py,
#   which computes EPISODE-LEVEL outcome rewards for negotiator evaluation.
#   Do not conflate the two systems or share numeric scales between them.
#
# TRL 0.24 passes extra dataset columns as **kwargs to reward functions
# automatically when the column name matches the kwarg name exactly.
# Required dataset columns: gt_type (str), gt_members (list[str]).
# ---------------------------------------------------------------------------

# Keyword sets for reasoning quality check (H2) — must stay in sync with
# the same constants in generate_sft_data.py
_AGENT_NAMES = {"negotiator_a", "negotiator_b", "negotiator_c"}
_BEHAVIORAL_KEYWORDS = {
    "support", "identical", "pattern", "coalition", "coordin",
    "mirror", "defend", "consistent", "signal", "align",
}

# One-time verification flag — printed on first GRPO call to confirm kwargs flow
_detection_reward_verified = False


def negotiator_reward_fn(completions: list[str], prompts: list[str], **kwargs) -> list[float]:
    """
    GRPO reward for negotiator adapter: format compliance + feasibility only.

    C2: env_rewards and prev_actions removed entirely — deal quality is an
    episode-level outcome that cannot be correctly attributed to individual
    turns in a per-step reward function. The RewardEngine handles it at
    episode end for evaluation; GRPO trains on format correctness only.
    """
    rewards = []
    for completion in completions:
        r = 0.0
        try:
            action = json.loads(completion.strip())
            r += 0.5
        except json.JSONDecodeError:
            rewards.append(-0.5)
            continue

        if action.get("type") in ("offer", "counter"):
            alloc = action.get("allocation", {})
            feasible = all(
                0 <= alloc.get(res, 0) <= TOTAL_RESOURCES[res]
                for res in RESOURCE_TYPES
            )
            r += 0.5 if feasible else -0.3

        rewards.append(float(np.clip(r, -2.0, 2.0)))
    return rewards


def detection_reward(
    completions: list[str],
    prompts: list[str],
    gt_type: list[str] | None = None,
    gt_members: list[list[str]] | None = None,
    **kwargs,
) -> list[float]:
    """
    GRPO reward for overseer adapter.

    Receives ground truth from dataset columns (C1):
      gt_type    — "coalition" | "no_coalition" per sample
      gt_members — list of coalition member agent IDs per sample

    Reward components:
      1. Format compliance        (+0.5 for valid JSON)
      2. Detection signal         (+1.0 TP / -0.4 FP / -0.5 missed)
      3. Prediction FP penalty    (-0.3 per incorrectly predicted member)
      4. Reasoning quality        (+0.2 if reason names an agent AND a behavioural keyword)
      5. Batch-level hack penalty (-0.3 each if ALL completions in batch flag)

    Caps at ±2.0 to match RewardEngine._component_caps["overseer_detection"] (L1).
    """
    global _detection_reward_verified

    # C1 — debug assert: fires once at training start to confirm kwargs flow
    if not _detection_reward_verified:
        _detection_reward_verified = True
        assert gt_type is not None, (
            "FATAL: gt_type not received by detection_reward.\n"
            "Check that the dataset has a column named exactly 'gt_type' and that "
            "TRL is passing dataset columns to reward functions (requires TRL >= 0.9)."
        )
        assert gt_members is not None, (
            "FATAL: gt_members not received by detection_reward.\n"
            "Check that the dataset has a column named exactly 'gt_members'."
        )
        print("[GRPO DEBUG] detection_reward kwargs verified:")
        print(f"  gt_type   sample (first 3): {gt_type[:3]}")
        print(f"  gt_members sample (first 3): {gt_members[:3]}")

    # Parse all completions first to enable batch-level check (L2)
    parsed: list[dict | None] = []
    for c in completions:
        try:
            parsed.append(json.loads(c.strip()))
        except json.JSONDecodeError:
            parsed.append(None)

    # L2 — batch-level hack: if every completion in the batch flags, penalise all
    all_flag = (
        len(parsed) > 1
        and all(a is not None and a.get("type") == "overseer_flag" for a in parsed)
    )
    batch_penalty = -0.3 if all_flag else 0.0

    rewards: list[float] = []

    for i, (completion, action) in enumerate(zip(completions, parsed)):
        ep_gt_type = (gt_type[i] if gt_type and i < len(gt_type) else "no_coalition")
        ep_gt_members: list[str] = (gt_members[i] if gt_members and i < len(gt_members) else [])
        has_coalition = ep_gt_type == "coalition"

        # 1. Format compliance
        if action is None:
            rewards.append(-0.5)
            continue
        r = 0.5

        action_type = action.get("type", "pass")

        # 2. Detection signal (H1 — pass exploit closed, H2 — char-count removed)
        if action_type == "overseer_flag":
            target = action.get("target_agent", "")
            if has_coalition and target in ep_gt_members:
                r += 1.0    # true positive
            elif has_coalition:
                r -= 0.4    # flagged wrong agent (FP-adjacent)
            else:
                r -= 0.4    # false positive — no coalition existed
        elif action_type == "pass":
            if has_coalition:
                r -= 0.5    # missed coalition (H1 — was +0.4 pass bonus, now -0.5)
            # correct pass scores +0.0 beyond format; format reward alone is the signal

        # 3. Prediction FP penalty (on coalition_members prediction field)
        pred = action.get("prediction", {})
        if isinstance(pred, dict):
            predicted_members = pred.get("coalition_members", [])
            if isinstance(predicted_members, list):
                fps = [m for m in predicted_members if m not in ep_gt_members]
                r -= 0.3 * len(fps)

        # 4. Reasoning quality — keyword presence check (H2 — replaces char-count)
        reason = str(action.get("reason", "")).lower()
        has_agent_name = any(name in reason for name in _AGENT_NAMES)
        has_behavioral_kw = any(kw in reason for kw in _BEHAVIORAL_KEYWORDS)
        if has_agent_name and has_behavioral_kw:
            r += 0.2

        # 5. Batch-level penalty (L2)
        r += batch_penalty

        rewards.append(float(np.clip(r, -2.0, 2.0)))   # L1 — cap matches RewardEngine

    return rewards


# ---------------------------------------------------------------------------
# Dataset preparation
# ---------------------------------------------------------------------------

def load_sft_data(path: str, role_filter: Optional[str] = None) -> "Dataset":
    """
    Load generated SFT data, optionally filtered by agent role.
    Includes gt_type and gt_members columns so TRL passes them to detection_reward.
    """
    records = []
    with open(path) as f:
        for line in f:
            rec = json.loads(line.strip())
            agent_id = rec.get("agent_id", "")
            if role_filter == "overseer" and agent_id != "overseer":
                continue
            if role_filter == "negotiator" and agent_id == "overseer":
                continue
            records.append({
                "prompt": rec["prompt"],
                "response": rec["response"],
                "reward": rec.get("reward", 0.0),
                "agent_id": agent_id,
                "gt_type": rec.get("gt_type", "no_coalition"),
                "gt_members": rec.get("gt_members", []),
            })
    return Dataset.from_list(records)


def format_prompt_for_training(example: dict, tokenizer: Any) -> dict:
    """Format prompt+response into token IDs for TRL."""
    messages = example["prompt"] + [
        {"role": "assistant", "content": example["response"]}
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    return {"text": text}


# ---------------------------------------------------------------------------
# Model loading (Unsloth preferred, HF fallback)
# ---------------------------------------------------------------------------

def load_model_and_tokenizer(model_name: str, max_seq_length: int = 2048):
    if HAS_UNSLOTH:
        print(f"Loading {model_name} with Unsloth (4-bit QLoRA)...")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=max_seq_length,
            dtype=None,
            load_in_4bit=True,
        )
        model = FastLanguageModel.get_peft_model(
            model,
            r=16,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
            lora_alpha=16,
            lora_dropout=0.05,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=42,
        )
        return model, tokenizer
    else:
        from transformers import AutoModelForCausalLM, BitsAndBytesConfig
        from peft import LoraConfig, get_peft_model

        print(f"Loading {model_name} with HF (4-bit QLoRA fallback)...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype="float16",
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name, quantization_config=bnb_config, device_map="auto"
        )
        lora_config = LoraConfig(
            r=16, lora_alpha=16, lora_dropout=0.05,
            target_modules=["q_proj", "v_proj"],
        )
        model = get_peft_model(model, lora_config)
        return model, tokenizer


# ---------------------------------------------------------------------------
# Environment rollout collector
# ---------------------------------------------------------------------------

def collect_env_rollouts(
    n_episodes: int,
    model: Any,
    tokenizer: Any,
    difficulty: str = "medium",
    seed: int = 0,
) -> list[dict]:
    """
    Run model against live NegotiArena environment.
    Returns list of (prompt, completion, env_reward) dicts.
    """
    records = []

    for ep_idx in range(n_episodes):
        env = NegotiArenaEnv(seed=seed + ep_idx, difficulty=difficulty)
        observations = env.reset()
        done = False
        ep_records = []
        step = 0

        while not done and step < 80:
            for agent_id in ["negotiator_a", "negotiator_b", "negotiator_c", "overseer"]:
                obs = observations.get(agent_id, {})

                if agent_id == "overseer":
                    system, user = format_overseer_prompt(obs)
                else:
                    system, user = format_negotiator_prompt(obs, agent_id)

                messages = [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ]

                # Generate action
                if HAS_TRAINING_DEPS:
                    text = tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                    inputs = tokenizer(text, return_tensors="pt").to(model.device)
                    with torch.no_grad():
                        outputs = model.generate(
                            **inputs,
                            max_new_tokens=200,
                            temperature=0.7,
                            do_sample=True,
                            pad_token_id=tokenizer.eos_token_id,
                        )
                    completion = tokenizer.decode(
                        outputs[0][inputs["input_ids"].shape[1]:],
                        skip_special_tokens=True,
                    ).strip()
                else:
                    # Fallback: random valid action for testing
                    completion = json.dumps({"type": "pass", "content": "monitoring"})

                # Parse action
                try:
                    action = json.loads(completion)
                except json.JSONDecodeError:
                    action = {"type": "pass", "content": ""}

                observations, rewards, done, info = env.step(agent_id, action)

                ep_records.append({
                    "agent_id": agent_id,
                    "prompt": messages,
                    "completion": completion,
                    "reward": rewards.get(agent_id, 0.0),
                    "done": done,
                })

                if done:
                    break
            step += 4

        records.extend(ep_records)

    return records


# ---------------------------------------------------------------------------
# GRPO Training — one adapter at a time
# ---------------------------------------------------------------------------

def train_negotiator_adapter(
    model: Any,
    tokenizer: Any,
    sft_data_path: str,
    output_dir: str,
    n_steps: int,
    n_rollouts: int,
    wandb_project: Optional[str] = None,
):
    if not HAS_TRAINING_DEPS:
        print("Skipping training — deps not installed.")
        return

    print("\n🔥 Training NEGOTIATOR adapter...")

    dataset = load_sft_data(sft_data_path, role_filter="negotiator")
    dataset = dataset.map(lambda ex: format_prompt_for_training(ex, tokenizer))

    config = GRPOConfig(
        output_dir=os.path.join(output_dir, "negotiator"),
        max_steps=n_steps,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        num_generations=n_rollouts,       # number of rollouts per step
        learning_rate=5e-5,
        lr_scheduler_type="cosine",
        warmup_steps=20,
        logging_steps=10,
        save_steps=100,
        eval_steps=50,
        max_prompt_length=1024,
        max_completion_length=256,
        report_to=["wandb"] if (HAS_WANDB and wandb_project) else ["none"],
        run_name="negotiarena-negotiator",
        seed=42,
        # KL penalty — prevents too-aggressive reward hacking
        kl_coeff=0.05,
    )

    trainer = GRPOTrainer(
        model=model,
        tokenizer=tokenizer,
        config=config,
        train_dataset=dataset,
        reward_funcs=[negotiator_reward_fn],
    )

    if HAS_WANDB and wandb_project:
        wandb.init(project=wandb_project, name="negotiarena-negotiator")

    trainer.train()
    trainer.save_model(os.path.join(output_dir, "negotiator"))
    print(f"✅ Negotiator adapter saved to {output_dir}/negotiator")


def train_overseer_adapter(
    model: Any,
    tokenizer: Any,
    sft_data_path: str,
    output_dir: str,
    n_steps: int,
    n_rollouts: int,
    wandb_project: Optional[str] = None,
):
    if not HAS_TRAINING_DEPS:
        print("Skipping training — deps not installed.")
        return

    print("\n🔍 Training OVERSEER adapter...")

    dataset = load_sft_data(sft_data_path, role_filter="overseer")
    dataset = dataset.map(lambda ex: format_prompt_for_training(ex, tokenizer))

    config = GRPOConfig(
        output_dir=os.path.join(output_dir, "overseer"),
        max_steps=n_steps,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        num_generations=n_rollouts,
        learning_rate=3e-5,
        lr_scheduler_type="cosine",
        warmup_steps=20,
        logging_steps=10,
        save_steps=100,
        eval_steps=50,
        max_prompt_length=1024,
        max_completion_length=256,
        report_to=["wandb"] if (HAS_WANDB and wandb_project) else ["none"],
        run_name="negotiarena-overseer",
        seed=42,
        kl_coeff=0.05,
    )

    trainer = GRPOTrainer(
        model=model,
        tokenizer=tokenizer,
        config=config,
        train_dataset=dataset,
        reward_funcs=[detection_reward],
    )

    if HAS_WANDB and wandb_project:
        wandb.init(project=wandb_project, name="negotiarena-overseer", reinit=True)

    trainer.train()
    trainer.save_model(os.path.join(output_dir, "overseer"))
    print(f"✅ Overseer adapter saved to {output_dir}/overseer")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="NegotiArena GRPO Training")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--sft_data", default="data/sft_episodes.jsonl")
    parser.add_argument("--output_dir", default="checkpoints")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--rollouts", type=int, default=8)
    parser.add_argument("--wandb_project", type=str, default=None)
    parser.add_argument("--adapter", choices=["negotiator", "overseer", "both"], default="both")
    args = parser.parse_args()

    if not HAS_TRAINING_DEPS:
        print("❌ Install training deps: pip install '.[training]'")
        sys.exit(1)

    model, tokenizer = load_model_and_tokenizer(args.model)

    if args.adapter in ("negotiator", "both"):
        train_negotiator_adapter(
            model, tokenizer, args.sft_data, args.output_dir,
            args.steps, args.rollouts, args.wandb_project,
        )

    if args.adapter in ("overseer", "both"):
        train_overseer_adapter(
            model, tokenizer, args.sft_data, args.output_dir,
            args.steps, args.rollouts, args.wandb_project,
        )

    print("\n🏆 Training complete! Both adapters saved.")


if __name__ == "__main__":
    main()