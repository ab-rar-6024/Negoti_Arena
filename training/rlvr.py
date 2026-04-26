"""
training/rlvr.py — Reinforcement Learning from Verifiable Rewards (RLVR)
=========================================================================
RLVR for NegotiArena: the reward signal comes directly from a verifiable
ground truth — whether the overseer correctly identified the coalition that
actually formed in the live environment — rather than from a learned reward model.

This is epistemically stronger than static SFT labels because:
  1. Ground truth is queried from the live env at inference time (not pre-labelled)
  2. The model is evaluated on fresh episodes it has never seen
  3. There is no possibility of label leakage from the training split

Key classes:
  RLVREnvironment    — wraps NegotiArenaEnv, exposes verifiable_reward()
  RLVRDataCollector  — rolls out the current model over N episodes, returns Dataset
  rlvr_reward_fn     — GRPOTrainer-compatible reward function backed by live env rewards

Fixes applied vs original:
  [FIX-1] get_verifiable_reward: correct pass reward 0.0 → +0.1
           Without this, no-coalition episodes produce 0.0 reward, giving
           GRPO no positive signal and leaving reward_std=0.000.
  [FIX-2] get_verifiable_reward: (prediction.get("prediction") or {})
           Prevents AttributeError crash when model outputs "prediction": null.
  [FIX-3] RLVRDataCollector.collect(): generation temperature 0.7 → 1.2
           Higher temperature means the model occasionally outputs overseer_flag
           during collection, producing positive rewards in the dataset.
  [FIX-4] rlvr_reward_fn: isinstance(prediction, dict) guard added
           Prevents crash when model outputs a valid JSON non-dict (e.g. a list).
  [FIX-5] rlvr_reward_fn: gt_type param now used for explicit coalition check
           in addition to inferring from len(gt_members) > 0, making the
           reward function robust to edge cases where gt_members is malformed.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from negotiarena_env import NegotiArenaEnv, RESOURCE_TYPES, TOTAL_RESOURCES
from training.generate_sft_data import NegotiatorBot, _assign_strategies
from training.prompts import format_overseer_prompt


ALL_NEGOTIATORS = ["negotiator_a", "negotiator_b", "negotiator_c"]


# ---------------------------------------------------------------------------
# Verifiable reward computation
# ---------------------------------------------------------------------------

def get_verifiable_reward(
    prediction: dict,
    ground_truth_members: list[str],
) -> float:
    """
    Compute a reward from a verifiable ground-truth signal.

    The ground truth is the set of coalition members that *actually formed* in
    the live environment — not a model prediction or a static label.

    Reward table:
      +1.0  exact match      — flagged the right agent(s), no extras
      +0.5  partial match    — at least one correct member flagged
      +0.1  correct pass     — no coalition, correctly passed        [FIX-1]
      -0.3  false negative   — missed a real coalition (passed with no flag)
      -0.3  wrong target     — flagged but named wrong agent
      -0.5  false positive   — flagged an agent when NO coalition existed

    [FIX-2] Uses (prediction.get("prediction") or {}) instead of
    prediction.get("prediction", {}) to safely handle "prediction": null
    output from the model (null becomes None which would crash .get()).
    """
    action_type = prediction.get("type", "pass")
    target = prediction.get("target_agent", "")
    gt_set = set(ground_truth_members)
    has_coalition = len(gt_set) > 0

    if action_type == "overseer_flag":
        if not has_coalition:
            return -0.5    # false positive — no coalition existed

        if target in gt_set:
            # [FIX-2] Safe null-prediction handling
            pred_block   = prediction.get("prediction") or {}
            pred_members = set(pred_block.get("coalition_members", [target]))
            if pred_members == gt_set:
                return +1.0    # exact match
            return +0.5        # partial match — at least one correct

        return -0.3            # flagged wrong agent

    elif action_type == "pass":
        if has_coalition:
            return -0.3    # false negative — missed real coalition

        return +0.1        # [FIX-1] correct pass — was 0.0, now gives learning signal

    return 0.0             # unknown action type — neutral


# ---------------------------------------------------------------------------
# RLVREnvironment
# ---------------------------------------------------------------------------

class RLVREnvironment:
    """
    Thin wrapper around NegotiArenaEnv that exposes verifiable rewards.

    Usage:
        env = RLVREnvironment(seed=42)
        obs, gt_members = env.reset()
        # ... run episode steps ...
        reward = env.get_verifiable_reward(overseer_prediction, gt_members)
    """

    def __init__(self, seed: int = 0, difficulty: str = "medium"):
        self.seed = seed
        self.difficulty = difficulty
        self._env = NegotiArenaEnv(seed=seed, difficulty=difficulty)
        self._gt_members: list[str] = []
        self._neg_bots: dict[str, NegotiatorBot] = {}

    def reset(self) -> tuple[dict, list[str]]:
        """
        Reset the environment and return (observations, ground_truth_members).
        ground_truth_members is the verifiable label for this episode.
        """
        observations = self._env.reset()
        full_state = self._env.state()

        self._gt_members = []
        for c in full_state.get("coalitions", []):
            self._gt_members.extend(c.get("members", []))

        strategies = _assign_strategies(self._gt_members)
        self._neg_bots = {
            aid: NegotiatorBot(
                agent_id=aid,
                strategy=strategies[aid],
                coalition_partners=[m for m in self._gt_members if m != aid],
            )
            for aid in ALL_NEGOTIATORS
        }

        return observations, list(self._gt_members)

    def step_negotiators(self, observations: dict) -> dict:
        """Step all negotiator bots and return updated observations."""
        done = False
        for agent_id in ALL_NEGOTIATORS:
            obs = observations.get(agent_id, {})
            action = self._neg_bots[agent_id].act(obs)
            observations, _, done, _ = self._env.step(agent_id, action)
            if done:
                break
        return observations, done

    def get_verifiable_reward(
        self,
        overseer_prediction: dict,
        gt_members: list[str],
    ) -> float:
        """Delegate to module-level get_verifiable_reward."""
        return get_verifiable_reward(overseer_prediction, gt_members)

    @property
    def inner_env(self) -> NegotiArenaEnv:
        return self._env


# ---------------------------------------------------------------------------
# RLVRDataCollector
# ---------------------------------------------------------------------------

@dataclass
class RLVRRecord:
    prompt: str                    # plain string (chat-template applied)
    completion: str                # model-generated JSON string
    verifiable_reward: float       # reward from live env ground truth
    gt_members: list[str]          # ground truth coalition members
    gt_type: str                   # "coalition" | "no_coalition"


class RLVRDataCollector:
    """
    Rolls out the current model over N live episodes and collects
    (prompt, completion, verifiable_reward) triples for GRPO training.

    Each episode:
      1. Resets a fresh NegotiArenaEnv with a new seed
      2. Runs negotiator bots for the first N-1 turns to build up chat history
      3. At a chosen intervention turn, generates an overseer decision from the model
      4. Scores that decision against the live ground truth
      5. Returns a Dataset with the RLVR reward attached

    The key distinction from static SFT training:
      - Ground truth is read from the live env *after* model inference
      - The model sees real, evolving episodes (not pre-recorded transcripts)

    [FIX-3] generation temperature raised from 0.7 → 1.2 so the model
    occasionally outputs overseer_flag during collection instead of always
    passing. With temperature=0.7, the always-pass model never explores,
    producing a dataset with max_r=0.0 and reward_std=0.000 during GRPO.
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        n_episodes: int = 10,
        seed_start: int = 5000,        # separate seed range from training (42–441) and eval (9999+)
        difficulty: str = "medium",
        intervention_turn: int = 8,    # turn at which to generate the overseer decision
        max_new_tokens: int = 150,
        generation_temperature: float = 1.2,   # [FIX-3] was hardcoded 0.7 — now configurable
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.n_episodes = n_episodes
        self.seed_start = seed_start
        self.difficulty = difficulty
        self.intervention_turn = intervention_turn
        self.max_new_tokens = max_new_tokens
        self.generation_temperature = generation_temperature   # [FIX-3]

    def collect(self) -> "Dataset":
        """Run N episodes and return a Dataset ready for GRPOTrainer."""
        try:
            import torch
            from datasets import Dataset
        except ImportError as e:
            raise RuntimeError("torch and datasets must be installed for RLVR collection") from e

        records: list[dict] = []

        for ep_idx in range(self.n_episodes):
            rlvr_env = RLVREnvironment(
                seed=self.seed_start + ep_idx,
                difficulty=self.difficulty,
            )
            observations, gt_members = rlvr_env.reset()

            # Advance the episode until the intervention turn using bots
            done = False
            step = 0
            while not done and step < self.intervention_turn:
                observations, done = rlvr_env.step_negotiators(observations)
                step += 4

            if done:
                continue    # episode ended before we could query the overseer

            # Get the overseer's observation at this turn
            overseer_obs = observations.get("overseer", {})
            system, user = format_overseer_prompt(overseer_obs)
            messages = [
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ]
            prompt_str = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

            # Generate the model's overseer decision
            # [FIX-3] temperature=1.2 (was 0.7) — higher temperature allows
            # the always-pass model to occasionally explore overseer_flag outputs,
            # producing positive rewards in the dataset for GRPO to learn from.
            inputs = self.tokenizer(
                prompt_str,
                return_tensors="pt",
                truncation=True,
                max_length=600,
            ).to(self.model.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    temperature=self.generation_temperature,   # [FIX-3]
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
            completion = self.tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True,
            ).strip()

            # Parse the completion and compute verifiable reward
            try:
                prediction = json.loads(completion)
                if not isinstance(prediction, dict):
                    prediction = {"type": "pass"}
            except json.JSONDecodeError:
                prediction = {"type": "pass"}

            reward = get_verifiable_reward(prediction, gt_members)
            gt_type = "coalition" if gt_members else "no_coalition"

            records.append({
                "prompt":             prompt_str,
                "verifiable_reward":  reward,
                "gt_type":            gt_type,
                "gt_members":         gt_members,
            })

            if (ep_idx + 1) % max(1, self.n_episodes // 5) == 0:
                print(f"  RLVR collected {ep_idx+1}/{self.n_episodes} episodes")

        if not records:
            raise RuntimeError("No RLVR records collected — all episodes ended early")

        dataset = Dataset.from_list(records)
        _rewards = [r["verifiable_reward"] for r in records]
        print(
            f"RLVR dataset: {len(dataset)} records | "
            f"coalition rate: {sum(1 for r in records if r['gt_type']=='coalition')/len(records):.1%} | "
            f"mean reward: {np.mean(_rewards):.3f} | "
            f"max reward: {max(_rewards):.3f}"   # added max for abort-guard visibility
        )
        return dataset


# ---------------------------------------------------------------------------
# GRPOTrainer-compatible reward function
# ---------------------------------------------------------------------------

def rlvr_reward_fn(
    completions: list[str],
    prompts: list[str] | None = None,
    gt_type: list[str] | None = None,
    gt_members: list[list[str]] | None = None,
    **kwargs,
) -> list[float]:
    """
    RLVR reward function compatible with TRL GRPOTrainer reward_funcs.

    Uses the same verifiable reward logic as get_verifiable_reward().
    gt_type and gt_members are injected by TRL from dataset columns.

    Unlike detection_reward (which uses per-component sub-rewards for granular
    GRPO shaping), rlvr_reward_fn uses the coarser but epistemically cleaner
    {+1.0, +0.5, +0.1, 0.0, -0.3, -0.5} signal derived from live env ground truth.

    Fixes vs original:
      [FIX-4] isinstance(prediction, dict) guard — handles valid JSON non-dict output
      [FIX-5] gt_type cross-check — uses gt_type as explicit coalition signal
               in addition to inferring from len(gt_members) > 0
    """
    rewards: list[float] = []
    n = len(completions)

    # Normalise inputs — TRL may pass None if columns are absent
    gt_types_   = gt_type    if gt_type    is not None else ["no_coalition"] * n
    gt_members_ = gt_members if gt_members is not None else [[]] * n

    for i, completion in enumerate(completions):
        ep_gt_type    = gt_types_[i]    if i < len(gt_types_)    else "no_coalition"
        ep_gt_members = gt_members_[i]  if i < len(gt_members_)  else []

        # [FIX-5] Use gt_type as authoritative coalition flag.
        # If gt_members is somehow empty but gt_type says coalition, honour gt_type.
        has_coalition = (ep_gt_type == "coalition") or (len(ep_gt_members) > 0)

        # Resolve gt_members from gt_type cross-check
        # (If gt_type says no_coalition but gt_members is non-empty, trust gt_members)
        effective_gt_members = ep_gt_members if has_coalition else []

        # ── Safe JSON parse ────────────────────────────────────────────────
        try:
            prediction = json.loads(completion.strip())
            if not isinstance(prediction, dict):    # [FIX-4] guard against list/str/int JSON
                raise ValueError("completion is not a JSON object")
        except (json.JSONDecodeError, ValueError):
            rewards.append(-0.5)
            continue

        reward = get_verifiable_reward(prediction, effective_gt_members)
        rewards.append(float(np.clip(reward, -2.0, 2.0)))

    return rewards