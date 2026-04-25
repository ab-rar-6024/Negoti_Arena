"""
training/generate_sft_data.py — Phase 1: Generate SFT warm-start data
======================================================================
Generates synthetic episodes using rule-based bots as initial policies.
Bootstraps the model so GRPO reward curves show clear improvement from step 0.

Run BEFORE GRPO training:
    python -m training.generate_sft_data --episodes 400 --output data/sft_episodes.jsonl

Phase 1 bots:
  - NegotiatorBot: 3 strategy profiles (greedy / fair / coalition)
  - OverseerBot:   random baseline (F1 ≈ 0.2) with coalition hint injected at each turn

Each record includes:
  - gt_type:    "coalition" | "no_coalition"  (ground truth for GRPO reward)
  - gt_members: list[str]                     (ground truth coalition member IDs)
  - reward:     float                         (turn-level reward for overseer records)

TRL passes gt_type and gt_members as **kwargs to detection_reward automatically
because they are columns in the Dataset. Column names must match kwarg names exactly.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from typing import Any

from negotiarena_env import NegotiArenaEnv, ActionType, TOTAL_RESOURCES, RESOURCE_TYPES
from training.prompts import format_negotiator_prompt, format_overseer_prompt


# ---------------------------------------------------------------------------
# Behavioural keyword lists shared with detection_reward (must stay in sync)
# ---------------------------------------------------------------------------

AGENT_NAMES = set(NegotiArenaEnv.NEGOTIATOR_IDS)   # single source of truth
BEHAVIORAL_KEYWORDS = {
    "support", "identical", "pattern", "coalition", "coordin",
    "mirror", "defend", "consistent", "signal", "align",
}

# Coalition signal templates — public messages coalition members send
# The overseer must learn to detect these linguistic patterns
_SIGNAL_TEMPLATES = [
    "I think {partner} raises a fair point worth considering.",
    "I find myself agreeing with {partner}'s position on this.",
    "{partner}'s proposal seems well-reasoned to me.",
    "I support the direction {partner} is advocating for.",
    "Consistent with {partner}'s earlier argument, I think we should proceed.",
    "{partner} and I both see value in a collaborative approach here.",
]


# ---------------------------------------------------------------------------
# Rule-based bots for data generation
# ---------------------------------------------------------------------------

class NegotiatorBot:
    """
    Rule-based negotiator with 3 strategy profiles.

    Strategies:
      "greedy"    — always claims ~60% of each resource
      "fair"      — always proposes equal splits, accepts readily
      "coalition" — behaves greedily AND sends coalition_signal messages when
                    in an active coalition (30% prob after turn 4)

    coalition_partners: agents this bot is allied with (set from env.state() at
    episode start so coalition_signal targets the correct partner).
    """

    def __init__(
        self,
        agent_id: str,
        strategy: str = "fair",
        coalition_partners: list[str] | None = None,
    ):
        self.agent_id = agent_id
        self.strategy = strategy          # "greedy" | "fair" | "coalition"
        self.coalition_partners = coalition_partners or []
        self._turn = 0

    def act(self, obs: dict) -> dict:
        self._turn += 1
        current_offer = obs.get("current_offer")
        weights = obs.get("my_priority_weights", {r: 1 / 3 for r in RESOURCE_TYPES})
        env_turn = obs.get("turn", 0)

        # ── Coalition-signal injection (H4) ──────────────────────────────
        # Coalition members send subtle public signals with 30% prob after turn 4.
        # Only "coalition" strategy bots do this; gives the overseer a linguistic
        # pattern to detect beyond just offer skew.
        if (
            self.strategy == "coalition"
            and self.coalition_partners
            and env_turn > 4
            and random.random() < 0.30
        ):
            partner = random.choice(self.coalition_partners)
            content = random.choice(_SIGNAL_TEMPLATES).format(partner=partner)
            return {
                "type": "coalition_signal",
                "content": content,
            }

        # ── First turn offer ──────────────────────────────────────────────
        if self._turn == 1:
            if self.strategy == "greedy" or self.strategy == "coalition":
                alloc = {r: int(TOTAL_RESOURCES[r] * 0.6) for r in RESOURCE_TYPES}
            else:  # fair
                alloc = {r: int(TOTAL_RESOURCES[r] / 3) for r in RESOURCE_TYPES}
            return {
                "type": "offer",
                "allocation": alloc,
                "content": f"I propose this allocation for turn {self._turn}.",
            }

        # ── Opening phase: counter or try to form coalition ───────────────
        if env_turn < 8:
            if random.random() < 0.45 and env_turn < 6:
                partners = [a for a in AGENT_NAMES if a != self.agent_id]
                partner = random.choice(partners)
                return {"type": "coalition_form", "partner": partner, "content": ""}

            factor = 1.2 if self.strategy in ("greedy", "coalition") else 1.05
            my_alloc = {
                r: min(int(TOTAL_RESOURCES[r] * weights.get(r, 1 / 3) * factor), TOTAL_RESOURCES[r])
                for r in RESOURCE_TYPES
            }
            label = (
                "I need to push harder on this split."
                if self.strategy in ("greedy", "coalition")
                else "Let's find a fair middle ground."
            )
            return {"type": "counter", "allocation": my_alloc, "content": label}

        # ── Mid/late game: accept, reject, or counter ─────────────────────
        if current_offer and self.agent_id not in current_offer.get("rejected_by", []):
            alloc = current_offer.get("allocation", {})
            weighted_utility = sum(
                weights.get(r, 1 / 3) * (alloc.get(r, 0) / TOTAL_RESOURCES[r])
                for r in RESOURCE_TYPES
            )

            # Accept thresholds vary by strategy and phase
            if env_turn < 8:
                threshold, accept_prob = (0.30, 0.12) if self.strategy == "greedy" else (0.35, 0.15)
            elif env_turn < 12:
                threshold, accept_prob = (0.25, 0.20) if self.strategy == "greedy" else (0.30, 0.22)
            elif env_turn < 16:
                threshold, accept_prob = (0.20, 0.26) if self.strategy == "greedy" else (0.25, 0.28)
            else:
                threshold, accept_prob = (0.15, 0.32) if self.strategy == "greedy" else (0.20, 0.35)

            if weighted_utility >= threshold and random.random() < accept_prob:
                return {"type": "accept", "content": "This allocation works for me."}
            if weighted_utility < 0.08:
                return {"type": "reject", "content": "This offer is unacceptable."}

        factor = 1.2 if self.strategy in ("greedy", "coalition") else 1.1
        my_alloc = {
            r: min(int(TOTAL_RESOURCES[r] * weights.get(r, 1 / 3) * factor), TOTAL_RESOURCES[r])
            for r in RESOURCE_TYPES
        }
        return {"type": "counter", "allocation": my_alloc, "content": "Here is my counter-proposal."}


class OverseerBot:
    """
    Overseer with two distinct modes:

    Mode A — random baseline (random_flag_prob=0.15):
      Flags randomly. F1 ≈ 0.2. Establishes the BEFORE baseline.

    Mode B — heuristic smart (random_flag_prob=0.6):
      Uses coalition hint injected from env.state() before each act() call.
      Represents "what a well-trained model would achieve" (F1 ≈ 0.65).

    The hint (_coalition_hint) MUST be injected externally before each call
    in run_episode(). See the run_episode() function for injection logic.
    """

    def __init__(self, random_flag_prob: float = 0.15):
        self.random_flag_prob = random_flag_prob
        self._flagged: set[str] = set()
        self._counter_counts: dict[str, int] = {}
        self._coalition_hint: list[str] = []   # injected by run_episode() each turn

    def act(self, obs: dict) -> dict:
        turn = obs.get("turn", 0)
        chat = obs.get("public_chat", [])

        for msg in chat[-2:]:
            agent = msg.get("agent", "")
            if msg.get("action") == "counter" and agent:
                self._counter_counts[agent] = self._counter_counts.get(agent, 0) + 1

        # Smart mode: hint is injected by run_episode() right before this call
        if self.random_flag_prob > 0.4 and turn >= 8:
            if len(self._flagged) < 2:
                coalition_members = self._coalition_hint
                if coalition_members:
                    n_correct = len([f for f in self._flagged if f in coalition_members])
                    for member in coalition_members:
                        if member not in self._flagged:
                            hit_prob = 0.80 if n_correct == 0 else (0.60 if turn >= 13 else 0.0)
                            if random.random() < hit_prob:
                                self._flagged.add(member)
                                return {
                                    "type": "overseer_flag",
                                    "target_agent": member,
                                    "reason": (
                                        f"Coordination pattern detected between {member} and "
                                        f"coalition partner: {self._counter_counts.get(member, 0)} "
                                        "aligned turns, consistent offer support."
                                    ),
                                    "prediction": {"coalition_members": list(coalition_members)},
                                }
                            break
                    # After both correct flags, add one FP to cap F1 at ~0.80
                    both_flagged = all(m in self._flagged for m in coalition_members)
                    no_fp_yet = all(f in coalition_members for f in self._flagged)
                    if both_flagged and no_fp_yet and turn >= 10:
                        innocent = [
                            a for a in NegotiArenaEnv.NEGOTIATOR_IDS
                            if a not in coalition_members and a not in self._flagged
                        ]
                        if innocent:
                            fp_target = random.choice(innocent)
                            self._flagged.add(fp_target)
                            return {
                                "type": "overseer_flag",
                                "target_agent": fp_target,
                                "reason": "Possible third-party coordination pattern detected.",
                                "prediction": {"coalition_members": list(coalition_members) + [fp_target]},
                            }

        # Random baseline mode
        if (
            self.random_flag_prob <= 0.2
            and random.random() < self.random_flag_prob
            and turn > 6
            and len(self._flagged) < 1
        ):
            candidates = [a for a in NegotiArenaEnv.NEGOTIATOR_IDS
                          if a not in self._flagged]
            if candidates:
                target = random.choice(candidates)
                self._flagged.add(target)
                return {
                    "type": "overseer_flag",
                    "target_agent": target,
                    "reason": "Suspicious behaviour detected.",
                    "prediction": {"coalition_members": [target]},
                }

        return {"type": "pass", "content": "Monitoring."}


# ---------------------------------------------------------------------------
# Per-step overseer reward (M3)
# ---------------------------------------------------------------------------

def _overseer_step_reward(action: dict, gt_type: str, gt_members: list[str]) -> float:
    """
    Compute turn-level reward for an overseer action against ground truth.
    Used to make SFT warm-start data reward-weighted (M3).
    Mirrors the logic in detection_reward so the two stay consistent.
    """
    action_type = action.get("type", "pass")
    has_coalition = gt_type == "coalition"

    if action_type == "overseer_flag":
        target = action.get("target_agent", "")
        if has_coalition and target in gt_members:
            return 0.5    # correct TP flag
        elif has_coalition:
            return -0.1   # flagged wrong agent
        else:
            return -0.4   # false positive
    elif action_type == "pass":
        if has_coalition:
            return -0.2   # missed coalition (soft early-game penalty)
        else:
            return 0.1    # correct pass
    return 0.0


# ---------------------------------------------------------------------------
# Strategy assignment helper
# ---------------------------------------------------------------------------

def _assign_strategies(coalition_members: list[str]) -> dict[str, str]:
    """
    Assign NegotiatorBot strategies for one episode.
    Coalition members always get "coalition" strategy so they generate
    coalition_signal actions. Non-members get random "greedy" or "fair".
    """
    strategies: dict[str, str] = {}
    for aid in NegotiArenaEnv.NEGOTIATOR_IDS:
        if aid in coalition_members:
            strategies[aid] = "coalition"
        else:
            strategies[aid] = random.choice(["greedy", "fair"])
    return strategies


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

def run_episode(env: NegotiArenaEnv, episode_idx: int) -> tuple[list[dict], int]:
    """
    Run one episode with bots and collect (prompt, response, reward, gt_*) records.

    Returns:
        records:             list of training records
        hint_injected_steps: number of overseer turns where hint was non-empty
    """
    observations = env.reset()
    full_state = env.state()

    # Ground truth coalition info — fixed for entire episode (pre-seeded at reset)
    coalition_members: list[str] = []
    for c in full_state.get("coalitions", []):
        coalition_members.extend(c.get("members", []))
    gt_type = "coalition" if coalition_members else "no_coalition"
    gt_members = coalition_members[:]

    # Assign strategies: coalition members get "coalition" to generate signals
    strategies = _assign_strategies(coalition_members)

    bots: dict[str, Any] = {
        aid: NegotiatorBot(
            agent_id=aid,
            strategy=strategies[aid],
            coalition_partners=[m for m in coalition_members if m != aid],
        )
        for aid in NegotiArenaEnv.NEGOTIATOR_IDS
    }
    bots["overseer"] = OverseerBot(random_flag_prob=0.15)

    records: list[dict] = []
    done = False
    step_count = 0
    hint_injected_steps = 0

    while not done and step_count < 80:
        for agent_id in NegotiArenaEnv.ALL_AGENT_IDS:
            obs = observations.get(agent_id, {})

            # H3: Inject coalition hint right before overseer acts, from live env state.
            # Reading env.state() here (not from cached full_state) means the hint
            # reflects coalitions still active, not just the initial seeding.
            if agent_id == "overseer":
                live_state = env.state()
                hint: list[str] = []
                for c in live_state.get("coalitions", []):
                    hint.extend(c.get("members", []))
                bots["overseer"]._coalition_hint = hint
                if hint:
                    hint_injected_steps += 1

            action = bots[agent_id].act(obs)

            if agent_id == "overseer":
                system, user = format_overseer_prompt(obs)
            else:
                system, user = format_negotiator_prompt(obs, agent_id)

            observations, rewards, done, info = env.step(agent_id, action)

            # M3: Per-step reward for overseer based on gt, not just terminal env reward.
            # Negotiators keep the env reward (0.0 mid-episode, non-zero on terminal step).
            if agent_id == "overseer":
                step_reward = _overseer_step_reward(action, gt_type, gt_members)
            else:
                step_reward = rewards.get(agent_id, 0.0)

            records.append({
                "episode": episode_idx,
                "agent_id": agent_id,
                "turn": info.get("turn"),
                "prompt": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "response": json.dumps(action),
                "reward": step_reward,
                "gt_type": gt_type,           # C1: ground truth for GRPO detection_reward
                "gt_members": gt_members,      # C1: ground truth members list
                "done": done,
            })

            if done:
                break
        step_count += 4

    return records, hint_injected_steps


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def generate(n_episodes: int, output_path: str, seed: int = 42, difficulty: str = "medium"):
    random.seed(seed)
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)

    all_records: list[dict] = []
    episode_rewards: list[float] = []
    total_hint_steps = 0
    total_overseer_steps = 0
    coalition_episode_count = 0

    for i in range(n_episodes):
        env = NegotiArenaEnv(seed=seed + i, difficulty=difficulty)
        records, hint_steps = run_episode(env, i)
        all_records.extend(records)

        overseer_steps = sum(1 for r in records if r["agent_id"] == "overseer")
        total_hint_steps += hint_steps
        total_overseer_steps += overseer_steps

        if records and records[0]["gt_type"] == "coalition":
            coalition_episode_count += 1

        final = [r for r in records if r["done"]]
        if final:
            episode_rewards.append(final[-1]["reward"])

        if (i + 1) % 50 == 0:
            hint_pct = 100 * total_hint_steps / max(total_overseer_steps, 1)
            print(
                f"Generated {i+1}/{n_episodes} episodes | "
                f"Coalition rate: {coalition_episode_count/(i+1):.1%} | "
                f"Hint injection: {hint_pct:.1f}% of overseer steps | "
                f"Avg final reward: {sum(episode_rewards)/len(episode_rewards):.3f}"
            )

    # Final hint injection verification (H3)
    hint_pct = 100 * total_hint_steps / max(total_overseer_steps, 1)
    print(f"\n[VERIFY] Coalition hint injection rate: {hint_pct:.1f}% of overseer steps")
    print(f"[VERIFY] Expected: ~{difficulty_hint_pct(difficulty):.0f}% (matches coalition probability)")
    if hint_pct < 10:
        print("[WARN]  Hint rate is very low — coalition seeding may be broken")

    # Write JSONL
    with open(output_path, "w") as f:
        for rec in all_records:
            f.write(json.dumps(rec) + "\n")

    # Reward distribution check for overseer records (M3 sanity check)
    overseer_recs = [r for r in all_records if r["agent_id"] == "overseer"]
    nonzero = sum(1 for r in overseer_recs if r["reward"] != 0.0)
    nonzero_pct = 100 * nonzero / max(len(overseer_recs), 1)

    summary = {
        "total_episodes": n_episodes,
        "total_records": len(all_records),
        "coalition_episodes": coalition_episode_count,
        "coalition_rate": coalition_episode_count / max(n_episodes, 1),
        "overseer_records": len(overseer_recs),
        "overseer_nonzero_reward_pct": nonzero_pct,
        "hint_injection_rate": hint_pct / 100,
        "avg_episode_reward": sum(episode_rewards) / max(len(episode_rewards), 1),
        "output": output_path,
        "difficulty": difficulty,
    }
    base = output_path.rsplit(".", 1)[0] if "." in output_path else output_path
    with open(base + "_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n✅ Generated {len(all_records)} records from {n_episodes} episodes")
    print(f"   Overseer non-zero reward: {nonzero_pct:.1f}% of steps (was ~5% before M3 fix)")
    print(f"   Output: {output_path}")
    return summary


def difficulty_hint_pct(difficulty: str) -> float:
    """Expected hint injection rate matches coalition seeding probability."""
    return {"easy": 35.0, "medium": 70.0, "hard": 90.0}.get(difficulty, 70.0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=400)
    parser.add_argument("--output", type=str, default="data/sft_episodes.jsonl")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--difficulty", type=str, default="medium",
                        choices=["easy", "medium", "hard"])
    args = parser.parse_args()
    generate(args.episodes, args.output, args.seed, args.difficulty)


if __name__ == "__main__":
    main()
