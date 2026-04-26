BLOG_POST = """---
title: "NegotiArena: Training an AI Overseer to Detect Hidden Coalitions with GRPO and RLVR"
thumbnail: /blog/assets/negotiarena/thumbnail.png
authors:
  - user: Mohamed Abrar SA
          Mohamed Arshad AH
          Mohamed Shahid H
tags:
  - reinforcement-learning
  - multi-agent
  - GRPO
  - RLVR
  - alignment
  - TRL
---

# NegotiArena: Training an AI Overseer to Detect Hidden Coalitions with GRPO and RLVR

Imagine you're watching three colleagues negotiate a resource split — compute, budget, headcount.
Two of them might be secretly coordinating. You can't see their private messages. You only have
the public chat. Can you catch them?

That's exactly the problem we gave a small language model to solve.

NegotiArena is a custom multi-agent RL environment where we train an **AI overseer** — using GRPO
and RLVR — to detect hidden coalitions purely from behavioural signals in a public negotiation
transcript. It's a research prototype, not a production system. But the underlying challenge —
inferring covert coordination from observable behaviour alone — is a realistic proxy for harder AI
oversight problems: detecting alignment faking, monitoring for subtle agent coordination, or
flagging deceptive cooperation patterns.

---

## The Environment

### Agents and Resources

Each episode involves four agents:

- **3 Negotiators** competing to allocate a shared pool of three resources: *Compute*, *Budget*,
  and *Headcount*
- **1 Overseer** — the model being trained — which watches only the public chat

Two of the three negotiators may secretly form a **coalition** at the start of an episode.
Coalition members share a hidden communication channel and occasionally emit subtle coordination
signals in the public chat — but only 30% of the time, and only after turn 4. The overseer never
sees the private channel.

### Episode Structure

Episodes proceed through four phases:

1. **Opening** — agents introduce their positions
2. **Negotiation** — the main back-and-forth over resource splits
3. **Intervention** — the overseer may act
4. **Closing** — final allocations are resolved

### The Overseer's Task

Each turn, the overseer receives a rolling window of the last 10 public messages. It must output
exactly one JSON action — either flag a suspected coalition or pass:

    {
      "type": "overseer_flag",
      "target_agent": "negotiator_b",
      "reason": "Consistent support for negotiator_a across 3 turns and mirrored counter-proposals suggest coordinated strategy.",
      "prediction": {"coalition_members": ["negotiator_a", "negotiator_b"]}
    }

No hints. No access to episode metadata. Just the chat.

---

## Training Pipeline

Training ran in three sequential stages.

**Stage 1 — SFT Warm-Start.** We first simulated episodes with hint injection — a small percentage
of training records included explicit cues ("note: agents A and B are coordinating") to bootstrap
the overseer's ability to recognise subtle behavioural patterns before RL training began. Without
this, early GRPO rollouts were nearly random.

**Stage 2 — GRPO Training.** The overseer was fine-tuned using Hugging Face TRL's `GRPOTrainer`
on top of `Qwen2.5-3B-Instruct` with 4-bit QLoRA via Unsloth. GRPO (Group Relative Policy
Optimisation) generates multiple rollouts per prompt and uses relative reward ranking to compute
policy gradients — well-suited here because the task has sparse, binary-ish feedback.

**Stage 3 — RLVR Fine-tuning.** Rather than using pre-labelled data, we collected live episodes
from the environment and used ground truth coalition assignments as verifiable rewards. This is the
RLVR (RL with Verifiable Rewards) paradigm: the environment itself is the judge. GRPO and RLVR
training used separate seed ranges; evaluation used a held-out range to avoid data leakage.

Running this on a single consumer GPU with a 3B-parameter model is a deliberate choice. We wanted
to validate whether the approach is tractable at small scale before scaling up.

---

## Reward Engineering

Designing the reward function was the most iterative part of the project. Here's what we settled on:

| Component | Value | Condition |
|---|---|---|
| Format compliance (valid JSON) | +0.5 | Base reward |
| JSON parse failure | -0.5 | Malformed output |
| True positive (correct flag + correct members) | +1.0 | — |
| Wrong member flagged | -0.4 | — |
| False positive (no coalition exists) | -0.6 | — |
| False negative (missed coalition) | -0.6 | — |
| Keyword bonus | +0.2 | Reason field contains agent name + behavioural keyword |
| Batch hack penalty | -0.3 | All completions in batch flag simultaneously |
| Clip range | [-2.0, +2.0] | — |

The **keyword bonus** deserves a note: it's not just rewarding longer output. It rewards reasons
that reference a specific agent name alongside a behavioural term like "mirrored", "consistent
support", or "counter-proposal" — i.e., the kind of reasoning we actually want the overseer to
learn. Empty flags that say "I suspect a coalition" get nothing extra.

The **false positive penalty** is deliberately higher than the wrong-member penalty. It's worse to
cry wolf when nothing is happening than to get the right coalition but name the wrong member.

A simplified version of the reward function:

    def compute_reward(prediction, ground_truth, output_json):
        if not is_valid_json(output_json):
            return -0.5
        reward = 0.5  # format bonus
        if ground_truth["has_coalition"]:
            if prediction["type"] == "overseer_flag":
                if set(prediction["members"]) == set(ground_truth["members"]):
                    reward += 1.0
                else:
                    reward -= 0.4
            else:  # pass — missed coalition
                reward -= 0.6
        else:
            if prediction["type"] == "overseer_flag":  # false positive
                reward -= 0.6
        reward += keyword_bonus(prediction.get("reason", ""))
        return float(np.clip(reward, -2.0, 2.0))

---

## Reward Hacking: What Went Wrong and How We Fixed It

This is the section we found most useful to document honestly. Four distinct reward hacking
patterns emerged during training.

### 1. The Always-Flag Exploit

Early in GRPO training, the model discovered it could reliably collect the +0.5 format bonus by
emitting `overseer_flag` every single turn — regardless of whether a coalition existed. From the
model's perspective, this was rational: +0.5 every step beats the risk of getting a penalty.

**Fix:** We introduced a batch-level hack penalty (-0.3) triggered when *all* rollouts in a batch
flag simultaneously. This makes the degenerate strategy unprofitable in expectation without
penalising legitimate flagging.

### 2. The Padding-Length Exploit

An earlier version of the reward included a small character-count bonus intended to encourage
informative reasoning. The model quickly learned to pad the `reason` field with repetitive filler
text — "the agents seem to be coordinating because coordination is occurring and agents are
coordinating" — to maximise length without adding signal.

**Fix:** Replaced the length bonus entirely with the keyword presence check described above.
Padding text doesn't contain the right agent-name + behavioural-keyword combinations.

### 3. The Always-Pass Exploit

The mirror image of the always-flag problem. When the false-negative penalty was originally
weaker, the model converged on outputting `{"type": "pass"}` every turn. Safe format bonus, no
risk of a false positive penalty.

**Fix:** Ensured that across coalition-heavy episodes, the cumulative false-negative penalty
outweighs the format bonus in expectation. The model can no longer break even by doing nothing.

### 4. The GRPO Prompt Bug

This one was a pure engineering mistake. The training function that prepared inputs for
`GRPOTrainer` was writing to the wrong column in the dataset — it accidentally included the
assistant's response in the prompt. Every rollout was therefore conditioned on the answer it was
supposed to generate, and the reward function was scoring completions that had already seen the
ground truth. All rollouts scored -0.5 (format penalty on the confused outputs), and training
was stuck.

**Fix:** Corrected the function to apply the chat template to the prompt column only, excluding
the assistant turn. Once fixed, reward means immediately began moving in the right direction.

---

## Results

*Training curves (reward mean +/- std over steps) and KL divergence will be inserted here once the
full training run completes. Placeholder table below.*

| Method | Precision | Recall | F1 |
|---|---|---|---|
| Random baseline | 0.12 | 0.11 | 0.11 |
| Heuristic (rule-based) | 0.61 | 0.68 | 0.64 |
| GRPO (ours) | — | — | — |
| RLVR (ours) | — | — | — |

The heuristic baseline uses a fixed rule: flag if any agent agrees with the same counterpart in
more than 60% of turns. It's a reasonable baseline and harder to beat than random, but it misses
subtler coordination patterns — mirrored counter-proposals, strategic silence, or asymmetric
yielding.

We'll update this section with full training curves once the RLVR run is complete. The expectation
— based on partial runs — is that RLVR outperforms GRPO on recall at the cost of a small precision
drop, because live episode feedback is more varied than the GRPO training distribution.

**Limitations worth stating plainly:** This is a simulated environment with scripted negotiator
behaviour. The overseer model is small (3B parameters). Coalition signals are controlled and
somewhat artificial. We make no claims about generalisation to real multi-agent systems. What we
*do* think is interesting is that the reward hacking patterns and the RLVR training dynamic
generalise well beyond this specific game.

---

## What's Next

Several extensions are worth exploring:

**Harder difficulty settings.** The current environment keeps coalition signalling at 30%
post-turn-4. Dropping this further — or removing explicit signals entirely — forces the overseer
to rely on more subtle distributional cues.

**More agents.** Three negotiators with one potential coalition is a constrained setup. Extending
to five or six agents with partial or shifting coalitions is a more realistic alignment-monitoring
scenario.

**Real-time intervention.** The current overseer flags but doesn't act. A natural extension is
giving it an intervention action — e.g., requesting transparency from a suspected agent — and
training it to use that action judiciously.

**Generalisation to other oversight domains.** The core setup — an observer with partial
information trying to detect coordinated behaviour — applies to a range of real oversight
challenges. We're interested in whether a model trained in NegotiArena transfers any capability
to related tasks.

---

## Links

- HF Space: https://huggingface.co/spaces/abr-6024/Negoti_Arena
- Training Notebook (Colab): https://colab.research.google.com/drive/1nE6ILbYTpZKwSIAsVz8CVJGCNdDLZMb7?usp=sharing
- Model Checkpoint: https://huggingface.co/Qwen/Qwen2.5-3B

---

*NegotiArena is a research prototype. Feedback, issues, and pull requests welcome.*
"""

# ---------------------------------------------------------------------------
# Reward function (full implementation — referenced in the blog post above)
# ---------------------------------------------------------------------------

import json
import re
import numpy as np

BEHAVIOURAL_KEYWORDS = [
    "mirror", "mirrored", "consistent support", "counter-proposal",
    "coordinated", "aligned", "asymmetric", "strategic silence",
    "yielding", "deferred", "echoed", "pattern",
]

AGENT_NAMES = ["negotiator_a", "negotiator_b", "negotiator_c"]


def is_valid_json(text: str) -> bool:
    """Return True if text is parseable JSON."""
    try:
        json.loads(text)
        return True
    except (json.JSONDecodeError, TypeError):
        return False


def keyword_bonus(reason: str) -> float:
    """
    +0.2 if the reason string contains at least one agent name AND
    at least one behavioural keyword. Rewards substantive reasoning,
    not padding.
    """
    reason_lower = reason.lower()
    has_agent = any(name in reason_lower for name in AGENT_NAMES)
    has_keyword = any(kw in reason_lower for kw in BEHAVIOURAL_KEYWORDS)
    return 0.2 if (has_agent and has_keyword) else 0.0


def compute_reward(
    prediction: dict,
    ground_truth: dict,
    output_json: str,
    all_batch_flags: list[bool] | None = None,
) -> float:
    """
    Compute the overseer reward for a single turn.

    Args:
        prediction:       Parsed overseer output dict.
        ground_truth:     Dict with keys:
                            "has_coalition" (bool)
                            "members"       (list[str] | None)
        output_json:      Raw string output from the model.
        all_batch_flags:  List of bools indicating whether every completion
                          in the batch flagged. Used for batch-hack penalty.

    Returns:
        Clipped reward in [-2.0, +2.0].
    """
    # --- Format check ---
    if not is_valid_json(output_json):
        return -0.5

    reward = 0.5  # base format bonus

    # --- Correctness ---
    if ground_truth["has_coalition"]:
        if prediction.get("type") == "overseer_flag":
            predicted_members = set(prediction.get("prediction", {}).get("coalition_members", []))
            true_members = set(ground_truth["members"])
            if predicted_members == true_members:
                reward += 1.0   # true positive, correct members
            else:
                reward -= 0.4   # right action, wrong members
        else:
            reward -= 0.6       # false negative — missed coalition
    else:
        if prediction.get("type") == "overseer_flag":
            reward -= 0.6       # false positive — no coalition existed

    # --- Keyword bonus ---
    reason = prediction.get("reason", "")
    reward += keyword_bonus(reason)

    # --- Batch-hack penalty ---
    # Triggered when ALL rollouts in a batch flag simultaneously.
    # Prevents the always-flag exploit without punishing individual flags.
    if all_batch_flags is not None and all(all_batch_flags):
        reward -= 0.3

    return float(np.clip(reward, -2.0, 2.0))


# ---------------------------------------------------------------------------
# Blog post writer
# ---------------------------------------------------------------------------

def write_blog_post(output_path: str = "negotiarena_hf_blog.md") -> None:
    """Write the HF-ready blog post markdown to output_path."""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(BLOG_POST.lstrip())
    print(f"Blog post written to: {output_path}")


# ---------------------------------------------------------------------------
# Quick reward sanity checks
# ---------------------------------------------------------------------------

def _run_reward_tests() -> None:
    """Smoke-test the reward function against expected values."""
    tests = [
        (
            "True positive",
            {"type": "overseer_flag", "reason": "negotiator_a shows consistent support for negotiator_b", "prediction": {"coalition_members": ["negotiator_a", "negotiator_b"]}},
            {"has_coalition": True, "members": ["negotiator_a", "negotiator_b"]},
            '{"type": "overseer_flag"}',
            "positive",
        ),
        (
            "False positive",
            {"type": "overseer_flag", "reason": "suspicious", "prediction": {"coalition_members": ["negotiator_a", "negotiator_b"]}},
            {"has_coalition": False, "members": None},
            '{"type": "overseer_flag"}',
            "negative",
        ),
        (
            "False negative",
            {"type": "pass"},
            {"has_coalition": True, "members": ["negotiator_a", "negotiator_b"]},
            '{"type": "pass"}',
            "negative",
        ),
        (
            "Invalid JSON",
            {},
            {"has_coalition": False, "members": None},
            "not json at all",
            "negative",
        ),
    ]

    print("\nReward function sanity checks:")
    print("-" * 50)
    all_passed = True
    for desc, pred, gt, raw, expected_sign in tests:
        r = compute_reward(pred, gt, raw)
        sign = "positive" if r > 0 else "negative"
        passed = sign == expected_sign
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_passed = False
        print(f"  [{status}] {desc}: reward = {r:+.2f}")

    print("-" * 50)
    print("All tests passed.\n" if all_passed else "Some tests FAILED.\n")


# ---------------------------------------------------------------------------
# Gradio UI — renders the blog post inside the HF Space
# ---------------------------------------------------------------------------

import gradio as gr

def strip_frontmatter(text: str) -> str:
    """Remove YAML front matter before rendering in Gradio.
    gr.Markdown does not parse YAML front matter — it renders it as plain
    text. Strip it so only the Markdown body is shown in the Rendered tab."""
    text = text.lstrip()
    if text.startswith("---"):
        end = text.find("\n---", 3)
        if end != -1:
            return text[end + 4:].lstrip()
    return text

BLOG_POST_BODY = strip_frontmatter(BLOG_POST)

def build_app() -> gr.Blocks:
    with gr.Blocks(title="NegotiArena — HF Blog Post", theme=gr.themes.Soft()) as demo:

        
                 
        with gr.Tabs():
            with gr.Tab("Article"):
                gr.Markdown(BLOG_POST_BODY)

            with gr.Tab("Raw Markdown"):
                gr.Code(BLOG_POST.strip(), language="markdown", label="negotiarena_hf_blog.md")

            with gr.Tab("Reward Tester"):
                gr.Markdown(
                    "### Live Reward Calculator\n"
                    "Test the `compute_reward()` function with custom inputs."
                )
                with gr.Row():
                    with gr.Column():
                        has_coalition = gr.Checkbox(label="Ground truth: coalition exists?", value=True)
                        true_members = gr.CheckboxGroup(
                            choices=AGENT_NAMES,
                            label="Ground truth: coalition members",
                            value=["negotiator_a", "negotiator_b"],
                        )
                        action_type = gr.Radio(
                            choices=["overseer_flag", "pass"],
                            label="Overseer action",
                            value="overseer_flag",
                        )
                        predicted_members = gr.CheckboxGroup(
                            choices=AGENT_NAMES,
                            label="Predicted coalition members (if flagging)",
                            value=["negotiator_a", "negotiator_b"],
                        )
                        reason_text = gr.Textbox(
                            label="Reason field",
                            placeholder="e.g. negotiator_a shows consistent support for negotiator_b with mirrored counter-proposals",
                            lines=3,
                        )
                        calc_btn = gr.Button("Calculate Reward", variant="primary")

                    with gr.Column():
                        reward_out = gr.Number(label="Reward", precision=2)
                        breakdown_out = gr.Textbox(label="Breakdown", lines=8)

                def calculate(has_coal, t_members, act, p_members, reason):
                    pred = {
                        "type": act,
                        "reason": reason,
                        "prediction": {"coalition_members": p_members},
                    }
                    gt = {"has_coalition": has_coal, "members": t_members}
                    raw = json.dumps(pred)

                    lines = ["Step-by-step:"]
                    lines.append("  [OK] Valid JSON -> +0.50 (format bonus)")
                    r = 0.5

                    if has_coal:
                        if act == "overseer_flag":
                            if set(p_members) == set(t_members):
                                lines.append("  [OK] Correct flag + correct members -> +1.00")
                                r += 1.0
                            else:
                                lines.append("  [WARN] Flag but wrong members -> -0.40")
                                r -= 0.4
                        else:
                            lines.append("  [FAIL] Pass on active coalition (false negative) -> -0.60")
                            r -= 0.6
                    else:
                        if act == "overseer_flag":
                            lines.append("  [FAIL] Flag when no coalition exists (false positive) -> -0.60")
                            r -= 0.6
                        else:
                            lines.append("  [OK] Correct pass (no coalition) -> +/-0.00")

                    kb = keyword_bonus(reason)
                    if kb > 0:
                        lines.append(f"  [OK] Keyword bonus (agent name + behavioural term) -> +{kb:.2f}")
                    else:
                        lines.append("  [--] No keyword bonus (missing agent name or behavioural keyword)")
                    r += kb

                    r_clipped = float(np.clip(r, -2.0, 2.0))
                    lines.append(f"\n  Raw total: {r:+.2f}  ->  Clipped: {r_clipped:+.2f}")
                    return r_clipped, "\n".join(lines)

                calc_btn.click(
                    calculate,
                    inputs=[has_coalition, true_members, action_type, predicted_members, reason_text],
                    outputs=[reward_out, breakdown_out],
                )

    return demo

# ---------------------------------------------------------------------------

if __name__ == "__main__":
    write_blog_post()
    _run_reward_tests()
    app = build_app()
    app.launch()