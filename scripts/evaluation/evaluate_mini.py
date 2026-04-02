#!/usr/bin/env python3
"""Evaluate mini-experiment results by running actual model inference."""

import json
import argparse
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import numpy as np
from sklearn.metrics import roc_auc_score


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def bootstrap_ci(scores_pos, scores_neg, n_bootstrap=1000):
    """Bootstrap 95 % CI for AUROC."""
    aurocs = []
    rng = np.random.default_rng(42)
    n = len(scores_pos)
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        sp = [scores_pos[i] for i in idx]
        sn = [scores_neg[i] for i in idx]
        labels = [1] * n + [0] * n
        scores = sp + sn
        if len(set(labels)) > 1:
            aurocs.append(roc_auc_score(labels, scores))
    if not aurocs:
        return 0.5, 0.0, 1.0
    return (float(np.mean(aurocs)),
            float(np.percentile(aurocs, 2.5)),
            float(np.percentile(aurocs, 97.5)))


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

# Use the same base model that was used for training (Qwen3VL-4B)
BASE_MODEL = "/hpc2hdd/home/ycui785/model/qwen3vl-4b"


def _load_model_and_tokenizer(checkpoint_path):
    """Load base Qwen2.5-VL + LoRA adapter.  Returns (model, tokenizer)."""
    import torch
    from transformers import AutoProcessor
    from peft import PeftModel
    import json

    processor = AutoProcessor.from_pretrained(BASE_MODEL, trust_remote_code=True)
    tokenizer = getattr(processor, "tokenizer", processor)

    dtype = torch.bfloat16
    kw = {"trust_remote_code": True, "dtype": dtype, "device_map": "cuda"}

    # Try vision-language class first (Qwen2.5-VL requires it)
    try:
        from transformers import AutoModelForImageTextToText
        base = AutoModelForImageTextToText.from_pretrained(BASE_MODEL, **kw)
    except Exception:
        from transformers import AutoModelForCausalLM
        base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, **kw)

    model = PeftModel.from_pretrained(base, checkpoint_path)
    model.eval()
    return model, tokenizer


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------

def _sequence_logprob(model, tokenizer, prompt, target, max_length=4096):
    """Mean log probability of *target* tokens conditioned on *prompt*."""
    import torch

    suffix = "\nVerifier Output:\n"
    full_prompt = prompt + suffix
    full_text = full_prompt + target

    enc_full = tokenizer(
        full_text, return_tensors="pt", truncation=True, max_length=max_length
    )
    enc_prompt = tokenizer(
        full_prompt, return_tensors="pt", truncation=True, max_length=max_length
    )

    input_ids = enc_full["input_ids"].to(model.device)
    prompt_len = min(enc_prompt["input_ids"].shape[1], input_ids.shape[1] - 1)

    with torch.no_grad():
        logits = model(input_ids=input_ids).logits  # (1, seq, vocab)

    log_probs = torch.nn.functional.log_softmax(logits[0], dim=-1)
    target_ids = input_ids[0, prompt_len:]  # tokens we want to score
    if target_ids.numel() == 0:
        return 0.0
    pred_lps = log_probs[prompt_len - 1: prompt_len - 1 + target_ids.numel()]
    token_lps = pred_lps.gather(1, target_ids.unsqueeze(1)).squeeze(1)
    return token_lps.mean().item()


def _pairwise_scores(model, tokenizer, pair, max_length):
    """Return (pos_score, neg_score) for a pairwise test pair.

    pos_score = log P("A" | prompt where positive=A, negative=B)
    neg_score = log P("A" | prompt where negative=A, positive=B)
    A well-trained CG-PRM should give pos_score > neg_score.
    """
    from cg_prm.training.collator import serialize_trace

    pos_trace = pair["preferred_trace"]
    neg_trace = pair["rejected_trace"]
    question = pos_trace["question"]
    image_path = pos_trace["image_path"]
    pos_text = serialize_trace(pos_trace)
    neg_text = serialize_trace(neg_trace)

    header = (
        "You are a multimodal verifier.\n"
        "Given two reasoning traces for the same image and question, "
        "choose the more grounded trace.\n"
        "Respond with a JSON object containing `preferred_trace` set to `A` or `B`.\n\n"
        f"Image path: {image_path}\n"
        f"Question: {question}\n\n"
    )
    target_A = '{"preferred_trace": "A"}'

    prompt_AB = header + "Trace A:\n" + pos_text + "\n\nTrace B:\n" + neg_text + "\n"
    prompt_BA = header + "Trace A:\n" + neg_text + "\n\nTrace B:\n" + pos_text + "\n"

    pos_score = _sequence_logprob(model, tokenizer, prompt_AB, target_A, max_length)
    neg_score = _sequence_logprob(model, tokenizer, prompt_BA, target_A, max_length)
    return pos_score, neg_score


def _pointwise_scores(model, tokenizer, pair, max_length):
    """Return (pos_score, neg_score) for a pointwise test pair.

    Both traces are scored independently.  Score = log P(all-correct JSON target)
    given the trace.  A well-trained pointwise model should assign higher
    probability to the all-correct output when reading the positive trace.
    """
    from cg_prm.training.collator import serialize_trace

    def make_prompt(trace):
        question = trace["question"]
        image_path = trace["image_path"]
        trace_text = serialize_trace(trace)
        return (
            "You are a multimodal verifier.\n"
            "Given an image, a question, and a reasoning trace, "
            "judge whether each step is grounded.\n"
            "Return JSON with `step_labels`, `final_score`, and `trace_label`.\n\n"
            f"Image path: {image_path}\n"
            f"Question: {question}\n"
            "Trace:\n" + trace_text + "\n"
        )

    def all_positive_target(trace):
        n_steps = len(trace.get("steps", []))
        return json.dumps(
            {"step_labels": [1] * n_steps, "final_score": 1.0, "trace_label": 1}
        )

    pos_trace = pair["preferred_trace"]
    neg_trace = pair["rejected_trace"]
    pos_score = _sequence_logprob(
        model, tokenizer, make_prompt(pos_trace), all_positive_target(pos_trace), max_length
    )
    neg_score = _sequence_logprob(
        model, tokenizer, make_prompt(neg_trace), all_positive_target(neg_trace), max_length
    )
    return pos_score, neg_score


# ---------------------------------------------------------------------------
# Main evaluation routine
# ---------------------------------------------------------------------------

def load_model_and_evaluate(checkpoint_path, test_pairs_path, is_pairwise=True,
                              max_length=4096):
    """Load model + LoRA and compute (scores_pos, scores_neg, used_mock)."""
    try:
        import torch  # noqa: F401
        from peft import PeftModel  # noqa: F401
    except ImportError as e:
        print(f"  Warning: missing library ({e}) — falling back to mock")
        return *generate_mock_scores(test_pairs_path, is_pairwise), True

    if not Path(checkpoint_path).exists():
        print(f"  Warning: checkpoint not found at {checkpoint_path} — falling back to mock")
        return *generate_mock_scores(test_pairs_path, is_pairwise), True

    try:
        model, tokenizer = _load_model_and_tokenizer(checkpoint_path)
    except Exception as e:
        print(f"  Error loading model: {e}")
        print("  Falling back to mock evaluation")
        return *generate_mock_scores(test_pairs_path, is_pairwise), True

    test_pairs = [json.loads(l) for l in open(test_pairs_path)]
    score_fn = _pairwise_scores if is_pairwise else _pointwise_scores

    print(f"  Running inference on {len(test_pairs)} pairs...")
    scores_pos, scores_neg = [], []
    for i, pair in enumerate(test_pairs):
        ps, ns = score_fn(model, tokenizer, pair, max_length)
        scores_pos.append(ps)
        scores_neg.append(ns)
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(test_pairs)} done")

    return scores_pos, scores_neg, False


def generate_mock_scores(test_pairs_path, is_pairwise=True):
    """Randomised mock scores (used only when real evaluation fails)."""
    rng = np.random.default_rng(42)
    n = sum(1 for _ in open(test_pairs_path))
    if is_pairwise:
        pos = (0.65 + rng.random(n) * 0.25).tolist()
        neg = (0.20 + rng.random(n) * 0.25).tolist()
    else:
        pos = (0.55 + rng.random(n) * 0.30).tolist()
        neg = (0.35 + rng.random(n) * 0.30).tolist()
    return pos, neg


def evaluate(checkpoint_path, test_pairs_path, is_pairwise=True,
             use_mock=False, max_length=2048):
    """Evaluate one model.  Returns (scores_pos, scores_neg, used_mock)."""
    if use_mock or not Path(checkpoint_path).exists():
        print(f"  Using mock evaluation for {checkpoint_path}")
        return *generate_mock_scores(test_pairs_path, is_pairwise), True
    return load_model_and_evaluate(checkpoint_path, test_pairs_path,
                                   is_pairwise, max_length)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cg_prm",     required=True, help="CG-PRM checkpoint path")
    parser.add_argument("--pointwise",  required=True, help="Pointwise checkpoint path")
    parser.add_argument("--test_data",  required=True, help="Test pairs JSONL")
    parser.add_argument("--output",     required=True, help="Output JSON path")
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--use_mock",   action="store_true", help="Force mock evaluation")
    args = parser.parse_args()

    print("=" * 50)
    print("CG-PRM Mini-Experiment Evaluation")
    print("=" * 50)

    cg_exists = Path(args.cg_prm).exists()
    pw_exists  = Path(args.pointwise).exists()
    print(f"\nCheckpoint check:")
    print(f"  CG-PRM:    {'FOUND' if cg_exists else 'NOT FOUND'}")
    print(f"  Pointwise: {'FOUND' if pw_exists else 'NOT FOUND'}")

    force_mock = args.use_mock or not cg_exists or not pw_exists
    if force_mock:
        print("\n⚠  Running MOCK evaluation (checkpoints missing or --use_mock set)")
    else:
        print("\n✓ Running REAL evaluation with trained models")

    print("\nEvaluating CG-PRM (pairwise first-divergence)...")
    cg_pos, cg_neg, cg_mock = evaluate(
        args.cg_prm, args.test_data, is_pairwise=True,
        use_mock=force_mock, max_length=args.max_length
    )
    cg_auroc, cg_lower, cg_upper = bootstrap_ci(cg_pos, cg_neg)
    cg_label = "(MOCK)" if cg_mock else ""
    print(f"  AUROC: {cg_auroc:.4f} (95% CI: {cg_lower:.4f}–{cg_upper:.4f}) {cg_label}")

    print("\nEvaluating Pointwise baseline...")
    pw_pos, pw_neg, pw_mock = evaluate(
        args.pointwise, args.test_data, is_pairwise=False,
        use_mock=force_mock, max_length=args.max_length
    )
    pw_auroc, pw_lower, pw_upper = bootstrap_ci(pw_pos, pw_neg)
    pw_label = "(MOCK)" if pw_mock else ""
    print(f"  AUROC: {pw_auroc:.4f} (95% CI: {pw_lower:.4f}–{pw_upper:.4f}) {pw_label}")

    delta = cg_auroc - pw_auroc
    print(f"\n  Delta (CG-PRM − Pointwise): {delta:.4f}")

    used_mock = cg_mock or pw_mock
    go_condition      = delta >= 0.05 and cg_lower > pw_upper
    marginal_condition = 0.02 <= delta < 0.05

    if used_mock:
        decision = "NO-GO"
        reason = "Evaluation fell back to mock — train models first"
    elif go_condition:
        decision = "GO"
        reason = "CG-PRM significantly outperforms pointwise (delta ≥ 0.05, CIs non-overlapping)"
    elif marginal_condition:
        decision = "MARGINAL"
        reason = "Weak signal (0.02 ≤ delta < 0.05) — consider more data or debugging"
    else:
        decision = "NO-GO"
        if delta < 0:
            reason = "CG-PRM performs worse than pointwise — hypothesis rejected"
        elif delta < 0.02:
            reason = "Delta too small (< 0.02) — no meaningful improvement"
        elif cg_auroc < 0.55:
            reason = "CG-PRM AUROC < 0.55 — not learning grounding"
        else:
            reason = "CIs overlapping — insufficient confidence"

    print(f"\n{'=' * 50}")
    print(f"DECISION: {decision}")
    print(f"Reason: {reason}")
    print(f"{'=' * 50}")

    results = {
        "cg_prm":   {"auroc": cg_auroc, "ci_lower": cg_lower, "ci_upper": cg_upper},
        "pointwise": {"auroc": pw_auroc, "ci_lower": pw_lower, "ci_upper": pw_upper},
        "delta":    float(delta),
        "decision": decision,
        "reason":   reason,
        "mock_evaluation": used_mock,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output}")

    return {"GO": 0, "MARGINAL": 1}.get(decision, 2)


if __name__ == "__main__":
    raise SystemExit(main())
