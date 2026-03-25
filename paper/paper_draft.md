# CG-PRM: Counterfactual Grounding Supervision for Multimodal Verifiers

**Draft status:** Pre-results first draft  
**Purpose:** paper-structured manuscript draft based on the current proposal  
**Important:** no numerical claims should be added until experiments are run

---

## Abstract

Multimodal large language models increasingly produce long-form reasoning traces for charts, documents, and compositional visual reasoning tasks. However, current verifier and process reward model training still over-relies on outcome supervision, which allows a dangerous failure mode to persist: a model may produce a correct final answer while using incorrect or unsupported visual evidence along the way. This gap is particularly harmful for verifiable multimodal reasoning, where the central question is not only whether the answer is correct, but whether the answer is supported by evidence that is identified and used correctly.

We propose **CG-PRM**, a counterfactual-grounding supervision recipe for multimodal verifiers. The core idea is to train a visual process reward model not only on verified clean traces, but also on **grounding-specific counterfactual negatives** that preserve local fluency while corrupting visual faithfulness at a specific intermediate step. CG-PRM combines four elements: a unified step-level trace schema, automatic construction of verified clean traces, targeted corruption families for hard negative synthesis, and a LoRA-trained visual verifier that predicts both per-step groundedness and final trace quality. The method is designed to target the most dangerous failure mode in multimodal verification: **answer-correct but evidence-wrong** reasoning.

To test whether counterfactual supervision teaches genuine grounding rather than synthetic artifact detection, we organize the empirical study around **DocVQA** as a realistic document-grounding benchmark, **GQA** as a real-image compositional grounding benchmark, and **VisualWebBench** as a modern web-grounding and action-grounding benchmark with a narrower evaluation scope. The evaluation includes a human-written grounding challenge set on the two core scientific benchmarks, a cross-corruptor split, free-form robustness tests, and fixed-budget reranking analysis. Rather than presenting CG-PRM as a new agent or an outcome-optimization method, we position it as a method for improving the supervision signal available to multimodal verifiers. This draft lays out the method, problem formulation, and evaluation protocol that will support the final paper once experiments are completed.

---

## 1. Introduction

Multimodal reasoning systems are now evaluated not only by their ability to produce final answers, but by the quality of the intermediate traces they expose to users or downstream agents. In chart understanding, document question answering, and compositional visual reasoning, a long-form explanation is often interpreted as evidence that the system has actually reasoned from the image. Yet recent work on multimodal chain-of-thought and verifier training shows that this inference is unsafe: a model can generate a fluent, step-by-step explanation that is only weakly tied to the actual visual evidence.

This problem is especially acute for process reward models (PRMs) and related verifier-style components. PRMs are attractive because they promise step-level supervision and can be used to select better candidate traces at inference time. In practice, however, most current training signals still privilege **outcome correctness** or **trace plausibility** over **evidence-faithful reasoning**. As a result, PRMs can easily over-score traces that read well, end correctly, and even mention plausible evidence, while missing the fact that the cited evidence is wrong, irrelevant, or incorrectly used.

We argue that the missing ingredient in current multimodal verifier training is not more outcome supervision, but supervision that explicitly targets **whether evidence is used correctly**. The most important failure case is therefore not simply “wrong answer plus wrong trace.” The more dangerous failure is **answer-correct but evidence-wrong reasoning**, because it is easy to miss under standard evaluation and easy to over-credit during model training.

This paper studies that failure mode through the lens of **counterfactual grounding supervision**. We propose **CG-PRM**, a method for training multimodal verifiers on both clean and corrupted reasoning traces, where the corruptions are designed to preserve local fluency while breaking step-level grounding. The goal is not to build a new reasoning agent. The goal is to learn a verifier that better distinguishes grounded evidence use from fluent but unfaithful trace construction.

Our method contribution is paired with a rigorous identification protocol because the central risk in this line of work is false attribution. A verifier trained on synthetic negatives may simply learn corruption templates, teacher style, or structural priors rather than genuine grounding. For this reason, the paper is organized around one core scientific question:

> Does counterfactual grounding supervision improve multimodal verifier sensitivity to answer-correct but evidence-wrong traces?

To answer this question cleanly, we use **DocVQA** as the document-grounding benchmark, **GQA** as the real-image compositional grounding benchmark, and **VisualWebBench** as the modern agent-adjacent external validation benchmark. We evaluate on synthetic corruptions, an independent cross-corruptor split, a human-written grounding challenge set on `DocVQA` and `GQA`, and free-form traces with deterministic segmentation rules. We treat fixed-budget reranking as a stress test of verifier utility rather than as the main claim of the paper.

The paper makes three primary contributions:

1. We propose **CG-PRM**, a counterfactual-grounding supervision recipe for multimodal verifiers.
2. We construct a rigorous evaluation protocol to test whether the learned verifier is sensitive to real grounding failures rather than synthetic artifacts.
3. We provide a focused empirical study of the most dangerous verifier failure mode: answer-correct but evidence-wrong reasoning.

---

## 2. Related Work

### 2.1 Multimodal Reasoning Faithfulness and Verification

Recent work on multimodal reasoning has made clear that fluent chain-of-thought does not imply faithful evidence use. Text-only consistency and reflection methods, such as `CURE`, improve process quality but remain vulnerable to plausible-yet-ungrounded reasoning. More recent visual reasoning work has moved toward process supervision, verifier-guided inference, and explicit grounding, but the field still lacks a clean training signal that targets evidence use itself rather than only final outcome correctness.

### 2.2 Process Reward Models and Verifier Training

Process reward models have become an increasingly common tool for selecting or supervising reasoning trajectories. In language-only settings, PRMs are used to judge intermediate steps rather than only final answers. In multimodal settings, works such as `VisualPRM` and related verifier approaches suggest that step-level scoring can improve reasoning quality, but they do not fully solve the distinction between fluent reasoning and grounded reasoning. Our work focuses specifically on this gap: how to supervise a verifier so that it becomes sensitive to evidence misuse, even when the answer remains correct.

### 2.3 Evidence Attribution and Grounding

Another nearby line of work focuses on explicit evidence selection, grounding, or attribution. These approaches often ask which region, span, or token supports the answer. That question is essential, but it is not identical to our target question. A trace may refer to the correct evidence while still reading the wrong attribute, composing the relation incorrectly, or drawing a wrong inference from the correct facts. Our `correct-evidence but wrong-use` taxonomy is designed precisely to separate these cases from pure evidence localization.

### 2.4 Tool-Augmented and Executable Verification

Tool-augmented and executable reasoning methods offer stronger replayability and step-level checking, but they also introduce new cost and interface constraints. Our project intentionally remains in the verifier-training regime rather than expanding into a full agent or tool-orchestration system. The contribution is a supervision recipe for multimodal verifiers, not a new reasoning execution architecture.

---

## 3. Problem Formulation

Let an example be a tuple `(x, q, y)` where `x` is an image, `q` is a question, and `y` is the gold answer. A reasoning trace `τ` is a sequence of steps

`τ = (s_1, s_2, ..., s_T)`

where each step has structured fields:

- `step_text`
- `step_type`
- `grounding_ref`
- `evidence_value`
- `label`
- `error_type`

The verifier receives `(x, q, τ)` and predicts:

1. a per-step groundedness score for each `s_t`,
2. a final trace score used for ranking or filtering.

The target distinction is not merely between correct and incorrect traces. Instead, we focus on four broad trace regimes:

1. answer wrong, reasoning wrong
2. answer correct, reasoning correct
3. answer wrong, reasoning partly grounded
4. answer correct, reasoning ungrounded

The fourth regime is the scientifically central one. A verifier that cannot separate cases (2) and (4) is not suitable for evidence-sensitive reasoning systems.

---

## 4. Method

### 4.1 Overview

CG-PRM consists of four stages:

1. generate candidate clean traces from a strong multimodal teacher,
2. verify and retain only those traces that are consistent with benchmark-specific evidence rules,
3. synthesize grounding-specific counterfactual negatives by corrupting one step while preserving local fluency,
4. train a LoRA-based visual verifier to score step-level groundedness and final trace quality.

The method is intentionally lightweight. We use `Qwen2.5-VL-7B-Instruct` as the teacher generator and `Qwen2.5-VL-3B-Instruct` as the main verifier backbone. There is no reinforcement learning and no full-model fine-tuning.

### 4.2 Unified Trace Schema

All traces share a unified schema so that the same verifier can be trained across both benchmarks. Each step is represented as:

```json
{
  "step_id": 1,
  "step_text": "I locate the invoice number in the top header block.",
  "step_type": "locate",
  "grounding_ref": "ocr_span:17",
  "evidence_value": "INV-1042",
  "label": 1,
  "error_type": "none"
}
```

This representation serves three purposes. First, it provides a common interface across document and synthetic reasoning tasks. Second, it makes automatic verification easier to implement and audit. Third, it provides a clean unit of supervision for the PRM.

### 4.3 Benchmark-Specific Canonical Traces

For `DocVQA`, the canonical reasoning pattern is:

`locate document span -> extract OCR/text evidence -> derive answer`

For `GQA`, the canonical reasoning pattern is:

`identify relevant objects -> verify attributes/relations -> derive answer`

For `VisualWebBench`, the canonical reasoning pattern is:

`locate relevant UI element or evidence block -> read text/icon/layout cue -> derive answer or next action`

These templates are not the paper’s final claim. They are the controlled starting point that lets us ask whether the verifier learns grounding at all. The paper later tests how much of that signal survives as structure is relaxed.

### 4.4 Verified Clean Trace Construction

Clean traces are generated by the teacher model under multiple prompt templates and decoding settings. The key design principle is that the verifier should not be trained on raw teacher traces. Instead, traces are retained only if they pass benchmark-specific verification.

For `DocVQA`, verification is based on OCR-backed span alignment and answer normalization.  
For `GQA`, verification is based on scene-graph-backed object, attribute, and relation checks.  
For `VisualWebBench`, verification is restricted to subsettings with stable target actions, answer labels, or element-level evidence signals.

The output of this stage is a bank of evidence-faithful clean traces, plus a log of rejected traces and rejection reasons.

### 4.5 Counterfactual Grounding Negatives

The central method contribution is the generation of **grounding-specific hard negatives**. Each corrupted trace changes one step while preserving local fluency.

The five corruption families are:

1. wrong region
2. correct region, wrong value
3. wrong spatial or logical relation
4. irrelevant but plausible evidence
5. correct final answer, incorrect intermediate evidence

The fifth family is particularly important because it creates the core failure mode of interest: the answer is still correct, but the trace is not evidence-faithful.

### 4.6 Correct-Evidence but Wrong-Use Taxonomy

To distinguish CG-PRM from pure evidence-localization baselines, we define three kinds of wrong-use failures:

1. correct evidence, wrong attribute readout
2. correct evidence, wrong relation/composition
3. correct evidence, wrong inference from correct facts

In `DocVQA`, categories (1) and (3) are the most natural. In `GQA`, categories (2) and (3) dominate, while (1) appears as incorrect attribute binding over correctly identified objects. In `VisualWebBench`, categories (1) and (3) dominate, with category (2) appearing when the correct UI elements are identified but composed into the wrong action.

This taxonomy is critical because it makes the method-vs-evidence-supervision comparison empirical. If an evidence localizer can find the right anchor but cannot detect wrong evidence use, then a verifier remains necessary.

### 4.7 Verifier Architecture and Scoring

The verifier takes `(image, question, full trace)` as input and produces:

1. per-step groundedness scores,
2. a final trace score.

The final score is computed as the mean of step scores with an explicit penalty when a grounding-critical step falls below threshold. This rule prevents a trace with one catastrophic grounding failure from receiving a high score simply because the rest of the wording is strong.

### 4.8 Why This Method Should Work

The core intuition is that standard PRM supervision under-constrains the distinction between language quality and evidence quality. Counterfactual grounding negatives change this training signal. They force the model to confront near-miss traces where the answer may remain correct, the wording remains fluent, and only the evidence link is broken. If the verifier improves under such supervision, then the improvement is evidence for a more grounding-sensitive training signal, not merely for a larger or better-tuned verifier.

---

## 5. Experimental Design

### 5.1 Benchmarks

We use two core scientific benchmarks and one modern external-validation benchmark.

**DocVQA** serves as the realistic visual-text grounding benchmark because it requires evidence location and extraction under OCR noise and layout variation.  
**GQA** serves as the real-image compositional grounding benchmark because it retains structured semantics through scene graphs and relation annotations without relying on a dated synthetic environment.  
**VisualWebBench** serves as the modern agent-adjacent benchmark because it stresses web grounding and action grounding under contemporary interfaces, although with a narrower verification scope than the two core benchmarks.

`Agent-X` is intentionally left outside the core paper story and can be added later as supplementary transfer evidence if the main results stabilize early.

### 5.2 Baselines

Primary baselines:

1. clean-only visual PRM
2. pairwise verifier
3. strong VLM-as-a-judge
4. evidence-supervision baseline
5. matched-budget self-consistency

Secondary diagnostics:

1. answer-only scoring
2. text-only PRM

The strong judge baseline follows a fixed contract: one pre-chosen judge model, single-round judging, image+question+trace input, one prompt template across benchmarks, no self-consistency in the main comparison, and explicit cost reporting.

### 5.3 Human-Written Grounding Challenge Set

The gold evaluation artifact contains `100` matched trace pairs for each of the two core scientific benchmarks, `DocVQA` and `GQA`. Each pair consists of:

1. a grounded reference trace,
2. a human-written ungrounded trace matched on step count and overall style.

This set is evaluation-only. Its primary metric is paired ranking accuracy; its secondary metric is trace-level AUPRC. All reported scores will include bootstrap confidence intervals.

### 5.4 Free-Form Tiers

To study interface dependence, we evaluate on:

1. canonical traces
2. lightly structured traces
3. natural free-form traces

For natural free-form traces, segmentation uses a deterministic rule based on punctuation and a fixed set of discourse markers. No task-specific manual editing is allowed.

### 5.5 Main Analyses

The paper is organized around four interpretation tiers:

1. **Core claim:** does CG-PRM reduce false acceptance on answer-correct but evidence-wrong traces?
2. **Gold and transfer validation:** does that gain survive on human-written and cross-corruptor data?
3. **Utility test:** does any value remain under fixed-budget reranking?
4. **Interface dependence:** where does performance degrade as structure is removed?

This hierarchy is important because it prevents mixed results from collapsing the paper story.

### 5.6 Main-Text vs Supplementary Experiments

Main-text essentials:

1. clean-only PRM vs CG-PRM
2. schema-only vs schema+counterfactual
3. canonical vs natural free-form
4. one cross-corruptor split

Supplementary:

1. full corruption-family sweep
2. single-step vs multi-step analysis
3. aggregation sweep
4. wider cross-generator variants
5. full lightly structured analysis

---

## 6. Tables and Figures To Fill Later

### Table 1. Main verifier comparison on core scientific benchmarks

| Method | GQA step AUROC | GQA human pair acc | DocVQA step AUROC | DocVQA human pair acc | Notes |
|---|---:|---:|---:|---:|---|
| Clean-only visual PRM | TODO | TODO | TODO | TODO | baseline |
| Pairwise verifier | TODO | TODO | TODO | TODO | strong baseline |
| Strong VLM judge | TODO | TODO | TODO | TODO | cost reported separately |
| Evidence supervision | TODO | TODO | TODO | TODO | wrong-use subset critical |
| CG-PRM | TODO | TODO | TODO | TODO | main method |

### Table 2. External validation and free-form robustness

| Method | VisualWebBench main metric | Cross-corruptor | Canonical | Lightly structured | Natural free-form |
|---|---:|---:|---:|---:|---:|
| Clean-only visual PRM | TODO | TODO | TODO | TODO | TODO |
| CG-PRM | TODO | TODO | TODO | TODO | TODO |

### Table 3. Fixed-budget reranking analysis

| Budget | Self-consistency | Strong judge rerank | CG-PRM rerank | Main interpretation |
|---|---:|---:|---:|---|
| Low | TODO | TODO | TODO | TODO |
| Medium | TODO | TODO | TODO | TODO |
| High | TODO | TODO | TODO | TODO |

### Figure placeholders

1. pipeline overview
2. corruption family taxonomy
3. human challenge set examples
4. free-form degradation by tier

---

## 7. Limitations and Scope

This paper is intentionally scoped. It does not claim a new reasoning architecture, and it does not claim that counterfactual supervision solves multimodal faithfulness in the general case. The method starts from structured traces because that is the cleanest place to identify whether the supervision signal works at all. If performance weakens in natural free-form settings, the claim should be restricted accordingly.

Another limitation is dependence on verifiable subsets. For `DocVQA`, only a subset of examples admit reliable OCR-backed verification; for `VisualWebBench`, the study is deliberately restricted to subsettings with sufficiently stable targets or element-level evidence signals. These are necessary compromises for initial scientific identification, but they raise external-validity questions that the paper must report explicitly through coverage and subset-bias analysis.

Finally, reranking is not guaranteed to be a strong practical win. In this draft, reranking is treated as a stress test of verifier utility rather than as a promised headline result.

---

## 8. Conclusion

This draft argues for a method-centered view of multimodal verifier training. The core claim is that outcome supervision alone is insufficient for training evidence-sensitive verifiers, and that **counterfactual grounding supervision** offers a cleaner route toward detecting answer-correct but evidence-wrong reasoning traces. CG-PRM operationalizes this idea through verified clean traces, grounding-specific hard negatives, and a lightweight LoRA-trained visual verifier.

The final paper should stand or fall on one main question: whether this supervision recipe yields a verifier that is measurably more sensitive to evidence misuse than strong alternatives. If it does, then the contribution is a strong empirical method paper. If the gains are more moderate but still robust, the paper remains valuable as a method-plus-evaluation contribution for multimodal verifier grounding.

---

## 9. Writing TODOs

Before submission, fill in:

1. numerical results in all tables
2. exact baseline model names
3. implementation details for training
4. dataset coverage statistics
5. one paragraph per benchmark on observed failure modes
6. a short limitations paragraph tied to actual results
