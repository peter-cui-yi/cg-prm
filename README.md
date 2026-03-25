# CG-PRM

Counterfactual Grounding Process Reward Models for Verifiable Multimodal Reasoning

This folder is the working project root for the `CG-PRM` paper and experiments. The current source-of-truth proposal is [docs/proposal.md](/Users/yicui/Documents/Github/cg-prm/docs/proposal.md). The paper draft lives at [paper/paper_draft.md](/Users/yicui/Documents/Github/cg-prm/paper/paper_draft.md).

## Goal

Build a **method-first multimodal verifier project** around one sharp claim:

> Counterfactual grounding supervision can improve multimodal verifier sensitivity to **answer-correct but evidence-wrong** reasoning traces.

The project is not trying to build a new agent or a new RL pipeline. It is trying to build:

1. a clean supervision recipe for multimodal verifiers,
2. a reliable experimental architecture to test whether that recipe learns genuine grounding,
3. a paper-ready empirical story on `CLEVR` and `DocVQA`.

## Current Status

Already created:

- [docs/proposal.md](/Users/yicui/Documents/Github/cg-prm/docs/proposal.md): current proposal
- [paper/paper_draft.md](/Users/yicui/Documents/Github/cg-prm/paper/paper_draft.md): initial paper draft
- [src/cg_prm](/Users/yicui/Documents/Github/cg-prm/src/cg_prm): package root

Planned but not yet implemented:

- dataset adapters
- trace schema code
- teacher trace generation
- automatic verification
- counterfactual corruption pipeline
- PRM training
- evaluation and analysis scripts

## Directory Layout

```text
cg-prm/
├── README.md
├── docs/
│   └── proposal.md
├── paper/
│   └── paper_draft.md
├── src/
│   └── cg_prm/
│       └── __init__.py
├── configs/
├── scripts/
├── prompts/
├── data/
└── results/
```

Recommended expansion:

```text
src/cg_prm/
├── data/
│   ├── schema.py
│   ├── docvqa.py
│   ├── clevr.py
│   └── manifests.py
├── generation/
│   ├── teacher.py
│   ├── prompts.py
│   └── segmentation.py
├── verification/
│   ├── docvqa_rules.py
│   ├── clevr_rules.py
│   └── validators.py
├── corruption/
│   ├── base.py
│   ├── families.py
│   ├── cross_corruptor.py
│   └── wrong_use.py
├── training/
│   ├── dataset_builder.py
│   ├── collator.py
│   ├── lora_train.py
│   └── scoring.py
├── evaluation/
│   ├── baselines.py
│   ├── human_challenge.py
│   ├── reranking.py
│   ├── metrics.py
│   └── reports.py
└── utils/
    ├── io.py
    ├── logging.py
    └── seed.py
```

## Experimental Architecture

The system should be built as a sequence of independent, testable stages.

### Stage 1: Benchmark Adapters

Purpose:

- normalize `DocVQA` and `CLEVR` into one common example format
- avoid task-specific logic leaking into downstream training code

Required output format per example:

```json
{
  "example_id": "docvqa_train_000001",
  "benchmark": "docvqa",
  "image_path": "data/raw/docvqa/.../image.png",
  "question": "What is the invoice number?",
  "answer": "INV-1042",
  "metadata": {
    "split": "train",
    "source_id": "..."
  }
}
```

Implementation rules:

- `DocVQA` adapter must expose OCR spans or OCR-aligned tokens if available.
- `CLEVR` adapter must expose symbolic scene metadata for exact verification.
- never let later modules read raw benchmark files directly; all downstream code reads normalized manifests.

### Stage 2: Trace Schema

Purpose:

- define the single trace representation used across clean traces, corruptions, training, and evaluation

Canonical step record:

```json
{
  "image": "path-or-id",
  "question": "string",
  "step_id": 1,
  "step_text": "I locate the invoice number field near the header.",
  "step_type": "locate",
  "grounding_ref": "ocr_span:17",
  "evidence_value": "INV-1042",
  "label": 1,
  "error_type": "none"
}
```

Implementation rules:

- keep the on-disk format as `jsonl`
- define one parser and one serializer in `schema.py`
- make validation strict; reject malformed traces early

### Stage 3: Free-Form Segmentation

Purpose:

- convert lightly structured or natural free-form outputs into step units reproducibly

Hard rule:

- segmentation must use a **deterministic rule** based on punctuation and a fixed list of discourse markers
- no task-specific manual adjustment
- no post-hoc tuning based on model success/failure

Minimum API:

```python
segment_trace(text: str, mode: str) -> list[str]
```

Where `mode` is one of:

- `canonical`
- `light`
- `free`

### Stage 4: Teacher Trace Generation

Purpose:

- generate candidate clean traces with `Qwen2.5-VL-7B-Instruct`

Requirements:

- support multiple prompt templates per benchmark
- support multiple decoding regimes
- store raw generations before filtering
- keep generation metadata:
  - model name
  - prompt id
  - decoding params
  - seed

Artifacts:

- `data/intermediate/<benchmark>/teacher_raw.jsonl`
- `data/intermediate/<benchmark>/teacher_segmented.jsonl`

### Stage 5: Automatic Verification

Purpose:

- filter clean traces and assign step labels where possible

`DocVQA` verification:

- verify answer-supporting span alignment
- verify whether the cited span actually contains or normalizes to the stated evidence

`CLEVR` verification:

- use scene graph / symbolic annotations for exact object, relation, and attribute checks

Hard rule:

- verification rules must be benchmark-specific but output the same schema
- log both pass/fail and failure reason

Artifacts:

- `data/verified/<benchmark>/clean_verified.jsonl`
- `data/verified/<benchmark>/clean_rejected.jsonl`

### Stage 6: Counterfactual Corruption

Purpose:

- generate grounding-specific hard negatives from verified clean traces

Required corruption families:

1. wrong region
2. correct region, wrong value
3. wrong relation or logical composition
4. irrelevant but plausible evidence
5. correct final answer, wrong intermediate evidence

Required additional split:

- one **independent cross-corruptor** implementation

Required subset:

- `correct-evidence but wrong-use`, with three subtypes:
  - wrong attribute readout
  - wrong relation/composition
  - wrong inference from correct facts

Artifacts:

- `data/corrupted/<benchmark>/train_main.jsonl`
- `data/corrupted/<benchmark>/eval_cross_corruptor.jsonl`
- `data/corrupted/<benchmark>/eval_wrong_use.jsonl`

### Stage 7: Human Challenge Set

Purpose:

- provide gold evaluation beyond synthetic corruptions

Spec:

- `100` matched trace pairs per benchmark
- only for `CLEVR` and `DocVQA`
- evaluation-only
- double annotation plus adjudication

Store:

- `data/human_challenge/docvqa.jsonl`
- `data/human_challenge/clevr.jsonl`
- `data/human_challenge/annotation_protocol.md`

Each record must include:

- grounded reference trace
- ungrounded trace
- scenario label
- answer-correctness label
- annotator ids
- adjudication status

### Stage 8: PRM Training

Purpose:

- train the `3B` verifier with LoRA only

Training data:

- verified clean traces
- main corruption traces
- optional balanced sampling by benchmark and error family

Required model outputs:

- per-step groundedness score
- final trace score

Required scoring rule:

- mean of step scores
- explicit penalty on grounding-critical steps below threshold

Training constraints:

- `3B` is the main model
- no RL
- no full-model tuning
- no hidden task-specific shortcuts in the collator

### Stage 9: Evaluation

Main-text evaluations:

1. clean-only visual PRM vs CG-PRM
2. pairwise verifier baseline
3. strong VLM-as-a-judge baseline
4. evidence-supervision baseline
5. one cross-corruptor split
6. one human challenge set
7. one schema robustness study
8. one fixed-budget reranking study

Supplementary evaluations:

- full corruption-family sweeps
- wider cross-generator variants
- `ChartQA` transfer
- richer aggregation sweeps

## Strong Judge Contract

This baseline must stay fixed once chosen.

Rules:

1. choose one main judge model before running the full evaluation
2. use **single-round judging** for the main paper table
3. judge sees `image + question + full trace`
4. use one prompt template across benchmarks, with only minimal benchmark tags
5. do not use judge self-consistency in the main comparison
6. report judge cost separately
7. if you add a stronger closed-source judge, keep it supplementary

## Main Metrics

Primary:

- step-level `AUROC`
- step-level `AUPRC`
- paired ranking accuracy on human challenge set

Secondary:

- trace-level ranking quality
- fixed-budget reranking utility
- coverage rate
- subset-bias report

Interpretation rule:

- small gains near run-to-run variance or bootstrap uncertainty are not enough
- answer-correct but evidence-wrong performance is the main scientific axis

## Build Order For Codex

Codex should implement in this order:

1. `schema.py` and manifests
2. benchmark adapters for `DocVQA` and `CLEVR`
3. deterministic free-form segmentation
4. teacher generation runner
5. automatic verification for both benchmarks
6. corruption family implementations
7. cross-corruptor generator
8. human challenge annotation tooling
9. LoRA training pipeline
10. evaluation scripts
11. result summarization and paper-table export

Do not start training before steps `1-7` are stable.

## Pipeline Driver

The project now includes a single orchestration entry point:

- [scripts/run_pipeline.py](/Users/yicui/Documents/Github/cg-prm/scripts/run_pipeline.py)
- [configs/pipeline_template.json](/Users/yicui/Documents/Github/cg-prm/configs/pipeline_template.json)

This driver runs the available end-to-end pipeline:

1. build benchmark manifests
2. prepare teacher requests
3. parse teacher outputs
4. verify clean traces
5. build main / cross-corruptor / wrong-use corruptions
6. build pointwise and pairwise training datasets

Important constraint:

- teacher inference itself is still external
- the pipeline expects `teacher_outputs_input` paths in the config if you want to proceed beyond request generation
- `main` corruptions feed training, while `cross` and `wrong_use` remain evaluation artifacts

## First Runnable Milestone

The first milestone is not “train the final model.” It is:

1. load both benchmarks through normalized adapters
2. generate `100-200` teacher traces per benchmark
3. verify at least a small clean subset
4. produce all five corruption families for that subset
5. export a training-ready `jsonl`

If this milestone is not solid, do not proceed to LoRA training.

## Acceptance Checklist

Before calling the architecture “ready,” verify:

- traces serialize and deserialize cleanly
- segmentation is deterministic
- verification failures are logged with reasons
- every corruption family preserves local fluency
- cross-corruptor outputs are distributionally distinct from main corruptor outputs
- human challenge format is frozen
- strong judge contract is frozen
- evaluation scripts can reproduce metrics from saved trace files without hidden state

## Non-Goals

Do not let the project drift into:

- new agent architecture work
- reinforcement learning
- full-model fine-tuning
- broad benchmark expansion before the `CLEVR + DocVQA` story is stable
- paper claims built around weak or noisy reranking gains

## Immediate Next Files To Implement

When Codex starts coding, the first concrete files should be:

1. `src/cg_prm/data/schema.py`
2. `src/cg_prm/data/docvqa.py`
3. `src/cg_prm/data/clevr.py`
4. `src/cg_prm/generation/segmentation.py`
5. `src/cg_prm/verification/docvqa_rules.py`
6. `src/cg_prm/verification/clevr_rules.py`
7. `src/cg_prm/corruption/families.py`

After those exist, add training and evaluation.
