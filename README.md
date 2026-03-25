# CG-PRM

Counterfactual Grounding Process Reward Models for Verifiable Multimodal Reasoning.

This repository contains the working codebase, proposal, and paper draft for the `CG-PRM` project. The current scientific scope is:

- core benchmarks: `DocVQA + GQA`
- modern external validation: `VisualWebBench`
- optional supplementary utility stress test: `Agent-X`

The central claim is narrow by design: counterfactual grounding supervision should make a multimodal verifier more sensitive to `answer-correct but evidence-wrong` reasoning traces.

## Current Project State

Implemented now:

- normalized benchmark adapters for `DocVQA`, `GQA`, and `VisualWebBench`
- shared trace schema and JSONL serialization
- deterministic segmentation for `canonical`, `light`, and `free` traces
- teacher request construction and teacher output parsing
- automatic verification for `DocVQA`, `GQA`, and weak verification for `VisualWebBench`
- counterfactual corruption generation, cross-corruptor generation, and `wrong_use` subsets
- pointwise and pairwise training-dataset builders
- reranking and verifier-side evaluation metrics
- minimal LoRA training scaffold
- one-entry pipeline driver for data preparation

Important boundary:

- teacher inference is still external to this repo
- the pipeline prepares requests and consumes generated teacher outputs, but it does not call a serving engine by itself

## Server Quickstart

On a fresh server:

```bash
git clone https://github.com/peter-cui-yi/cg-prm.git
cd cg-prm
bash scripts/setup_server.sh
```

What `setup_server.sh` does:

- creates `.venv` if missing
- upgrades `pip`, `setuptools`, and `wheel`
- installs `torch` by default
- installs the Python packages in `requirements.txt`
- optionally installs `bitsandbytes`

Useful environment variables:

- `PYTHON_BIN=/path/to/python3`
- `VENV_DIR=/path/to/venv`
- `TORCH_SPEC=torch==2.6.0`
- `TORCH_INDEX_URL=https://download.pytorch.org/whl/cu124`
- `INSTALL_TORCH=0`
- `INSTALL_BITSANDBYTES=1`

## First Pipeline Run

Use the provided template:

```bash
cp configs/pipeline_template.json /ABS/PATH/TO/pipeline.json
```

Then edit the benchmark paths in that copied config and run:

```bash
source .venv/bin/activate
bash scripts/first_run.sh /ABS/PATH/TO/pipeline.json
```

Expected first-run behavior:

- manifests are built
- teacher request JSONL files are written
- the pipeline usually stops at `waiting_for_teacher_outputs`

That stop is expected. At that point you should run your teacher model externally, write the corresponding `teacher_outputs.jsonl`, then run the same command again to unlock:

- teacher output parsing
- trace verification
- corruption generation
- pointwise and pairwise training-set export

## Training Entry Point

After the pipeline has produced training JSONL files, start from one of:

- `configs/pointwise_lora_template.json`
- `configs/pairwise_lora_template.json`

and launch:

```bash
python scripts/train_lora.py --config /ABS/PATH/TO/lora_config.json
```

This training layer is intentionally minimal. It is a lightweight LoRA SFT scaffold for verifier experiments, not yet a full production multimodal trainer.

## Repository Layout

```text
cg-prm/
├── README.md
├── requirements.txt
├── docs/
│   └── proposal.md
├── paper/
│   └── paper_draft.md
├── configs/
│   ├── pipeline_template.json
│   ├── pointwise_lora_template.json
│   └── pairwise_lora_template.json
├── scripts/
│   ├── build_manifests.py
│   ├── prepare_teacher_requests.py
│   ├── parse_teacher_outputs.py
│   ├── verify_traces.py
│   ├── build_corruptions.py
│   ├── build_training_dataset.py
│   ├── run_pipeline.py
│   ├── train_lora.py
│   ├── setup_server.sh
│   └── first_run.sh
└── src/cg_prm/
    ├── data/
    ├── generation/
    ├── verification/
    ├── corruption/
    ├── training/
    └── evaluation/
```

## Main Code Modules

- `src/cg_prm/data/`
  - benchmark adapters
  - normalized manifests
  - shared schema
- `src/cg_prm/generation/`
  - teacher prompts
  - trace parsing
  - deterministic segmentation
- `src/cg_prm/verification/`
  - benchmark-specific trace validation
- `src/cg_prm/corruption/`
  - main corruption families
  - cross-corruptor generation
  - `wrong_use` trace construction
- `src/cg_prm/training/`
  - dataset builders
  - collators
  - minimal LoRA trainer
- `src/cg_prm/evaluation/`
  - metrics
  - reranking logic

## Canonical Experiment Flow

1. Normalize raw benchmark data into JSONL manifests.
2. Build teacher requests from those manifests.
3. Generate teacher outputs outside this repo.
4. Parse teacher outputs into structured traces.
5. Verify clean traces.
6. Build counterfactual and cross-corruptor traces.
7. Export pointwise and pairwise training datasets.
8. Train the verifier with LoRA.
9. Evaluate with step-level metrics, ranking metrics, and reranking analysis.

## Key Files

- proposal: [docs/proposal.md](/Users/yicui/Documents/Github/cg-prm/docs/proposal.md)
- paper draft: [paper/paper_draft.md](/Users/yicui/Documents/Github/cg-prm/paper/paper_draft.md)
- pipeline template: [configs/pipeline_template.json](/Users/yicui/Documents/Github/cg-prm/configs/pipeline_template.json)
- setup script: [scripts/setup_server.sh](/Users/yicui/Documents/Github/cg-prm/scripts/setup_server.sh)
- first-run script: [scripts/first_run.sh](/Users/yicui/Documents/Github/cg-prm/scripts/first_run.sh)

## Notes For Running On Servers

- keep raw datasets outside the repo and point the config to absolute paths
- keep `summary_output` enabled so each run leaves a machine-readable progress record
- treat `VisualWebBench` as evaluation-first by default; the template excludes it from training-set assembly
- if your environment already provides PyTorch, run `INSTALL_TORCH=0 bash scripts/setup_server.sh`
