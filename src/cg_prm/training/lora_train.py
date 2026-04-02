"""Minimal LoRA training scaffold for CG-PRM verifier SFT."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from cg_prm.data.schema import read_jsonl
from cg_prm.training.collator import PairwiseTraceCollator, PointwiseTraceCollator


@dataclass(slots=True)
class LoRAConfig:
    """LoRA hyperparameters."""

    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    bias: str = "none"
    target_modules: list[str] = field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    )


@dataclass(slots=True)
class TrainingConfig:
    """Training config loaded from JSON."""

    model_name_or_path: str
    train_file: str
    output_dir: str
    task_type: str = "pointwise"
    eval_file: str | None = None
    max_length: int = 4096
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    num_train_epochs: float = 1.0
    learning_rate: float = 2e-4
    weight_decay: float = 0.0
    warmup_ratio: float = 0.03
    logging_steps: int = 10
    save_steps: int = 200
    save_total_limit: int = 2
    bf16: bool = True
    gradient_checkpointing: bool = True
    load_in_4bit: bool = False
    trust_remote_code: bool = True
    report_to: list[str] = field(default_factory=list)
    lora: LoRAConfig = field(default_factory=LoRAConfig)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "TrainingConfig":
        lora_payload = dict(payload.get("lora") or {})
        return cls(
            model_name_or_path=str(payload["model_name_or_path"]),
            train_file=str(payload["train_file"]),
            output_dir=str(payload["output_dir"]),
            task_type=str(payload.get("task_type", "pointwise")),
            eval_file=payload.get("eval_file"),
            max_length=int(payload.get("max_length", 4096)),
            per_device_train_batch_size=int(payload.get("per_device_train_batch_size", 1)),
            per_device_eval_batch_size=int(payload.get("per_device_eval_batch_size", 1)),
            gradient_accumulation_steps=int(payload.get("gradient_accumulation_steps", 8)),
            num_train_epochs=float(payload.get("num_train_epochs", 1.0)),
            learning_rate=float(payload.get("learning_rate", 2e-4)),
            weight_decay=float(payload.get("weight_decay", 0.0)),
            warmup_ratio=float(payload.get("warmup_ratio", 0.03)),
            logging_steps=int(payload.get("logging_steps", 10)),
            save_steps=int(payload.get("save_steps", 200)),
            save_total_limit=int(payload.get("save_total_limit", 2)),
            bf16=bool(payload.get("bf16", True)),
            gradient_checkpointing=bool(payload.get("gradient_checkpointing", True)),
            load_in_4bit=bool(payload.get("load_in_4bit", False)),
            trust_remote_code=bool(payload.get("trust_remote_code", True)),
            report_to=list(payload.get("report_to", [])),
            lora=LoRAConfig(
                r=int(lora_payload.get("r", 16)),
                alpha=int(lora_payload.get("alpha", 32)),
                dropout=float(lora_payload.get("dropout", 0.05)),
                bias=str(lora_payload.get("bias", "none")),
                target_modules=list(
                    lora_payload.get(
                        "target_modules",
                        [
                            "q_proj",
                            "k_proj",
                            "v_proj",
                            "o_proj",
                            "gate_proj",
                            "up_proj",
                            "down_proj",
                        ],
                    )
                ),
            ),
        )


class JsonListDataset:
    """Small in-memory dataset wrapper for Trainer."""

    def __init__(self, items: list[dict[str, Any]]) -> None:
        self.items = items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.items[index]


def load_config(path: str | Path) -> TrainingConfig:
    """Load a training config from JSON."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return TrainingConfig.from_dict(payload)


def _load_records(path: str | Path) -> list[dict[str, Any]]:
    return [dict(item) for item in read_jsonl(path)]


def _resolve_tokenizer(config: TrainingConfig):
    from transformers import AutoProcessor, AutoTokenizer

    try:
        processor = AutoProcessor.from_pretrained(
            config.model_name_or_path,
            trust_remote_code=config.trust_remote_code,
        )
        tokenizer = getattr(processor, "tokenizer", None) or processor
        return processor, tokenizer
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(
            config.model_name_or_path,
            trust_remote_code=config.trust_remote_code,
        )
        return None, tokenizer


def _resolve_model(config: TrainingConfig):
    import torch
    from transformers import AutoModelForCausalLM

    dtype = torch.bfloat16 if config.bf16 else torch.float16
    model_kwargs: dict[str, Any] = {"trust_remote_code": config.trust_remote_code}
    if config.load_in_4bit:
        model_kwargs["load_in_4bit"] = True
        model_kwargs["device_map"] = "auto"
    else:
        model_kwargs["dtype"] = dtype

    try:
        from transformers import AutoModelForImageTextToText

        return AutoModelForImageTextToText.from_pretrained(
            config.model_name_or_path,
            **model_kwargs,
        )
    except Exception:
        return AutoModelForCausalLM.from_pretrained(
            config.model_name_or_path,
            **model_kwargs,
        )


def _apply_lora(model, config: TrainingConfig):
    from peft import LoraConfig, TaskType, get_peft_model

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=config.lora.r,
        lora_alpha=config.lora.alpha,
        lora_dropout=config.lora.dropout,
        bias=config.lora.bias,
        target_modules=config.lora.target_modules,
    )
    return get_peft_model(model, lora_config)


def train_from_config(config: TrainingConfig) -> None:
    """Run LoRA SFT training from a loaded config."""
    from transformers import Trainer, TrainingArguments

    processor, tokenizer = _resolve_tokenizer(config)
    if getattr(tokenizer, "pad_token_id", None) is None and getattr(tokenizer, "eos_token", None):
        tokenizer.pad_token = tokenizer.eos_token

    train_records = _load_records(config.train_file)
    eval_records = _load_records(config.eval_file) if config.eval_file else None
    train_dataset = JsonListDataset(train_records)
    eval_dataset = JsonListDataset(eval_records) if eval_records is not None else None

    if config.task_type == "pointwise":
        data_collator = PointwiseTraceCollator(tokenizer=tokenizer, max_length=config.max_length)
    elif config.task_type == "pairwise":
        data_collator = PairwiseTraceCollator(tokenizer=tokenizer, max_length=config.max_length)
    else:
        raise ValueError("`task_type` must be `pointwise` or `pairwise`.")

    model = _resolve_model(config)
    model = _apply_lora(model, config)
    if config.gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        # Required when gradient checkpointing + PEFT: input embeddings must propagate grads
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
    
    # Workaround for DDP + LoRA + gradient checkpointing "ready twice" error
    if hasattr(model, "_set_static_graph"):
        model._set_static_graph()

    training_args = TrainingArguments(
        output_dir=config.output_dir,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        num_train_epochs=config.num_train_epochs,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_ratio=config.warmup_ratio,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        save_total_limit=config.save_total_limit,
        bf16=config.bf16,
        report_to=config.report_to,
        remove_unused_columns=False,
        eval_strategy="steps" if eval_dataset is not None else "no",
        eval_steps=config.save_steps if eval_dataset is not None else None,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        processing_class=processor if processor is not None else tokenizer,
    )
    trainer.train()
    trainer.save_model(config.output_dir)
    if processor is not None and hasattr(processor, "save_pretrained"):
        processor.save_pretrained(config.output_dir)
    elif hasattr(tokenizer, "save_pretrained"):
        tokenizer.save_pretrained(config.output_dir)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run minimal LoRA SFT training for CG-PRM verifier data."
    )
    parser.add_argument("--config", required=True, help="Path to a JSON training config.")
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    config = load_config(args.config)
    train_from_config(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
