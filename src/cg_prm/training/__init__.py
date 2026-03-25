"""Training-data builders for CG-PRM."""

from cg_prm.training.collator import (
    PairwiseTraceCollator,
    PointwiseTraceCollator,
    format_pairwise_example,
    format_pointwise_example,
    serialize_trace,
)
from cg_prm.training.dataset_builder import (
    PairwiseTrainingExample,
    PointwiseTrainingExample,
    build_pairwise_dataset,
    build_pointwise_dataset,
    load_traces,
)

__all__ = [
    "PairwiseTrainingExample",
    "PairwiseTraceCollator",
    "PointwiseTrainingExample",
    "PointwiseTraceCollator",
    "build_pairwise_dataset",
    "build_pointwise_dataset",
    "format_pairwise_example",
    "format_pointwise_example",
    "load_traces",
    "serialize_trace",
]
