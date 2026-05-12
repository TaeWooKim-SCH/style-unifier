"""src.encoding 패키지 공개 API."""

from src.encoding.batch_consistency import (
    StyleStatistics,
    enforce_consistency,
    extract_style_statistics,
)
from src.encoding.ip_adapter_wrapper import StyleEncoder
from src.encoding.multi_ref import aggregate_references, compute_clip_similarities
from src.encoding.shared_attention import (
    SharedKVAttnProcessor,
    apply_shared_attention,
    remove_shared_attention,
    validate_batch_for_shared_attention,
)

__all__ = [
    "StyleEncoder",
    "StyleStatistics",
    "SharedKVAttnProcessor",
    "aggregate_references",
    "apply_shared_attention",
    "compute_clip_similarities",
    "enforce_consistency",
    "extract_style_statistics",
    "remove_shared_attention",
    "validate_batch_for_shared_attention",
]
