"""Multi-reference 임베딩 집계 모듈 (그룹 D4).

StyleEncoder.encode_multi()를 사용자 친화적인 함수 인터페이스로 노출한다.
Gradio UI 및 StyleUnificationPipeline 통합 지점으로 작동한다.

지원 전략:
- "mean"    : uniform 평균 (기본).
- "weighted": target과의 CLIP 유사도 softmax 가중 (target 필수, 없으면 mean fallback).
- "first"   : references[0]만 사용 (single-reference baseline).
- "manual"  : 명시적 weights 리스트 지정.
"""

from __future__ import annotations

import torch
import torch.nn.functional as functional
from PIL import Image

from src.encoding.ip_adapter_wrapper import StyleEncoder
from src.utils.device import get_device
from src.utils.logging import get_logger

logger = get_logger(__name__)

_VALID_STRATEGIES = frozenset({"mean", "weighted", "first", "manual"})


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _normalize_weights(weights: list[float], n: int) -> torch.Tensor:
    """Weights 리스트를 검증하고 합이 1인 float32 Tensor로 반환한다.

    Args:
        weights: 가중치 리스트.
        n: references 개수.

    Returns:
        (n,) float32 Tensor, 합 = 1.

    Raises:
        ValueError: 길이 불일치 또는 모든 원소의 합이 0일 때.
    """
    if len(weights) != n:
        raise ValueError(f"weights length must match references length ({n}), got {len(weights)}.")
    total = sum(weights)
    if total == 0:
        raise ValueError("Sum of weights must not be zero.")
    return torch.tensor(weights, dtype=torch.float32) / total


def _weighted_sum_embeddings(
    embeddings: list[torch.Tensor],
    weights_t: torch.Tensor,
) -> torch.Tensor:
    """가중합으로 임베딩 리스트를 단일 (1, seq_len, dim) Tensor로 집계한다.

    Args:
        embeddings: 각각 (1, seq_len, dim) CPU float32 Tensor 리스트.
        weights_t: (N,) float32 Tensor. 합이 1이어야 한다.

    Returns:
        (1, seq_len, dim) float32 Tensor.
    """
    stacked = torch.stack([e.squeeze(0) for e in embeddings], dim=0)  # (N, seq, dim)
    result: torch.Tensor = (weights_t[:, None, None] * stacked).sum(dim=0, keepdim=True)
    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def aggregate_references(
    references: list[Image.Image],
    *,
    strategy: str = "mean",
    weights: list[float] | None = None,
    target: Image.Image | None = None,
    style_encoder: StyleEncoder | None = None,
    device: torch.device | None = None,
) -> torch.Tensor:
    """다중 reference를 하나의 IP-Adapter 임베딩으로 집계한다.

    Args:
        references: PIL Image 리스트 (N >= 1).
        strategy: 집계 전략.

            - ``"mean"``: uniform 평균.
            - ``"weighted"``: target과의 CLIP 유사도 softmax 가중 (target 필수).
              target이 None이면 "mean"으로 fallback하고 WARNING을 기록한다.
            - ``"first"``: references[0]만 반환 (single-reference baseline).
            - ``"manual"``: ``weights`` 인자로 명시적 가중치를 지정한다.

        weights: ``strategy="manual"`` 일 때 사용. ``len(weights) == len(references)``
            이어야 하며, 합이 1로 자동 정규화된다.
        target: ``strategy="weighted"`` 일 때 가중치 기준으로 사용할 변환 대상 이미지.
            None이면 mean fallback + WARNING.
        style_encoder: 외부에서 생성한 StyleEncoder 인스턴스.
            None이면 내부에서 ``StyleEncoder(device=device)`` 로 생성한다.
        device: 추론 device. None이면 ``get_device()`` 로 자동 선택.
            style_encoder가 제공되면 이 인자는 무시된다.

    Returns:
        (1, seq_len, dim) float32 torch.Tensor (CPU).
        IP-Adapter Plus (ViT-H) 기준으로 seq_len=257, dim=1280.
        ``pipe(..., ip_adapter_image_embeds=[emb.to(device)])`` 로 전달 가능.

    Raises:
        ValueError: references가 빈 리스트일 때.
        ValueError: strategy가 지원되지 않는 값일 때.
        ValueError: ``strategy="manual"`` 인데 weights가 None일 때.
        ValueError: weights 길이가 references 길이와 다를 때.
        ValueError: weights 합이 0일 때.

    Example:
        >>> refs = [Image.open("style1.png"), Image.open("style2.png")]
        >>> emb = aggregate_references(refs, strategy="mean")
        >>> # IP-Adapter에 전달:
        >>> # pipe(..., ip_adapter_image_embeds=[emb.to(device)])
    """
    if not references:
        raise ValueError("references must not be empty.")
    if strategy not in _VALID_STRATEGIES:
        raise ValueError(f"strategy must be one of {sorted(_VALID_STRATEGIES)}, got {strategy!r}.")

    resolved_device = device if device is not None else get_device()
    encoder = style_encoder if style_encoder is not None else StyleEncoder(device=resolved_device)

    n = len(references)
    logger.info(
        "aggregate_references: N=%d, strategy=%r, device=%s",
        n,
        strategy,
        resolved_device,
    )

    # N=1 → strategy 무관하게 encode_single 직접 호출
    if n == 1:
        logger.debug("Single reference — encode_single regardless of strategy.")
        return encoder.encode_single(references[0])

    # manual 전략: weights 검증 + 가중합
    if strategy == "manual":
        if weights is None:
            raise ValueError("strategy='manual' requires weights to be provided.")
        weights_t = _normalize_weights(weights, n)
        logger.debug("Manual weights (normalized): %s", weights_t.tolist())
        embeddings = [encoder.encode_single(ref) for ref in references]
        return _weighted_sum_embeddings(embeddings, weights_t)

    # weighted 전략: target 없으면 mean fallback
    if strategy == "weighted":
        if target is None:
            logger.warning(
                "strategy='weighted' requires target image, but target=None. Falling back to strategy='mean'."
            )
            return encoder.encode_multi(references, strategy="mean")
        target_embedding = encoder.encode_single(target)
        return encoder.encode_multi(
            references,
            strategy="weighted",
            target_embedding=target_embedding,
        )

    # mean / first → StyleEncoder.encode_multi에 위임
    return encoder.encode_multi(references, strategy=strategy)


def compute_clip_similarities(
    references: list[Image.Image],
    target: Image.Image,
    *,
    style_encoder: StyleEncoder | None = None,
    device: torch.device | None = None,
) -> list[float]:
    """각 reference와 target 사이의 CLIP 유사도를 반환한다 (디버깅/시각화용).

    IP-Adapter Plus CLIP 임베딩의 시퀀스 방향 평균 벡터(mean pool)를 사용해
    reference별 cosine similarity를 계산한다. ``strategy="weighted"`` 내부
    가중치를 시각화할 때 유용하다.

    Args:
        references: N개 PIL Image.
        target: 변환 대상 PIL Image.
        style_encoder: 인스턴스 재사용. None이면 내부에서 생성.
        device: 추론 device. None이면 ``get_device()`` 로 자동 선택.
            style_encoder가 제공되면 이 인자는 무시된다.

    Returns:
        N개 cosine similarity float 리스트 (-1 ~ 1, 실제로는 0.3 ~ 0.95 범위).
        i번째 원소가 references[i]와 target의 유사도.

    Raises:
        ValueError: references가 빈 리스트일 때.

    Example:
        >>> sims = compute_clip_similarities([ref1, ref2], target)
        >>> print(sims)  # [0.82, 0.67]
    """
    if not references:
        raise ValueError("references must not be empty.")

    resolved_device = device if device is not None else get_device()
    encoder = style_encoder if style_encoder is not None else StyleEncoder(device=resolved_device)

    logger.info(
        "compute_clip_similarities: N=%d references, device=%s",
        len(references),
        resolved_device,
    )

    # (1, seq_len, dim) → (dim,) mean-pooled vector
    target_emb = encoder.encode_single(target)
    target_vec = target_emb.squeeze(0).mean(dim=0)  # (dim,)

    similarities: list[float] = []
    for idx, ref in enumerate(references):
        ref_emb = encoder.encode_single(ref)
        ref_vec = ref_emb.squeeze(0).mean(dim=0)  # (dim,)
        sim: float = functional.cosine_similarity(
            ref_vec.unsqueeze(0), target_vec.unsqueeze(0), dim=-1
        ).item()
        similarities.append(sim)
        logger.debug("ref[%d] cosine similarity with target: %.4f", idx, sim)

    return similarities
