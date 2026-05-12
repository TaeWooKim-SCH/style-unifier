"""§5.5.2 — Cross-image attention (StyleAligned 응용).

WARNING: 검증 단계 architectural approach — ADR-010.

이 모듈은 SDXL UNet self-attention의 K/V를 batch dim 0(reference)에서 가져와
batch dim 1..N(생성 대상)에 broadcast하여 mechanism level의 배치 일관성을 시도한다.
첫 시도에 작동 보장 없으며, A4 vs A5 ablation으로 채택 여부 결정.

알려진 위험 (§5.5.2 그대로):
1. ControlNet 충돌 — reference 형태가 출력에 새어 들어갈 가능성
2. VRAM 부담 — RTX 4070 12GB에서 N <= 4 권장
3. 단일 이미지 품질 저하 — reference 과적합
4. tensor shape 디버깅 필요

본 모듈은 apply_shared_attention(pipeline) 단일 진입점을 제공한다.
실패해도 D1 (enforce_consistency) 후처리 baseline이 fallback 보장.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F  # noqa: N812
from PIL import Image

from src.utils.logging import get_logger

logger = get_logger(__name__)

# RTX 4070 12GB 기준 권장 상한. 실험 결과에 따라 조정 가능.
_DEFAULT_MAX_BATCH_SIZE = 4


class SharedKVAttnProcessor:
    """Self-attention K/V를 batch[0] (reference)에서 가져와 batch[1..N]에 broadcast.

    WARNING: 검증 단계 — ADR-010. 첫 시도에 작동 보장 없음.

    diffusers 0.27.2의 ``AttnProcessor2_0`` 인터페이스를 따른다.
    ``scaled_dot_product_attention`` (PyTorch 2.0+)을 사용하므로 PyTorch 2.0 이상 필수.

    입력 가정 — 강하게 강제:
        - batch[0]은 reference 이미지의 latent.
        - batch[1..N]은 생성 대상 source latent.
        - 이 순서가 깨지면 결과가 무의미.
        - ``apply_shared_attention`` docstring의 Example 참조.

    Attributes:
        share_layers: 미사용. ``apply_shared_attention`` 에서 layer 선택 담당.
            향후 확장을 위해 시그니처에 유지.
    """

    def __init__(self, share_layers: list[int] | None = None) -> None:
        """초기화.

        Args:
            share_layers: 미사용 (``apply_shared_attention`` 에서 layer 선택 담당).
                향후 확장을 위해 시그니처에 유지.

        Raises:
            ImportError: PyTorch 2.0 미만 (scaled_dot_product_attention 없음).
        """
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                "SharedKVAttnProcessor requires PyTorch 2.0+. Please upgrade: pip install 'torch>=2.0'"
            )
        self.share_layers = share_layers

    def __call__(
        self,
        attn: Any,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        temb: torch.Tensor | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Self-attention 호출. self-attention일 때만 K/V 공유 적용.

        ``encoder_hidden_states`` 가 None (self-attention) 일 때만 K/V broadcast.
        Cross-attention (encoder_hidden_states is not None) 은 표준 처리.

        Args:
            attn: ``diffusers.models.attention_processor.Attention`` 인스턴스.
            hidden_states: ``(B, seq, dim)`` 또는 ``(B, C, H, W)`` tensor.
            encoder_hidden_states: cross-attention용 텍스트 임베딩. None이면 self-attention.
            attention_mask: optional attention mask.
            temb: optional timestep embedding (spatial_norm 용).
            *args: deprecated scale 인자 등 diffusers 호환.
            **kwargs: 추가 diffusers 인자.

        Returns:
            attention 결과 tensor. 입력과 동일한 shape.
        """
        is_self_attention = encoder_hidden_states is None

        # spatial_norm 처리 (일부 UNet block에서 사용)
        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        # (B, C, H, W) -> (B, H*W, C) 변환. 복원을 위해 shape 저장.
        channel: int = 0
        height: int = 0
        width: int = 0
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            # scaled_dot_product_attention 은 (batch, heads, src_len, tgt_len) mask를 기대
            prepared: torch.Tensor = attn.prepare_attention_mask(
                attention_mask, sequence_length, batch_size
            )
            attention_mask = prepared.view(batch_size, attn.heads, -1, prepared.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        # cross-attention이면 encoder_hidden_states에서 K/V 계산 (공유 불가)
        if encoder_hidden_states is None:
            kv_source = hidden_states
        elif attn.norm_cross:
            kv_source = attn.norm_encoder_hidden_states(encoder_hidden_states)
        else:
            kv_source = encoder_hidden_states

        key = attn.to_k(kv_source)
        value = attn.to_v(kv_source)

        # self-attention이고 batch >= 2일 때만 K/V broadcast 수행
        if is_self_attention:
            if batch_size < 2:
                logger.warning(
                    "SharedKVAttnProcessor called with B=%d < 2; skipping K/V share. 최소 reference + source 1개 필요.",
                    batch_size,
                )
            else:
                key, value = _broadcast_kv_from_reference(key, value, batch_size)

        # multi-head reshape: (B, seq, dim) -> (B, heads, seq, head_dim)
        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # scaled dot-product attention (PyTorch 2.0+)
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        # (B, heads, seq, head_dim) -> (B, seq, inner_dim)
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # linear proj + dropout
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        # (B, C, H, W) 로 복원
        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(
                batch_size, channel, height, width
            )

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


def _broadcast_kv_from_reference(
    key: torch.Tensor,
    value: torch.Tensor,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """batch[0] (reference)의 K/V를 batch[1..N]에 broadcast한다.

    batch[0] 자체는 원래 K/V를 유지하고, batch[1..N]만 reference K/V로 교체한다.

    Args:
        key: ``(B, seq, inner_dim)`` tensor.
        value: ``(B, seq, inner_dim)`` tensor.
        batch_size: batch 크기 B. 2 이상 보장된 상태에서 호출해야 함.

    Returns:
        (key_shared, value_shared) 각 ``(B, seq, inner_dim)`` tensor.
    """
    ref_k = key[0:1]  # (1, seq, inner_dim)
    ref_v = value[0:1]  # (1, seq, inner_dim)

    shared_k = ref_k.expand(batch_size - 1, -1, -1)  # (N, seq, inner_dim)
    shared_v = ref_v.expand(batch_size - 1, -1, -1)  # (N, seq, inner_dim)

    # batch[0]: 원래 K/V 유지. batch[1..N]: reference K/V로 교체.
    key_out = torch.cat([ref_k, shared_k], dim=0)  # (B, seq, inner_dim)
    value_out = torch.cat([ref_v, shared_v], dim=0)  # (B, seq, inner_dim)

    logger.debug(
        "K/V broadcast: ref shape=%s, broadcasted to B=%d",
        list(ref_k.shape),
        batch_size,
    )
    return key_out, value_out


def apply_shared_attention(
    pipeline: Any,
    *,
    share_layers: list[int] | None = None,
    max_batch_size: int = _DEFAULT_MAX_BATCH_SIZE,
) -> None:
    """Pipeline의 UNet self-attention processor를 ``SharedKVAttnProcessor`` 로 교체.

    WARNING: 검증 단계 — ADR-010. 호출 후 pipeline.pipe(...) 사용 시
    batch dim 0이 reference, 1..N이 source라는 가정으로 동작한다.
    이 가정이 깨지면 결과가 무의미.

    VRAM 가드: max_batch_size 초과 입력은 pipeline 호출 직전에
    ``validate_batch_for_shared_attention`` 로 별도 차단해야 한다.

    Args:
        pipeline: ``StableDiffusionXLControlNetImg2ImgPipeline`` 또는 호환 인스턴스.
            ``pipeline.unet`` 속성 필요.
        share_layers: 공유할 self-attention layer 인덱스 리스트 (0-based).
            None이면 모든 self-attention layer에서 공유.
            예: ``[10, 15, 20]`` — mid-block 근처 일부 layer만 공유.
            일부 layer 공유 시 ControlNet과의 충돌이 완화될 수 있음 (실험 검증 필요).
        max_batch_size: 동시 처리 batch 상한. pipeline 메타데이터로 저장.
            RTX 4070 12GB 기준 N <= 4 권장.

    Raises:
        AttributeError: pipeline에 ``unet`` 속성 없을 때.
        ImportError: diffusers가 설치되지 않았을 때.
        ValueError: ``share_layers`` 에 음수 또는 총 self-attention layer 수를 초과하는 인덱스.

    Example:
        >>> apply_shared_attention(pipeline.pipe, share_layers=[10, 15], max_batch_size=4)
        >>> batch_imgs = [reference, source1, source2, source3]
        >>> validate_batch_for_shared_attention(batch_imgs, max_batch_size=4)
        >>> outputs = pipeline.pipe(image=batch_imgs, ...).images[1:]  # batch[0] 버림
    """
    unet = getattr(pipeline, "unet", None)
    if unet is None:
        raise AttributeError(
            f"pipeline 인스턴스 {type(pipeline).__name__!r} 에 'unet' 속성이 없습니다. StableDiffusionXLControlNetImg2ImgPipeline 또는 호환 pipeline을 전달하세요."
        )

    # diffusers Attention 클래스 — lazy import (macOS import sanity 유지)
    try:
        from diffusers.models.attention_processor import Attention
    except ImportError as exc:
        raise ImportError(
            "diffusers Attention 클래스를 찾을 수 없습니다. pip install 'diffusers>=0.27.2'"
        ) from exc

    # self-attention layer 수집
    self_attn_modules: list[tuple[str, Any]] = []
    for name, module in unet.named_modules():
        if isinstance(module, Attention) and not module.is_cross_attention:
            self_attn_modules.append((name, module))

    total_self_attn = len(self_attn_modules)
    logger.info(
        "apply_shared_attention: UNet에서 self-attention layer %d개 발견 (share_layers=%s)",
        total_self_attn,
        share_layers,
    )

    if total_self_attn == 0:
        logger.warning(
            "self-attention layer를 찾지 못했습니다. UNet 구조가 예상과 다를 수 있습니다. SharedKVAttnProcessor 미적용."
        )
        return

    # share_layers 인덱스 검증
    if share_layers is not None:
        _validate_share_layers(share_layers, total_self_attn)
        target_indices = set(share_layers)
    else:
        target_indices = set(range(total_self_attn))

    # 대상 layer에만 SharedKVAttnProcessor 적용. 원본 processor를 dict로 보존.
    original_processors: dict[str, Any] = {}
    applied_count = 0
    for layer_idx, (name, module) in enumerate(self_attn_modules):
        if layer_idx in target_indices:
            original_processors[name] = module.processor
            module.set_processor(SharedKVAttnProcessor())  # type: ignore[arg-type]
            applied_count += 1
            logger.debug("SharedKVAttnProcessor 적용: layer_idx=%d, name=%s", layer_idx, name)

    logger.info(
        "apply_shared_attention 완료: %d/%d self-attention layer에 SharedKVAttnProcessor 적용.",
        applied_count,
        total_self_attn,
    )

    # 후속 검증 및 remove_shared_attention 에서 사용하는 메타데이터
    pipeline._shared_attention_active = True
    pipeline._shared_attention_max_batch = max_batch_size
    pipeline._shared_attention_total_layers = total_self_attn
    pipeline._shared_attention_applied_count = applied_count
    # remove_shared_attention 에서 원본 복원에 사용
    pipeline._shared_attention_original_processors = original_processors


def _validate_share_layers(share_layers: list[int], total_self_attn: int) -> None:
    """share_layers 인덱스 유효성 검사.

    Args:
        share_layers: 검사할 인덱스 리스트.
        total_self_attn: 실제 self-attention layer 총수.

    Raises:
        ValueError: 음수 인덱스 또는 범위 초과 인덱스.
    """
    invalid = [idx for idx in share_layers if idx < 0 or idx >= total_self_attn]
    if invalid:
        raise ValueError(
            f"share_layers에 유효하지 않은 인덱스가 있습니다: {invalid}. 유효 범위: 0 ~ {total_self_attn - 1} (self-attention layer 총 {total_self_attn}개)."
        )


def remove_shared_attention(pipeline: Any) -> None:
    """apply_shared_attention 이전 processor로 복원. fallback 시 호출.

    WARNING: 검증 단계 — ADR-010. D5 비활성화(fallback) 시 호출하여
    D1 (enforce_consistency) 후처리 baseline으로 돌아간다.

    ``apply_shared_attention`` 에서 저장한 원본 processor dict를 사용해 각 layer를
    정확히 이전 상태로 복원한다. 저장된 dict가 없을 경우(apply 없이 remove 호출)에는
    경고 후 ``AttnProcessor2_0`` fallback을 적용한다.

    Args:
        pipeline: ``apply_shared_attention`` 으로 수정된 pipeline.
            ``unet`` 속성 없으면 경고 후 no-op.
    """
    unet = getattr(pipeline, "unet", None)
    if unet is None:
        logger.warning("remove_shared_attention: pipeline에 'unet' 속성이 없습니다. no-op.")
        return

    try:
        from diffusers.models.attention_processor import Attention, AttnProcessor2_0
    except ImportError:
        logger.error(
            "remove_shared_attention: diffusers를 import할 수 없습니다. processor 복원 실패."
        )
        return

    original: dict[str, Any] | None = getattr(
        pipeline, "_shared_attention_original_processors", None
    )

    if original is None:
        logger.warning(
            "remove_shared_attention: _shared_attention_original_processors 없음. "
            "apply_shared_attention 없이 호출됐거나 이미 복원됨. "
            "AttnProcessor2_0 fallback을 적용합니다."
        )
        restored_count = 0
        for _name, module in unet.named_modules():
            if isinstance(module, Attention) and isinstance(
                module.processor, SharedKVAttnProcessor
            ):
                module.set_processor(AttnProcessor2_0())  # type: ignore[arg-type]
                restored_count += 1
        pipeline._shared_attention_active = False
        logger.info(
            "remove_shared_attention fallback 완료: %d개 layer를 AttnProcessor2_0 으로 복원.",
            restored_count,
        )
        return

    # 원본 processor로 복원
    restored_count = 0
    for name, module in unet.named_modules():
        if name in original and isinstance(module, Attention):
            module.set_processor(original[name])  # type: ignore[arg-type]
            restored_count += 1
            logger.debug("원본 processor 복원: name=%s, processor=%s", name, type(original[name]))

    delattr(pipeline, "_shared_attention_original_processors")
    pipeline._shared_attention_active = False

    logger.info(
        "remove_shared_attention 완료: %d개 layer를 원본 processor로 복원.",
        restored_count,
    )


def validate_batch_for_shared_attention(
    batch: list[Image.Image] | torch.Tensor,
    max_batch_size: int = _DEFAULT_MAX_BATCH_SIZE,
) -> None:
    """Shared attention 호출 직전 batch 검증.

    pipeline 호출 전 반드시 이 함수를 호출하여 batch 크기를 검증한다.
    batch[0]이 reference, batch[1..N]이 source라는 순서는 이 함수에서 강제하지 않으나,
    호출자가 반드시 이 순서를 보장해야 한다.

    Args:
        batch: PIL Image 리스트 또는 ``(B, C, H, W)`` tensor.
            PIL 리스트의 경우 batch[0]=reference, batch[1..]=source.
        max_batch_size: VRAM 상한. RTX 4070 12GB 기준 기본값 4 권장.

    Raises:
        TypeError: batch가 list도 torch.Tensor도 아닐 때.
        ValueError: batch 길이가 2 미만 (reference만 있고 source 없음) 이거나
            ``max_batch_size`` 초과.

    Example:
        >>> images = [reference_img, source1_img, source2_img]
        >>> validate_batch_for_shared_attention(images, max_batch_size=4)
        >>> # 이후 pipeline 호출
    """
    if isinstance(batch, torch.Tensor):
        if batch.ndim != 4:
            raise ValueError(
                f"tensor batch는 (B, C, H, W) 4-dim이어야 합니다. got ndim={batch.ndim}"
            )
        n = batch.shape[0]
    elif isinstance(batch, list):
        n = len(batch)
    else:
        raise TypeError(
            f"batch는 list[Image.Image] 또는 torch.Tensor여야 합니다. got {type(batch).__name__}"
        )

    if n < 2:
        raise ValueError(
            f"batch 크기가 {n}입니다. shared attention에는 최소 2개 필요: batch[0]=reference, batch[1..N]=source. reference만 있고 source가 없으면 K/V 공유 의미가 없습니다."
        )

    if n > max_batch_size:
        raise ValueError(
            f"batch 크기 {n}이 max_batch_size={max_batch_size}를 초과합니다. RTX 4070 12GB 기준 N <= 4 권장 (VRAM 한계). N이 더 필요하면 sequential 처리 또는 enable_model_cpu_offload 고려."
        )

    # PIL list 입력 시 해상도 일치 검증.
    # tensor 입력은 이미 (B, C, H, W)로 모든 원소가 동일 shape이므로 별도 검증 불필요.
    if isinstance(batch, list) and n >= 2:
        ref_size = batch[0].size  # type: ignore[union-attr]
        for i, img in enumerate(batch[1:], start=1):
            img_size = img.size  # type: ignore[union-attr]
            if img_size != ref_size:
                raise ValueError(
                    f"batch[{i}].size {img_size} != reference size {ref_size}. "
                    "shared attention에서는 모든 이미지 해상도가 동일해야 합니다. "
                    "UNet attention seq_len이 달라지면 K/V expand가 런타임 에러를 유발합니다."
                )

    logger.debug("validate_batch_for_shared_attention: n=%d, max=%d — OK", n, max_batch_size)
