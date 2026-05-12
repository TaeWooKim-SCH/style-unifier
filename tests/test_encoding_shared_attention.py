"""shared_attention.py 단위 테스트 (§5.5.2 그룹 D5).

UNet/Attention 모듈은 mock으로 대체한다.
모델 로딩 없이 validate_batch, _broadcast_kv_from_reference, apply_shared_attention 로직만 검증.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from PIL import Image

from src.encoding.shared_attention import (
    _broadcast_kv_from_reference,
    apply_shared_attention,
    remove_shared_attention,
    validate_batch_for_shared_attention,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_pil_images(n: int) -> list[Image.Image]:
    """n개의 작은 PIL RGBA 이미지를 반환한다."""
    imgs = []
    for i in range(n):
        arr = np.zeros((32, 32, 4), dtype=np.uint8)
        arr[:, :, i % 3] = 255
        arr[:, :, 3] = 255
        imgs.append(Image.fromarray(arr, mode="RGBA"))
    return imgs


# ---------------------------------------------------------------------------
# validate_batch_for_shared_attention — PIL list
# ---------------------------------------------------------------------------


def test_validate_batch_single_image_raises():
    """B=1(reference만)이면 ValueError가 발생하는가."""
    imgs = _make_pil_images(1)
    with pytest.raises(ValueError, match="2"):
        validate_batch_for_shared_attention(imgs)


def test_validate_batch_exceeds_max_batch_size_raises():
    """B > max_batch_size이면 ValueError가 발생하는가."""
    imgs = _make_pil_images(5)
    with pytest.raises(ValueError, match="max_batch_size"):
        validate_batch_for_shared_attention(imgs, max_batch_size=4)


def test_validate_batch_valid_two_images_no_raise():
    """B=2이면 예외 없이 통과하는가."""
    imgs = _make_pil_images(2)
    validate_batch_for_shared_attention(imgs, max_batch_size=4)  # 예외 없으면 통과


def test_validate_batch_valid_max_batch_no_raise():
    """B == max_batch_size이면 예외 없이 통과하는가."""
    imgs = _make_pil_images(4)
    validate_batch_for_shared_attention(imgs, max_batch_size=4)


def test_validate_batch_invalid_type_raises():
    """batch가 list도 tensor도 아니면 TypeError가 발생하는가."""
    with pytest.raises(TypeError):
        validate_batch_for_shared_attention("not_a_batch")  # type: ignore[arg-type]


def test_validate_batch_tensor_valid():
    """(B=2, C=3, H=32, W=32) tensor 입력이 통과하는가."""
    t = torch.zeros(2, 3, 32, 32)
    validate_batch_for_shared_attention(t, max_batch_size=4)


def test_validate_batch_tensor_1d_raises():
    """ndim != 4인 tensor가 ValueError를 발생시키는가."""
    t = torch.zeros(2)
    with pytest.raises(ValueError, match="4-dim"):
        validate_batch_for_shared_attention(t)


# ---------------------------------------------------------------------------
# _broadcast_kv_from_reference — 핵심 K/V 공유 로직
# ---------------------------------------------------------------------------


def test_broadcast_kv_from_reference_output_shape():
    """B=3일 때 output shape가 (3, seq, dim)인가."""
    key = torch.randn(3, 16, 64)
    value = torch.randn(3, 16, 64)
    k_out, v_out = _broadcast_kv_from_reference(key, value, batch_size=3)
    assert k_out.shape == (3, 16, 64)
    assert v_out.shape == (3, 16, 64)


def test_broadcast_kv_from_reference_batch0_unchanged():
    """batch[0]의 K/V가 변경되지 않는가."""
    key = torch.randn(3, 16, 64)
    value = torch.randn(3, 16, 64)
    k_out, v_out = _broadcast_kv_from_reference(key, value, batch_size=3)
    assert torch.allclose(k_out[0], key[0])
    assert torch.allclose(v_out[0], value[0])


def test_broadcast_kv_from_reference_batch1_matches_batch0():
    """batch[1]의 K/V가 batch[0]의 K/V와 동일한가."""
    key = torch.randn(3, 16, 64)
    value = torch.randn(3, 16, 64)
    k_out, v_out = _broadcast_kv_from_reference(key, value, batch_size=3)
    assert torch.allclose(k_out[1], key[0])
    assert torch.allclose(v_out[1], value[0])


def test_broadcast_kv_from_reference_batch2_matches_batch0():
    """batch[2]의 K/V가 batch[0]의 K/V와 동일한가 (B=4 케이스)."""
    key = torch.randn(4, 16, 64)
    value = torch.randn(4, 16, 64)
    k_out, v_out = _broadcast_kv_from_reference(key, value, batch_size=4)
    assert torch.allclose(k_out[2], key[0])
    assert torch.allclose(v_out[2], value[0])


# ---------------------------------------------------------------------------
# apply_shared_attention — pipeline mock 검증
# ---------------------------------------------------------------------------


def _make_mock_pipeline_with_unet(n_self_attn: int = 3) -> MagicMock:
    """named_modules에 n_self_attn개의 self-attention 모듈을 가진 mock pipeline."""
    try:
        from diffusers.models.attention_processor import Attention as _Attention

        attention_cls = _Attention
    except ImportError:
        attention_cls = MagicMock

    pipeline = MagicMock()
    unet = MagicMock()
    pipeline.unet = unet

    modules = []
    for i in range(n_self_attn):
        attn = MagicMock(spec=attention_cls)
        attn.is_cross_attention = False
        modules.append((f"attn_{i}", attn))

    unet.named_modules.return_value = modules
    return pipeline


def test_apply_shared_attention_no_unet_raises():
    """pipeline에 unet 속성이 없으면 AttributeError가 발생하는가."""
    pipeline = MagicMock(spec=[])  # unet 없는 mock
    with pytest.raises(AttributeError):
        apply_shared_attention(pipeline)


def test_apply_shared_attention_sets_metadata():
    """apply_shared_attention 호출 후 pipeline 메타데이터가 설정되는가."""
    try:
        from diffusers.models.attention_processor import Attention
    except ImportError:
        pytest.skip("diffusers not installed")

    # named_modules에 실제 Attention 인스턴스 mock
    pipeline = MagicMock()
    unet = MagicMock()
    pipeline.unet = unet

    attn_module = MagicMock(spec=Attention)
    attn_module.is_cross_attention = False
    # apply_shared_attention이 module.processor를 저장하므로 processor 속성 필요
    attn_module.processor = MagicMock(name="original_processor")
    unet.named_modules.return_value = [("attn_0", attn_module)]

    apply_shared_attention(pipeline, max_batch_size=4)

    assert pipeline._shared_attention_active is True
    assert pipeline._shared_attention_max_batch == 4


# ---------------------------------------------------------------------------
# remove_shared_attention — 이슈 I1: 원본 processor 복원 검증
# ---------------------------------------------------------------------------


def _make_pipeline_with_original_processors(n_self_attn: int = 2):
    """apply + remove 사이클을 테스트하기 위한 mock pipeline 구성."""
    try:
        from diffusers.models.attention_processor import Attention as _Attention

        attention_cls = _Attention
    except ImportError:
        attention_cls = MagicMock

    pipeline = MagicMock()
    unet = MagicMock()
    pipeline.unet = unet

    # 각 모듈에 고유한 mock processor를 사전 설정
    sentinel_processors = [MagicMock(name=f"original_proc_{i}") for i in range(n_self_attn)]
    modules = []
    for i in range(n_self_attn):
        attn = MagicMock(spec=attention_cls)
        attn.is_cross_attention = False
        attn.processor = sentinel_processors[i]
        modules.append((f"attn_{i}", attn))

    unet.named_modules.return_value = modules
    return pipeline, modules, sentinel_processors


def test_remove_shared_attention_restores_original_processor():
    """apply 후 remove 하면 원본 processor가 복원되는가."""
    try:
        import diffusers  # noqa: F401
    except ImportError:
        pytest.skip("diffusers not installed")

    pipeline, modules, sentinel_processors = _make_pipeline_with_original_processors(n_self_attn=2)

    apply_shared_attention(pipeline)
    # apply 이후 _shared_attention_original_processors 가 설정됐는지 확인
    assert hasattr(pipeline, "_shared_attention_original_processors")

    remove_shared_attention(pipeline)

    # 각 모듈의 set_processor가 원본 sentinel processor로 호출됐는지 검증
    for i, (_name, attn_module) in enumerate(modules):
        calls = attn_module.set_processor.call_args_list
        # apply 로 SharedKVAttnProcessor, remove 로 sentinel_processor 순서
        assert len(calls) >= 2, f"attn_{i}: set_processor가 최소 2회 호출되어야 함"
        last_call_arg = calls[-1][0][0]
        assert (
            last_call_arg is sentinel_processors[i]
        ), f"attn_{i}: 마지막 set_processor 인자가 원본 processor여야 함, got {last_call_arg}"


def test_remove_shared_attention_clears_metadata():
    """remove 후 _shared_attention_active가 False이고 original_processors 속성이 사라지는가."""
    try:
        import diffusers  # noqa: F401
    except ImportError:
        pytest.skip("diffusers not installed")

    pipeline, _modules, _procs = _make_pipeline_with_original_processors(n_self_attn=1)

    apply_shared_attention(pipeline)
    remove_shared_attention(pipeline)

    assert pipeline._shared_attention_active is False
    assert not hasattr(pipeline, "_shared_attention_original_processors")


def test_remove_shared_attention_without_apply_uses_fallback():
    """apply 없이 remove를 호출하면 AttnProcessor2_0 fallback을 적용하고 에러 없이 반환하는가."""
    try:
        from diffusers.models.attention_processor import Attention
    except ImportError:
        pytest.skip("diffusers not installed")

    from src.encoding.shared_attention import SharedKVAttnProcessor

    pipeline = MagicMock()
    unet = MagicMock()
    pipeline.unet = unet

    # _shared_attention_original_processors 없는 상태 시뮬레이션
    del pipeline._shared_attention_original_processors

    attn_module = MagicMock(spec=Attention)
    attn_module.is_cross_attention = False
    attn_module.processor = SharedKVAttnProcessor()
    unet.named_modules.return_value = [("attn_0", attn_module)]

    # 에러 없이 반환해야 한다
    remove_shared_attention(pipeline)
    assert pipeline._shared_attention_active is False


# ---------------------------------------------------------------------------
# validate_batch_for_shared_attention — 이슈 I2: 해상도 불일치 검증
# ---------------------------------------------------------------------------


def test_validate_batch_resolution_mismatch_raises():
    """PIL list에서 reference(batch[0])와 크기가 다른 이미지가 있으면 ValueError가 발생하는가."""
    ref_img = Image.new("RGBA", (64, 64))
    source_img = Image.new("RGBA", (32, 32))  # 해상도 불일치
    batch = [ref_img, source_img]
    with pytest.raises(ValueError, match="size"):
        validate_batch_for_shared_attention(batch, max_batch_size=4)


def test_validate_batch_resolution_mismatch_second_source_raises():
    """batch[2]의 크기가 다를 때도 ValueError가 발생하는가."""
    ref_img = Image.new("RGBA", (64, 64))
    source1 = Image.new("RGBA", (64, 64))  # OK
    source2 = Image.new("RGBA", (32, 48))  # 불일치
    batch = [ref_img, source1, source2]
    with pytest.raises(ValueError, match="batch\\[2\\]"):
        validate_batch_for_shared_attention(batch, max_batch_size=4)


def test_validate_batch_resolution_same_no_raise():
    """모든 이미지 해상도가 동일하면 예외가 발생하지 않는가."""
    imgs = [Image.new("RGBA", (64, 64)) for _ in range(3)]
    validate_batch_for_shared_attention(imgs, max_batch_size=4)  # 예외 없으면 통과
