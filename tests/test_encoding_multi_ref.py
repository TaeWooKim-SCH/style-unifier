"""multi_ref.py 단위 테스트 (그룹 D4).

StyleEncoder는 mock으로 대체한다.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from PIL import Image

from src.encoding.multi_ref import aggregate_references, compute_clip_similarities

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_img() -> Image.Image:
    arr = np.zeros((64, 64, 3), dtype=np.uint8)
    arr[:, :] = [128, 64, 32]
    return Image.fromarray(arr, mode="RGB")


@pytest.fixture
def another_img() -> Image.Image:
    arr = np.zeros((64, 64, 3), dtype=np.uint8)
    arr[:, :] = [32, 64, 128]
    return Image.fromarray(arr, mode="RGB")


def _make_mock_encoder(seq_len: int = 257, dim: int = 1280) -> MagicMock:
    """encode_single이 (1, seq_len, dim) tensor를 반환하는 mock StyleEncoder."""
    encoder = MagicMock()
    encoder.encode_single.return_value = torch.ones(1, seq_len, dim)
    encoder.encode_multi.return_value = torch.ones(1, seq_len, dim)
    return encoder


# ---------------------------------------------------------------------------
# 기본 오류 케이스
# ---------------------------------------------------------------------------


def test_aggregate_references_empty_raises(sample_img):
    """빈 references 리스트에서 ValueError가 발생하는가."""
    mock_enc = _make_mock_encoder()
    with pytest.raises(ValueError, match="empty"):
        aggregate_references([], strategy="mean", style_encoder=mock_enc)


def test_aggregate_references_invalid_strategy_raises(sample_img):
    """지원하지 않는 strategy에서 ValueError가 발생하는가."""
    mock_enc = _make_mock_encoder()
    with pytest.raises(ValueError, match="strategy"):
        aggregate_references([sample_img], strategy="unknown_strategy", style_encoder=mock_enc)


def test_aggregate_references_manual_without_weights_raises(sample_img, another_img):
    """strategy='manual'인데 weights=None이면 ValueError가 발생하는가."""
    mock_enc = _make_mock_encoder()
    with pytest.raises(ValueError, match="weights"):
        aggregate_references(
            [sample_img, another_img],
            strategy="manual",
            weights=None,
            style_encoder=mock_enc,
        )


def test_aggregate_references_manual_weights_length_mismatch_raises(sample_img, another_img):
    """strategy='manual'에서 weights 길이가 references 길이와 다를 때 ValueError인가."""
    mock_enc = _make_mock_encoder()
    with pytest.raises(ValueError, match="length"):
        aggregate_references(
            [sample_img, another_img],
            strategy="manual",
            weights=[1.0],  # 길이 1, references 길이 2
            style_encoder=mock_enc,
        )


def test_aggregate_references_manual_weights_sum_zero_raises(sample_img, another_img):
    """strategy='manual'에서 weights 합이 0이면 ValueError가 발생하는가."""
    mock_enc = _make_mock_encoder()
    with pytest.raises(ValueError, match="[Ss]um|zero"):
        aggregate_references(
            [sample_img, another_img],
            strategy="manual",
            weights=[0.0, 0.0],
            style_encoder=mock_enc,
        )


# ---------------------------------------------------------------------------
# N=1 경로 — strategy 무관 encode_single 호출
# ---------------------------------------------------------------------------


def test_aggregate_references_single_ref_calls_encode_single(sample_img):
    """N=1이면 strategy와 무관하게 encode_single이 호출되는가."""
    mock_enc = _make_mock_encoder()
    result = aggregate_references([sample_img], strategy="mean", style_encoder=mock_enc)
    mock_enc.encode_single.assert_called_once_with(sample_img)
    assert result.shape == (1, 257, 1280)


def test_aggregate_references_single_ref_mean_vs_first_same_result(sample_img):
    """N=1에서 strategy='mean'과 strategy='first'는 동일한 결과를 내는가."""
    mock_enc = _make_mock_encoder()
    r_mean = aggregate_references([sample_img], strategy="mean", style_encoder=mock_enc)
    r_first = aggregate_references([sample_img], strategy="first", style_encoder=mock_enc)
    assert torch.allclose(r_mean, r_first)


# ---------------------------------------------------------------------------
# strategy별 정상 경로
# ---------------------------------------------------------------------------


def test_aggregate_references_mean_returns_correct_shape(sample_img, another_img):
    """strategy='mean'이 (1, seq_len, dim) shape 텐서를 반환하는가."""
    mock_enc = _make_mock_encoder()
    result = aggregate_references(
        [sample_img, another_img], strategy="mean", style_encoder=mock_enc
    )
    assert result.shape == (1, 257, 1280)


def test_aggregate_references_first_returns_correct_shape(sample_img, another_img):
    """strategy='first'가 (1, seq_len, dim) shape 텐서를 반환하는가."""
    mock_enc = _make_mock_encoder()
    result = aggregate_references(
        [sample_img, another_img], strategy="first", style_encoder=mock_enc
    )
    assert result.shape == (1, 257, 1280)


def test_aggregate_references_manual_normalized_weights(sample_img, another_img):
    """strategy='manual'에서 weights=[1.0, 3.0]은 자동 정규화되어 정상 동작하는가."""
    mock_enc = _make_mock_encoder()
    result = aggregate_references(
        [sample_img, another_img],
        strategy="manual",
        weights=[1.0, 3.0],
        style_encoder=mock_enc,
    )
    assert result.shape == (1, 257, 1280)


def test_aggregate_references_weighted_no_target_falls_back_to_mean(sample_img, another_img):
    """strategy='weighted'에서 target=None이면 mean fallback으로 동작하는가."""
    mock_enc = _make_mock_encoder()
    result = aggregate_references(
        [sample_img, another_img],
        strategy="weighted",
        target=None,
        style_encoder=mock_enc,
    )
    # mean fallback이 호출되어 encode_multi가 실행되거나 결과가 반환됨
    assert result.shape == (1, 257, 1280)


# ---------------------------------------------------------------------------
# compute_clip_similarities
# ---------------------------------------------------------------------------


def test_compute_clip_similarities_empty_raises(sample_img):
    """빈 references에서 ValueError가 발생하는가."""
    mock_enc = _make_mock_encoder()
    with pytest.raises(ValueError, match="empty"):
        compute_clip_similarities([], sample_img, style_encoder=mock_enc)


def test_compute_clip_similarities_returns_list_of_floats(sample_img, another_img):
    """N개 references에 대해 N개 float 리스트를 반환하는가."""
    mock_enc = _make_mock_encoder()
    result = compute_clip_similarities(
        [sample_img, another_img], sample_img, style_encoder=mock_enc
    )
    assert isinstance(result, list)
    assert len(result) == 2
    assert all(isinstance(v, float) for v in result)
