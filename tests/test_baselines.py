"""baselines 모듈 단위 테스트.

base.py (BaselineWrapper ABC, CachedBaselineMixin),
gpt_image.py (GPTImageBaseline), ipadapter_naive.py (IPAdapterNaiveBaseline).

모든 API/모델 호출은 mock 또는 환경변수 monkeypatch로 우회한다.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.baselines.base import BaselineWrapper, CachedBaselineMixin

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tiny_source() -> Image.Image:
    """32x32 RGBA source 이미지."""
    arr = np.zeros((32, 32, 4), dtype=np.uint8)
    arr[:, :] = [255, 0, 0, 255]
    return Image.fromarray(arr, mode="RGBA")


@pytest.fixture
def tiny_reference() -> Image.Image:
    """32x32 RGBA reference 이미지."""
    arr = np.zeros((32, 32, 4), dtype=np.uint8)
    arr[:, :] = [0, 0, 255, 255]
    return Image.fromarray(arr, mode="RGBA")


# ---------------------------------------------------------------------------
# BaselineWrapper — ABC 강제
# ---------------------------------------------------------------------------


def test_baseline_wrapper_cannot_instantiate_directly():
    """BaselineWrapper를 직접 인스턴스화하면 TypeError가 발생하는가."""
    with pytest.raises(TypeError):
        BaselineWrapper()  # type: ignore[abstract]


def test_baseline_wrapper_concrete_subclass_works():
    """transform을 구현한 구체 클래스는 인스턴스화 가능한가."""

    class ConcreteBaseline(BaselineWrapper):
        name = "concrete"

        def transform(self, source: Image.Image, reference: Image.Image) -> Image.Image:
            return source.convert("RGBA")

    baseline = ConcreteBaseline()
    assert baseline.name == "concrete"


def test_baseline_wrapper_transform_batch_calls_transform_n_times(tiny_source, tiny_reference):
    """transform_batch 기본 구현이 transform을 N번 호출하는가."""

    class ConcreteBaseline(BaselineWrapper):
        name = "concrete"
        call_count = 0

        def transform(self, source: Image.Image, reference: Image.Image) -> Image.Image:
            self.call_count += 1
            return source.convert("RGBA")

    baseline = ConcreteBaseline()
    sources = [tiny_source, tiny_source, tiny_source]
    results = baseline.transform_batch(sources, tiny_reference)

    assert baseline.call_count == 3
    assert len(results) == 3


# ---------------------------------------------------------------------------
# CachedBaselineMixin — 캐시 키 결정론성 + hit/miss
# ---------------------------------------------------------------------------


class _ConcreteWithCache(BaselineWrapper, CachedBaselineMixin):
    name = "test_cached"

    def __init__(self, cache_dir: Path | None = None) -> None:
        self.cache_dir = cache_dir

    def transform(self, source: Image.Image, reference: Image.Image) -> Image.Image:
        return source.convert("RGBA")


def test_cache_key_deterministic(tiny_source, tiny_reference):
    """같은 입력에서 _cache_key가 동일한 키를 반환하는가."""
    wrapper = _ConcreteWithCache()
    key1 = wrapper._cache_key(tiny_source, tiny_reference, "prompt")
    key2 = wrapper._cache_key(tiny_source, tiny_reference, "prompt")
    assert key1 == key2


def test_cache_key_different_prompt_differs(tiny_source, tiny_reference):
    """프롬프트가 다르면 캐시 키가 달라지는가."""
    wrapper = _ConcreteWithCache()
    key1 = wrapper._cache_key(tiny_source, tiny_reference, "prompt_a")
    key2 = wrapper._cache_key(tiny_source, tiny_reference, "prompt_b")
    assert key1 != key2


def test_cache_key_is_64_char_hex(tiny_source, tiny_reference):
    """캐시 키가 64자 hex string인가 (SHA-256)."""
    wrapper = _ConcreteWithCache()
    key = wrapper._cache_key(tiny_source, tiny_reference)
    assert len(key) == 64
    assert all(c in "0123456789abcdef" for c in key)


def test_check_cache_no_cache_dir_returns_none(tiny_source, tiny_reference):
    """cache_dir=None이면 _check_cache가 None을 반환하는가."""
    wrapper = _ConcreteWithCache(cache_dir=None)
    key = wrapper._cache_key(tiny_source, tiny_reference)
    result = wrapper._check_cache(key)
    assert result is None


def test_check_cache_miss_returns_none(tmp_path, tiny_source, tiny_reference):
    """cache_dir 설정 후 존재하지 않는 키로 _check_cache가 None을 반환하는가."""
    wrapper = _ConcreteWithCache(cache_dir=tmp_path)
    result = wrapper._check_cache("nonexistent_key_12345")
    assert result is None


def test_save_and_check_cache_round_trip(tmp_path, tiny_source, tiny_reference):
    """_save_to_cache 후 같은 키로 _check_cache가 이미지를 반환하는가."""
    wrapper = _ConcreteWithCache(cache_dir=tmp_path)
    key = wrapper._cache_key(tiny_source, tiny_reference)
    wrapper._save_to_cache(key, tiny_source)
    result = wrapper._check_cache(key)
    assert result is not None
    assert result.mode == "RGBA"


def test_save_cache_creates_png_file(tmp_path, tiny_source, tiny_reference):
    """_save_to_cache가 실제 PNG 파일을 생성하는가."""
    wrapper = _ConcreteWithCache(cache_dir=tmp_path)
    key = wrapper._cache_key(tiny_source, tiny_reference)
    wrapper._save_to_cache(key, tiny_source)
    assert (tmp_path / f"{key}.png").exists()


# ---------------------------------------------------------------------------
# GPTImageBaseline — API 키 가드
# ---------------------------------------------------------------------------


def test_gpt_image_baseline_without_api_key_raises(monkeypatch, tmp_path):
    """OPENAI_API_KEY 미설정 시 OSError가 발생하는가."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    from src.baselines.gpt_image import GPTImageBaseline

    with pytest.raises(OSError, match="OPENAI_API_KEY"):
        GPTImageBaseline(cache_dir=tmp_path)


def test_gpt_image_baseline_with_api_key_succeeds(monkeypatch, tmp_path):
    """OPENAI_API_KEY 설정 시 GPTImageBaseline이 생성되는가."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-fake-key")

    from src.baselines.gpt_image import GPTImageBaseline

    baseline = GPTImageBaseline(cache_dir=tmp_path, max_calls=10)
    assert baseline.name == "gpt_image_2_0"
    assert baseline.max_calls == 10
    assert baseline.call_count == 0


def test_gpt_image_baseline_max_calls_guard(monkeypatch, tmp_path, tiny_source, tiny_reference):
    """max_calls 도달 후 transform 호출 시 RuntimeError가 발생하는가."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-fake-key")

    from src.baselines.gpt_image import GPTImageBaseline

    baseline = GPTImageBaseline(cache_dir=tmp_path, max_calls=0)
    with pytest.raises(RuntimeError, match="max_calls"):
        baseline.transform(tiny_source, tiny_reference)


# ---------------------------------------------------------------------------
# IPAdapterNaiveBaseline — lazy 로딩 검증
# ---------------------------------------------------------------------------


def test_ipadapter_naive_lazy_not_loaded():
    """IPAdapterNaiveBaseline 생성 직후 _loaded가 False인가."""
    from src.baselines.ipadapter_naive import IPAdapterNaiveBaseline

    baseline = IPAdapterNaiveBaseline()
    assert baseline._loaded is False


def test_ipadapter_naive_pipe_is_none_initially():
    """생성 직후 pipe 속성이 None인가."""
    from src.baselines.ipadapter_naive import IPAdapterNaiveBaseline

    baseline = IPAdapterNaiveBaseline()
    assert baseline.pipe is None


def test_ipadapter_naive_invalid_scale_raises():
    """ip_adapter_scale > 1.0이면 ValueError가 발생하는가."""
    from src.baselines.ipadapter_naive import IPAdapterNaiveBaseline

    with pytest.raises(ValueError, match="ip_adapter_scale"):
        IPAdapterNaiveBaseline(ip_adapter_scale=1.5)


def test_ipadapter_naive_name():
    """name 속성이 'ipadapter_naive'인가."""
    from src.baselines.ipadapter_naive import IPAdapterNaiveBaseline

    baseline = IPAdapterNaiveBaseline()
    assert baseline.name == "ipadapter_naive"
