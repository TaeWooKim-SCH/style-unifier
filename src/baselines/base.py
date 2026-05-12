"""모든 baseline 구현체의 공통 인터페이스 정의.

evaluator의 ablation loop가 GPT Image / IP-Adapter only / StyleUnificationPipeline 등
구체 클래스에 의존하지 않고 다형적으로 호출하도록 한다 (Dependency Inversion).
"""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from pathlib import Path

from PIL import Image

from src.utils.logging import get_logger

logger = get_logger(__name__)


class BaselineWrapper(ABC):
    """모든 baseline 구현체의 공통 인터페이스.

    Attributes:
        name: 결과·로그·실험 폴더에서 식별자로 사용. 구체 클래스 정의 시 명시.
            예: ``"gpt_image_2_0"``, ``"ipadapter_naive"``.
        cache_dir: 결과 캐시 디렉토리. ``None`` 이면 캐시 비활성화.
    """

    name: str = "baseline"
    cache_dir: Path | None = None

    @abstractmethod
    def transform(
        self,
        source: Image.Image,
        reference: Image.Image,
    ) -> Image.Image:
        """단일 source + 단일 reference -> 단일 RGBA 출력.

        구체 클래스가 반드시 구현. RGBA 출력 보장은 호출 측 책임이 아니라
        본 메서드의 책임.

        Args:
            source: 변환 대상 PIL Image.
            reference: 스타일 reference PIL Image.

        Returns:
            변환된 PIL Image. RGBA 권장 (어렵다면 RGB도 허용하되
            호출자가 RGBA 변환 필요).
        """

    def transform_batch(
        self,
        sources: list[Image.Image],
        reference: Image.Image,
    ) -> list[Image.Image]:
        """기본 구현: ``transform()`` 을 N번 순차 호출.

        배치 일관성 메커니즘이 있는 구현체(예: StyleUnificationPipeline의
        shared attention)는 본 메서드를 override한다.

        Args:
            sources: N개 PIL Image.
            reference: 단일 reference PIL Image.

        Returns:
            N개 PIL Image. 각 원소는 ``transform()`` 반환과 동일한 형식.
        """
        logger.debug(
            "%s.transform_batch: processing %d images sequentially",
            self.name,
            len(sources),
        )
        return [self.transform(src, reference) for src in sources]


class CachedBaselineMixin:
    """``cache_dir`` 기반 결과 캐싱 mixin.

    캐시 키는 ``name + source.tobytes() + reference.tobytes() + prompt`` 의
    SHA-256 hex digest로 구성된다. Hit 시 디스크에서 PNG를 로드하여 반환한다.

    Subclass가 ``transform()`` 호출 직전에 ``_check_cache()`` 와
    ``_save_to_cache()`` 를 사용할 수 있다.

    Note:
        이 mixin은 ``BaselineWrapper`` 와 함께 다중 상속으로 사용한다.
        예: ``class GPTImageBaseline(BaselineWrapper, CachedBaselineMixin)``
    """

    name: str
    cache_dir: Path | None

    def _cache_key(
        self,
        source: Image.Image,
        reference: Image.Image,
        prompt: str = "",
    ) -> str:
        """결과 캐시 경로용 SHA-256 hex digest를 반환한다.

        Args:
            source: 변환 대상 PIL Image.
            reference: 스타일 reference PIL Image.
            prompt: 프롬프트 문자열 (없으면 빈 문자열).

        Returns:
            64자 hex string.
        """
        h = hashlib.sha256()
        h.update(self.name.encode())
        h.update(source.tobytes())
        h.update(reference.tobytes())
        h.update(prompt.encode())
        return h.hexdigest()

    def _check_cache(self, key: str) -> Image.Image | None:
        """캐시에서 결과 이미지를 조회한다.

        Args:
            key: ``_cache_key()`` 반환값.

        Returns:
            캐시 hit 시 RGBA PIL Image, miss 시 ``None``.
        """
        if self.cache_dir is None:
            return None
        path = self.cache_dir / f"{key}.png"
        if path.exists():
            logger.info("Cache hit: %s", path.name)
            return Image.open(path).convert("RGBA")
        return None

    def _save_to_cache(self, key: str, image: Image.Image) -> None:
        """결과 이미지를 캐시 디렉토리에 PNG로 저장한다.

        Args:
            key: ``_cache_key()`` 반환값.
            image: 저장할 PIL Image.
        """
        if self.cache_dir is None:
            return
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        path = self.cache_dir / f"{key}.png"
        image.save(path, format="PNG")
        logger.debug("Saved to cache: %s", path.name)
