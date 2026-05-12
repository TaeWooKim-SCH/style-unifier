"""src.baselines 패키지 공개 API.

평가용 baseline wrapper 클래스들을 노출한다.
- ``BaselineWrapper``: 모든 구현체의 ABC.
- ``CachedBaselineMixin``: 디스크 캐싱 mixin.
- ``GPTImageBaseline``: GPT Image 2.0 API wrapper (ADR-008).
- ``IPAdapterNaiveBaseline``: IP-Adapter only ablation baseline.
"""

from src.baselines.base import BaselineWrapper, CachedBaselineMixin
from src.baselines.gpt_image import GPTImageBaseline
from src.baselines.ipadapter_naive import IPAdapterNaiveBaseline

__all__ = [
    "BaselineWrapper",
    "CachedBaselineMixin",
    "GPTImageBaseline",
    "IPAdapterNaiveBaseline",
]
