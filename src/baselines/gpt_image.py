"""GPT Image 2.0 (OpenAI API) wrapper.

ADR-008 — GPT Image 2.0은 경쟁자가 아닌 평가 baseline이다.
ablation 표의 ``A_GPT_Image`` 컬럼에서 사용한다.

주의: openai 패키지는 requirements.txt에 포함되어 있지 않다.
실제 API 호출 시에는 ``pip install openai`` 가 필요하다.
lazy import로 처리하여 미설치 환경에서도 import는 정상 동작한다.
"""

from __future__ import annotations

import base64
import io
import os
from pathlib import Path

from PIL import Image

from src.baselines.base import BaselineWrapper, CachedBaselineMixin
from src.utils.logging import get_logger

logger = get_logger(__name__)

_DEFAULT_PROMPT = (
    "Transform the source asset to match the style of the reference image. "
    "Preserve the source shape and pose. Output: clean game asset illustration."
)

# API 호출 기본 상한 — 비용 누적 방지
_DEFAULT_MAX_CALLS = 50


class GPTImageBaseline(BaselineWrapper, CachedBaselineMixin):
    """GPT Image 2.0 API를 통한 스타일 변환.

    호출 비용이 발생하므로 다음 가드가 적용된다.

    - API 키는 ``OPENAI_API_KEY`` 환경변수에서만 로드 (인자 노출 금지).
    - ``cache_dir`` 를 통한 디스크 캐싱 — 같은 source/reference 페어 재호출 비용 0.
    - ``max_calls`` 인자로 인스턴스당 누적 API 호출 횟수 상한.

    Attributes:
        name: ``"gpt_image_2_0"``.
        cache_dir: 결과 캐시 디렉토리. ``None`` 이면 비활성화.
        max_calls: 인스턴스 생애 동안 허용되는 최대 API 호출 횟수.
        prompt_template: source/reference 기반 프롬프트 문자열.

    Example:
        >>> import os
        >>> os.environ["OPENAI_API_KEY"] = "sk-..."
        >>> baseline = GPTImageBaseline(cache_dir=Path("data/cache/gpt"))
        >>> result = baseline.transform(source_img, ref_img)
    """

    name = "gpt_image_2_0"

    def __init__(
        self,
        cache_dir: Path | None = None,
        *,
        max_calls: int = _DEFAULT_MAX_CALLS,
        prompt_template: str = _DEFAULT_PROMPT,
    ) -> None:
        """GPTImageBaseline 초기화.

        Args:
            cache_dir: 결과 캐시 디렉토리. ``None`` 이면 캐시 비활성화
                (매 호출마다 API를 실제로 호출한다).
            max_calls: 누적 API 호출 상한. 초과 시 ``RuntimeError``.
            prompt_template: API에 전달할 프롬프트 문자열.

        Raises:
            EnvironmentError: ``OPENAI_API_KEY`` 환경변수 미설정 시.
        """
        self.cache_dir = cache_dir
        self.max_calls = max_calls
        self.prompt_template = prompt_template
        self._call_count = 0

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise OSError(
                "OPENAI_API_KEY environment variable not set. "
                + "GPTImageBaseline requires OpenAI API access."
            )
        # API 키는 인스턴스 내부에서만 사용. 절대 로깅하지 말 것.
        self._api_key = api_key

        logger.info(
            "GPTImageBaseline initialized (max_calls=%d, cache=%s)",
            self.max_calls,
            self.cache_dir,
        )

    @property
    def call_count(self) -> int:
        """현재까지 누적 API 호출 횟수."""
        return self._call_count

    def transform(
        self,
        source: Image.Image,
        reference: Image.Image,
    ) -> Image.Image:
        """OpenAI API 호출 + 캐싱으로 스타일 변환을 수행한다.

        캐시 hit 시 API를 호출하지 않는다.

        Args:
            source: 변환 대상 PIL Image.
            reference: 스타일 reference PIL Image.

        Returns:
            변환된 RGBA PIL Image.

        Raises:
            RuntimeError: ``max_calls`` 초과 시.
            ImportError: ``openai`` 패키지 미설치 시 (``_call_api`` 내부).
            EnvironmentError: API 키 관련 문제 발생 시.
        """
        cache_key = self._cache_key(source, reference, prompt=self.prompt_template)
        cached = self._check_cache(cache_key)
        if cached is not None:
            return cached

        if self._call_count >= self.max_calls:
            raise RuntimeError(
                f"GPTImageBaseline: max_calls ({self.max_calls}) reached. "
                + "Increase max_calls or enable cache to avoid repeated API calls."
            )

        result = self._call_api(source, reference)
        self._call_count += 1
        logger.info(
            "GPT Image API call %d/%d completed",
            self._call_count,
            self.max_calls,
        )

        self._save_to_cache(cache_key, result)
        return result

    def _call_api(
        self,
        source: Image.Image,
        reference: Image.Image,
    ) -> Image.Image:
        """실제 OpenAI API를 호출하여 변환 결과를 반환한다.

        openai 패키지를 lazy import하므로, 미설치 환경에서는
        이 메서드 호출 시점에 ``ImportError`` 가 발생한다.

        Note:
            OpenAI gpt-image-1 API 시그니처는 변경될 수 있다.
            이 구현은 best-effort이며, Linux에서 실제 호출 시
            공식 문서를 확인하고 필요하면 조정할 것.

            ``pip install openai`` 가 선행 필요하다.

        Args:
            source: 변환 대상 PIL Image.
            reference: 스타일 reference PIL Image.

        Returns:
            변환된 RGBA PIL Image.

        Raises:
            ImportError: openai 패키지 미설치 시.
        """
        try:
            from openai import OpenAI  # pyright: ignore[reportMissingImports]
        except ImportError as exc:
            raise ImportError(
                "openai package is required for GPTImageBaseline. "
                + "Install with: pip install openai"
            ) from exc

        client = OpenAI(api_key=self._api_key)

        source_b64 = _image_to_b64(source)
        _image_to_b64(reference)  # reference 인코딩 검증만 (실 전달은 아래 TODO 참조)

        # ⚠️ TODO(Linux): reference 이미지가 실질적으로 API에 전달되지 않는다.
        # PNG base64의 앞부분은 모든 PNG에서 동일한 헤더이므로 reference 시각
        # 정보가 prompt로 흘러가지 않는다. Linux 실측 전 다음 중 하나 채택 필요:
        #   1) images.edit()에 mask 파라미터로 reference 전달 가능 여부 확인
        #   2) multi-image 입력을 지원하는 chat completions(gpt-4o)로 우회
        #   3) reference를 짧은 자연어 스타일 설명으로 변환해 prompt에 삽입
        # 현재 코드는 sanity skeleton — A_GPT 컬럼은 reference 미반영 상태로 측정됨.
        combined_prompt = self.prompt_template

        logger.warning(
            "GPTImageBaseline._call_api: reference image is NOT effectively transmitted "
            "to the API in current skeleton. See TODO in code. Linux verification required."
        )
        logger.debug("Calling GPT Image API (prompt length=%d)", len(combined_prompt))

        # gpt-image-1 edit API 호출
        # 실제 API 인터페이스가 변경될 수 있으므로 Linux 검증 필요
        response = client.images.edit(
            model="gpt-image-1",
            image=io.BytesIO(base64.b64decode(source_b64)),
            prompt=combined_prompt,
            response_format="b64_json",
            size="1024x1024",
        )

        result_b64: str = response.data[0].b64_json  # type: ignore[union-attr]
        result_bytes = base64.b64decode(result_b64)
        return Image.open(io.BytesIO(result_bytes)).convert("RGBA")


def _image_to_b64(img: Image.Image) -> str:
    """PIL Image를 base64 인코딩된 PNG 문자열로 변환한다.

    Args:
        img: 변환할 PIL Image.

    Returns:
        base64 인코딩된 PNG bytes의 UTF-8 문자열.
    """
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()
