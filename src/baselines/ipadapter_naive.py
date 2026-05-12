"""IP-Adapter only baseline (ControlNet·후처리 없음).

본 시스템(StyleUnificationPipeline)과의 ablation 비교용이다.
그룹 F1 ablation의 A1 (IP-Adapter only) 컬럼에서 사용한다.
ControlNet 없이 형태 보존이 약한 것이 의도된 약점이다.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from PIL import Image

from src.baselines.base import BaselineWrapper
from src.utils.device import get_device, get_dtype
from src.utils.logging import get_logger

logger = get_logger(__name__)

_DEFAULT_BASE_CHECKPOINT = "cagliostrolab/animagine-xl-3.1"
_DEFAULT_IP_ADAPTER_REPO = "h94/IP-Adapter"
_DEFAULT_IP_ADAPTER_SUBFOLDER = "sdxl_models"
_DEFAULT_IP_ADAPTER_WEIGHT = "ip-adapter-plus_sdxl_vit-h.safetensors"
_DEFAULT_IP_ADAPTER_SCALE = 0.7

_DEFAULT_PROMPT = "high quality game asset"
_DEFAULT_NUM_STEPS = 25
_DEFAULT_GUIDANCE_SCALE = 6.0
_DEFAULT_STRENGTH = 0.55


class IPAdapterNaiveBaseline(BaselineWrapper):
    """SDXL + IP-Adapter only. ControlNet·전처리·후처리 모두 비활성화.

    ``StyleUnificationPipeline`` 대비 형태 보존이 의도적으로 약하다.
    ablation 실험에서 ControlNet·후처리가 없을 때 얼마나 형태가 무너지는지
    정량적으로 보여주는 것이 이 클래스의 목적이다.

    Attributes:
        name: ``"ipadapter_naive"``.
        cache_dir: 미사용. 캐시 없이 매번 실시간 추론.

    Example:
        >>> baseline = IPAdapterNaiveBaseline()
        >>> result = baseline.transform(source_img, ref_img)  # Linux에서만 실행
    """

    name = "ipadapter_naive"
    cache_dir: Path | None = None

    def __init__(
        self,
        base_checkpoint: str = _DEFAULT_BASE_CHECKPOINT,
        ip_adapter_repo: str = _DEFAULT_IP_ADAPTER_REPO,
        ip_adapter_subfolder: str = _DEFAULT_IP_ADAPTER_SUBFOLDER,
        ip_adapter_weight: str = _DEFAULT_IP_ADAPTER_WEIGHT,
        ip_adapter_scale: float = _DEFAULT_IP_ADAPTER_SCALE,
        device: torch.device | None = None,
    ) -> None:
        """IPAdapterNaiveBaseline 초기화. 실제 모델 로딩은 lazy로 수행된다.

        Args:
            base_checkpoint: HuggingFace Hub의 SDXL 기반 모델 ID.
            ip_adapter_repo: IP-Adapter 가중치가 있는 HuggingFace 레포.
            ip_adapter_subfolder: 레포 내 가중치 서브폴더.
            ip_adapter_weight: IP-Adapter 가중치 파일명.
            ip_adapter_scale: IP-Adapter 스타일 강도 (0.0-1.0).
            device: 추론 device. ``None`` 이면 ``get_device()`` 자동 선택.
        """
        if not 0.0 <= ip_adapter_scale <= 1.0:
            raise ValueError(f"ip_adapter_scale must be in [0.0, 1.0], got {ip_adapter_scale}")

        self.base_checkpoint = base_checkpoint
        self.ip_adapter_repo = ip_adapter_repo
        self.ip_adapter_subfolder = ip_adapter_subfolder
        self.ip_adapter_weight = ip_adapter_weight
        self.ip_adapter_scale = ip_adapter_scale

        self.device: torch.device = device if device is not None else get_device()
        self.dtype: torch.dtype = get_dtype(self.device)

        self.pipe: Any = None
        self._loaded = False

        logger.info(
            "IPAdapterNaiveBaseline created (device=%s, dtype=%s, lazy)",
            self.device,
            self.dtype,
        )

    def _load_pipeline(self) -> None:
        """SDXL Img2Img (ControlNet 없음) + IP-Adapter를 lazy 로딩한다.

        이미 로딩된 경우 즉시 반환한다.

        Raises:
            ImportError: diffusers 패키지 미설치 시.
        """
        if self._loaded:
            return

        try:
            from diffusers import (  # pyright: ignore[reportMissingImports]
                StableDiffusionXLImg2ImgPipeline,
            )
        except ImportError as exc:
            raise ImportError("diffusers package is required for IPAdapterNaiveBaseline.") from exc

        logger.info("Loading SDXL img2img: %s", self.base_checkpoint)
        self.pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
            self.base_checkpoint,
            torch_dtype=self.dtype,
        )
        self.pipe.enable_model_cpu_offload()
        self.pipe.enable_vae_slicing()

        try:
            self.pipe.enable_xformers_memory_efficient_attention()
        except ModuleNotFoundError:
            logger.warning("xformers not installed, using default attention")

        logger.info(
            "Loading IP-Adapter: %s / %s / %s",
            self.ip_adapter_repo,
            self.ip_adapter_subfolder,
            self.ip_adapter_weight,
        )
        self.pipe.load_ip_adapter(
            self.ip_adapter_repo,
            subfolder=self.ip_adapter_subfolder,
            weight_name=self.ip_adapter_weight,
        )
        self.pipe.set_ip_adapter_scale(self.ip_adapter_scale)

        self._loaded = True
        logger.info("IPAdapterNaiveBaseline pipeline loaded")

    def transform(
        self,
        source: Image.Image,
        reference: Image.Image,
    ) -> Image.Image:
        """단순 img2img + IP-Adapter로 스타일 변환을 수행한다.

        ControlNet이 없으므로 형태 보존이 약하다 (의도된 약점).

        Args:
            source: 변환 대상 PIL Image.
            reference: 스타일 reference PIL Image.

        Returns:
            변환된 RGBA PIL Image.

        Raises:
            RuntimeError: 파이프라인 로딩 실패 시.
            ImportError: diffusers 패키지 미설치 시.
        """
        self._load_pipeline()
        if self.pipe is None:
            raise RuntimeError("IPAdapterNaiveBaseline: pipeline failed to load.")

        with torch.inference_mode():
            output = self.pipe(
                prompt=_DEFAULT_PROMPT,
                image=source.convert("RGB"),
                ip_adapter_image=reference,
                num_inference_steps=_DEFAULT_NUM_STEPS,
                guidance_scale=_DEFAULT_GUIDANCE_SCALE,
                strength=_DEFAULT_STRENGTH,
            )

        result: Image.Image = output.images[0].convert("RGBA")
        return result
