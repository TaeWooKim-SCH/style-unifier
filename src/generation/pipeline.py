"""메인 스타일 통일 파이프라인 모듈.

SDXL + ControlNet (lineart) + IP-Adapter Plus 를 조합하여
단일 source 에셋을 reference 이미지 스타일로 변환한다.
전처리(bg_removal, lineart 추출)와 후처리(alpha 복원, palette quantize)가
파이프라인 내부에 포함되어 있어, 호출자는 PIL Image만 전달하면 된다.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt
import torch
import yaml
from PIL import Image

from src.postprocessing.alpha_restore import restore_alpha
from src.postprocessing.palette_quantize import quantize_to_palette
from src.preprocessing.bg_removal import remove_background
from src.preprocessing.lineart import extract_lineart
from src.preprocessing.palette import extract_palette
from src.utils.device import get_device, get_dtype
from src.utils.logging import get_logger
from src.utils.repro import make_generator

if TYPE_CHECKING:
    from diffusers.pipelines.controlnet.pipeline_controlnet_sd_xl_img2img import (
        StableDiffusionXLControlNetImg2ImgPipeline,
    )

logger = get_logger(__name__)

# 기본 프롬프트 (CLAUDE.md §5.3.1)
_DEFAULT_PROMPT = "high quality game asset, clean vector illustration"
_DEFAULT_NEGATIVE_PROMPT = "blurry, realistic, photo, 3d render"

# 필수 config 키 — load_config() 에서 검증
_REQUIRED_CONFIG_KEYS = ("model", "sampling", "preprocessing", "postprocessing")


def load_config(config_path: Path) -> dict:
    """YAML 파일을 로드하고 필수 키를 검증해 반환한다.

    Args:
        config_path: YAML 설정 파일 경로.

    Returns:
        설정 딕셔너리.

    Raises:
        FileNotFoundError: config_path가 존재하지 않을 때.
        KeyError: 필수 키(model, sampling, preprocessing, postprocessing)가
            없을 때.
        yaml.YAMLError: YAML 파싱 실패 시.

    Example:
        >>> cfg = load_config(Path("configs/default.yaml"))
        >>> pipeline = StyleUnificationPipeline(cfg)
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as fh:
        cfg: dict = yaml.safe_load(fh)

    for key in _REQUIRED_CONFIG_KEYS:
        if key not in cfg:
            raise KeyError(
                f"Required config key '{key}' is missing in {config_path}. Required keys: {_REQUIRED_CONFIG_KEYS}"
            )

    logger.info("Config loaded from %s", config_path)
    return cfg


class StyleUnificationPipeline:
    """단일 reference + 단일 source → 단일 변환 결과 RGBA 출력.

    내부적으로 SDXL + ControlNet (lineart) + IP-Adapter (Plus) 를 조합한다.
    전처리(bg_removal, lineart 추출)와 후처리(alpha 복원, palette quantize)도
    파이프라인에 포함되어 호출자는 source/reference PIL 이미지만 전달하면 된다.

    모델 로딩은 lazy 방식이므로 인스턴스 생성 자체는 비용이 없다.
    첫 ``transform()`` 호출 시 실제 모델이 로드된다.

    Args:
        config: ``configs/default.yaml`` 구조의 dict. ``load_config()`` 로 로드.
        device: 추론 device. None이면 ``get_device()``.

    Attributes:
        config: 보관된 설정.
        device: 실행 device.
        dtype: device에 맞는 dtype (cuda/mps → fp16, cpu → fp32).
        pipe: 내부 diffusers pipeline. 첫 transform() 호출 전까지 None.

    Note:
        ``StyleEncoder`` 는 그룹 D(multi-reference / shared attention)에서 명시적으로
        연결될 예정이다. 현재 단일 reference 흐름은 diffusers 내부 IP-Adapter 인코더에
        직접 위임한다 (``ip_adapter_image=reference``).

    Example:
        >>> cfg = load_config(Path("configs/default.yaml"))
        >>> pipeline = StyleUnificationPipeline(cfg)
        >>> result = pipeline.transform(source_img, reference_img, seed=42)
    """

    def __init__(
        self,
        config: dict,
        device: torch.device | None = None,
    ) -> None:
        """파이프라인 초기화. 모델 로딩은 첫 transform() 호출 시 지연된다.

        Args:
            config: ``configs/default.yaml`` 구조의 dict.
            device: 추론 device. None이면 ``get_device()`` 로 자동 선택.
        """
        self.config = config
        self.device: torch.device = device if device is not None else get_device()
        self.dtype: torch.dtype = get_dtype(self.device)

        # lazy-load 대상 — 첫 transform() 호출 전까지 None
        self.pipe: StableDiffusionXLControlNetImg2ImgPipeline | None = None

        self._loaded: bool = False

        logger.info(
            "StyleUnificationPipeline initialized (device=%s, dtype=%s). Models will be loaded on first transform() call.",
            self.device,
            self.dtype,
        )

    # ------------------------------------------------------------------
    # Private helpers — pipeline loading
    # ------------------------------------------------------------------

    def _require_pipe_loaded(self) -> StableDiffusionXLControlNetImg2ImgPipeline:
        """``self.pipe`` 가 로드되어 있음을 보장하고 반환한다.

        Raises:
            RuntimeError: ``_load_pipeline()`` 이 선행되지 않았을 때.
        """
        if self.pipe is None:
            raise RuntimeError(
                "StyleUnificationPipeline.pipe is not initialized. "
                "Call _load_pipeline() (or transform()) first."
            )
        return self.pipe

    def _load_pipeline(self) -> None:
        """파이프라인 전체(ControlNet, SDXL, IP-Adapter, Scheduler)를 로드한다.

        이미 로드되어 있으면 아무 작업도 하지 않는다 (멱등).
        실제 로딩은 ``src.generation.pipeline_loader`` 모듈의 함수에 위임한다.
        """
        if self._loaded:
            return

        from src.generation.pipeline_loader import (
            apply_vram_optimizations,
            build_pipeline,
            load_ip_adapter,
            setup_scheduler,
        )

        self.pipe = build_pipeline(self.config, self.dtype)
        apply_vram_optimizations(self.pipe)
        load_ip_adapter(self.pipe, self.config["model"]["ip_adapter"])
        setup_scheduler(self.pipe, self.config["sampling"])

        self._loaded = True
        logger.info("StyleUnificationPipeline fully loaded.")

    # ------------------------------------------------------------------
    # Private helpers — transform internals
    # ------------------------------------------------------------------

    def _preprocess_source(
        self,
        source: Image.Image,
    ) -> tuple[Image.Image, Image.Image, npt.NDArray[np.float32]]:
        """Source 이미지를 전처리해 foreground_rgba, lineart, mask를 반환한다.

        Args:
            source: 원본 소스 PIL Image (RGB 또는 RGBA).

        Returns:
            (foreground_rgba, lineart_img, mask).
            foreground_rgba: RGBA PIL Image.
            lineart_img: RGB PIL Image (ControlNet 입력).
            mask: (H, W) float32 [0, 1].
        """
        bg_model: str = self.config["preprocessing"].get("bg_removal_model", "briaai/RMBG-1.4")
        foreground_rgba, mask = remove_background(source, model_name=bg_model)
        logger.debug("bg_removal done: size=%s", foreground_rgba.size)

        detector: str = self.config["preprocessing"].get("lineart_detector", "lineart_anime")
        lineart_img = extract_lineart(foreground_rgba, detector=detector)
        logger.debug("lineart extracted: size=%s", lineart_img.size)

        return foreground_rgba, lineart_img, mask

    def _build_controlnet_args(
        self,
        lineart_img: Image.Image,
    ) -> tuple[list[Image.Image], list[float]]:
        """Enabled ControlNet의 control_image 리스트와 scale 리스트를 빌드한다.

        현재는 lineart만 지원. depth 등 추가 시 이 함수를 확장한다.

        Args:
            lineart_img: 전처리에서 추출한 lineart RGB 이미지.

        Returns:
            (control_images, conditioning_scales).
        """
        enabled_cns = [c for c in self.config["model"]["controlnet"] if c.get("enabled", True)]

        control_images: list[Image.Image] = []
        scales: list[float] = []

        for cn_cfg in enabled_cns:
            # lineart / canny 타입 모두 lineart_img를 사용
            control_images.append(lineart_img)
            scales.append(float(cn_cfg["scale"]))

        return control_images, scales

    def _postprocess(
        self,
        generated: Image.Image,
        mask: npt.NDArray[np.float32],
        reference: Image.Image,
    ) -> tuple[Image.Image, npt.NDArray[np.uint8] | None]:
        """생성 결과에 alpha 복원과 palette quantize를 적용한다.

        Args:
            generated: SDXL 출력 RGB 이미지.
            mask: (H, W) float32 [0, 1]. remove_background() 반환값.
            reference: 원본 reference 이미지 (palette 추출용).

        Returns:
            ``(result_rgba, palette)``. palette는 ``palette_quantize=False`` 일 때 None.
        """
        post_cfg = self.config["postprocessing"]
        feather_px: int = int(post_cfg.get("alpha_feather", 1))

        # generated 크기와 mask 크기 불일치 시 mask를 리사이즈
        gen_w, gen_h = generated.size
        mask_h, mask_w = mask.shape
        if (mask_h, mask_w) != (gen_h, gen_w):
            logger.debug(
                "Resizing mask from (%d, %d) to (%d, %d)",
                mask_h,
                mask_w,
                gen_h,
                gen_w,
            )
            mask_pil = Image.fromarray((mask * 255).astype(np.uint8), mode="L")
            mask_pil = mask_pil.resize((gen_w, gen_h), Image.Resampling.BILINEAR)
            mask = np.array(mask_pil, dtype=np.float32) / 255.0

        result_rgba = restore_alpha(generated, mask, feather_px=feather_px)

        palette: npt.NDArray[np.uint8] | None = None
        if post_cfg.get("palette_quantize", True):
            palette_k: int = int(post_cfg.get("palette_k", 12))
            palette_strength: float = float(post_cfg.get("palette_strength", 0.7))
            palette = extract_palette(reference, k=palette_k)
            result_rgba = quantize_to_palette(result_rgba, palette, strength=palette_strength)
            logger.debug(
                "Palette quantize applied (k=%d, strength=%.2f)", palette_k, palette_strength
            )

        return result_rgba, palette

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def transform(
        self,
        source: Image.Image,
        reference: Image.Image,
        *,
        prompt: str | None = None,
        negative_prompt: str | None = None,
        seed: int | None = None,
        return_intermediates: bool = False,
    ) -> Image.Image | dict:
        """source를 reference 스타일로 변환해 RGBA 이미지를 반환한다.

        Args:
            source: 변환할 원본 게임 에셋 PIL Image. RGB 또는 RGBA.
            reference: 스타일 reference PIL Image. RGB 또는 RGBA.
            prompt: 생성 프롬프트. None이면 기본값 사용.
            negative_prompt: 네거티브 프롬프트. None이면 기본값 사용.
            seed: 재현성 seed. None이면 랜덤 동작.
            return_intermediates: True면 중간 결과물을 포함한 dict를 반환.

        Returns:
            RGBA PIL Image. return_intermediates=True면 dict:
            ``{"result": RGBA, "lineart": RGB, "foreground": RGBA,
            "mask": ndarray, "palette": ndarray | None}``.

        Raises:
            ImportError: diffusers 등 필수 라이브러리 미설치 시.
            OSError: 모델 다운로드/로딩 실패 시.
            ValueError: 설정 값이 유효하지 않을 때.
        """
        self._load_pipeline()
        pipe = self._require_pipe_loaded()

        # Generator 생성 (seed 있을 때만)
        generator: torch.Generator | None = (
            make_generator(seed, device=self.device) if seed is not None else None
        )

        # 전처리
        foreground_rgba, lineart_img, mask = self._preprocess_source(source)
        source_rgb = foreground_rgba.convert("RGB")

        # 프롬프트 결정
        final_prompt = prompt if prompt is not None else _DEFAULT_PROMPT
        final_negative_prompt = (
            negative_prompt if negative_prompt is not None else _DEFAULT_NEGATIVE_PROMPT
        )

        # ControlNet 인자 빌드
        control_images, conditioning_scales = self._build_controlnet_args(lineart_img)

        # 단일 ControlNet이면 리스트 unwrap (diffusers 인터페이스 호환)
        control_arg: Any = control_images[0] if len(control_images) == 1 else control_images
        scale_arg: Any = (
            conditioning_scales[0] if len(conditioning_scales) == 1 else conditioning_scales
        )

        sampling_cfg = self.config["sampling"]
        num_steps: int = int(sampling_cfg["steps"])
        guidance_scale: float = float(sampling_cfg["cfg_scale"])
        strength: float = float(sampling_cfg["denoising_strength"])

        logger.info(
            "Running inference (steps=%d, cfg=%.1f, strength=%.2f, seed=%s)",
            num_steps,
            guidance_scale,
            strength,
            seed,
        )

        with torch.inference_mode():
            pipe_output: Any = pipe(
                prompt=final_prompt,
                negative_prompt=final_negative_prompt,
                image=source_rgb,
                control_image=control_arg,
                ip_adapter_image=reference,
                num_inference_steps=num_steps,
                guidance_scale=guidance_scale,
                strength=strength,
                controlnet_conditioning_scale=scale_arg,
                generator=generator,
            )
        generated: Image.Image = pipe_output.images[0]
        logger.debug("Inference done: generated size=%s", generated.size)

        # 후처리 (alpha 복원 + palette quantize)
        result_rgba, palette = self._postprocess(generated, mask, reference)

        # RGBA 보장 (CLAUDE.md 원칙 4)
        if result_rgba.mode != "RGBA":
            result_rgba = result_rgba.convert("RGBA")

        logger.info(
            "transform() complete: output size=%s, mode=%s", result_rgba.size, result_rgba.mode
        )

        if return_intermediates:
            return {
                "result": result_rgba,
                "lineart": lineart_img,
                "foreground": foreground_rgba,
                "mask": mask,
                "palette": palette,
            }

        return result_rgba

    def unload(self) -> None:
        """파이프라인 모델을 메모리에서 해제하고 CUDA 캐시를 비운다.

        연속적인 실험 사이에 VRAM을 회수할 때 호출한다.
        unload() 이후 transform()을 호출하면 모델이 다시 lazy-load된다.
        """
        if not self._loaded:
            logger.debug("unload() called but pipeline is not loaded; no-op.")
            return

        del self.pipe
        self.pipe = None
        self._loaded = False

        if self.device.type == "cuda":
            torch.cuda.empty_cache()
            logger.info("CUDA cache cleared.")

        logger.info("StyleUnificationPipeline unloaded.")

    # ------------------------------------------------------------------
    # HF_TOKEN 환경변수 경고 (optional utility)
    # ------------------------------------------------------------------

    @staticmethod
    def check_hf_token() -> None:
        """HF_TOKEN 환경변수가 없으면 경고를 로깅한다.

        일부 모델(animagine-xl-3.1 등)은 HuggingFace 토큰 없이도 다운로드되지만,
        gated 모델을 사용할 경우 토큰이 필요하다.
        """
        if os.environ.get("HF_TOKEN") is None:
            logger.warning(
                "HF_TOKEN is not set. Some gated models may fail to download. Set it with: export HF_TOKEN=<your_token>"
            )
