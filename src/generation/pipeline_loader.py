"""StyleUnificationPipeline 모델 로딩 헬퍼.

이 모듈은 diffusers 파이프라인 인스턴스를 빌드·구성하는 책임만 담당한다.
변환 로직은 ``pipeline.py`` 의 ``StyleUnificationPipeline`` 클래스가 처리한다.

모든 함수는 무거운 import를 함수 내부에서 lazy하게 수행하므로,
diffusers가 없는 환경에서도 이 모듈을 import하는 것 자체는 안전하다.
"""

from __future__ import annotations

from typing import Any

import torch

from src.utils.logging import get_logger

logger = get_logger(__name__)

# DPMSolverMultistepScheduler 식별자
_DPM_SCHEDULER_NAME = "DPMSolverMultistepScheduler"


def load_controlnets(config: dict, dtype: torch.dtype) -> list[Any]:
    """config의 enabled=True인 ControlNet 모델을 로드해 리스트로 반환한다.

    Args:
        config: ``configs/default.yaml`` 구조의 설정 dict.
        dtype: 모델 로딩에 사용할 torch dtype (fp16 / fp32).

    Returns:
        ControlNetModel 인스턴스 리스트. enabled=True인 항목만 포함.

    Raises:
        ImportError: diffusers가 설치되어 있지 않을 때.
        OSError: 모델 다운로드/로딩 실패 시.
    """
    try:
        from diffusers import ControlNetModel  # type: ignore[import-untyped]
    except ImportError as exc:
        raise ImportError("diffusers is required. Install with: pip install diffusers") from exc

    controlnet_cfgs = [c for c in config["model"]["controlnet"] if c.get("enabled", True)]

    logger.info("Loading %d ControlNet model(s)...", len(controlnet_cfgs))
    controlnets: list[Any] = []
    for cn_cfg in controlnet_cfgs:
        logger.info("  - %s (%s)", cn_cfg["type"], cn_cfg["repo"])
        model: Any = ControlNetModel.from_pretrained(
            cn_cfg["repo"],
            torch_dtype=dtype,
        )
        controlnets.append(model)

    return controlnets


def build_pipeline(config: dict, dtype: torch.dtype) -> Any:
    """ControlNet을 로드하고 SDXL img2img 파이프라인을 생성해 반환한다.

    반환된 파이프라인은 IP-Adapter, scheduler, VRAM 최적화가 적용되지 않은
    빈 상태다. 이후 ``apply_vram_optimizations()``, ``load_ip_adapter()``,
    ``setup_scheduler()`` 를 순서대로 호출해야 한다.

    Args:
        config: ``configs/default.yaml`` 구조의 설정 dict.
        dtype: 모델 로딩에 사용할 torch dtype.

    Returns:
        ``StableDiffusionXLControlNetImg2ImgPipeline`` 인스턴스.

    Raises:
        ImportError: diffusers가 설치되어 있지 않을 때.
        OSError: base_checkpoint 다운로드/로딩 실패 시.
    """
    try:
        from diffusers.pipelines.controlnet.pipeline_controlnet_sd_xl_img2img import (  # type: ignore[import-untyped]
            StableDiffusionXLControlNetImg2ImgPipeline,
        )
    except ImportError as exc:
        raise ImportError("diffusers is required. Install with: pip install diffusers") from exc

    controlnets = load_controlnets(config, dtype)

    base_checkpoint: str = config["model"]["base_checkpoint"]
    logger.info("Loading base pipeline: %s", base_checkpoint)

    # controlnets가 1개면 리스트 unwrap (diffusers 인터페이스 호환)
    cn_arg: Any = controlnets[0] if len(controlnets) == 1 else controlnets

    pipe: Any = StableDiffusionXLControlNetImg2ImgPipeline.from_pretrained(
        base_checkpoint,
        controlnet=cn_arg,
        torch_dtype=dtype,
    )
    return pipe


def apply_attention_optimizations(pipe: Any) -> None:
    """xformers attention 최적화를 적용한다.

    반드시 ``load_ip_adapter()`` **이전**에 호출해야 한다. 이유:
    ``enable_xformers_memory_efficient_attention()`` 은 모든 attention processor를
    ``XFormersAttnProcessor`` 로 reset하므로, IP-Adapter가 cross-attn에 설치하는
    ``IPAdapterAttnProcessor`` 를 뒤에서 덮어쓰면 IP-Adapter가 무력화된다.
    이 순서로는 self-attn에는 xformers, cross-attn에는 IP-Adapter processor가
    공존한다.

    Args:
        pipe: xformers를 적용할 diffusers 파이프라인 인스턴스.
    """
    try:
        pipe.enable_xformers_memory_efficient_attention()
        logger.info("xformers memory-efficient attention enabled.")
    except ModuleNotFoundError:
        logger.warning(
            "xformers not installed; using default attention. "
            "Install xformers for lower VRAM usage: pip install xformers"
        )


def apply_memory_optimizations(pipe: Any) -> None:
    """CPU offload + VAE slicing을 적용한다.

    반드시 ``load_ip_adapter()`` **이후**에 호출해야 한다. 이유:
    ``enable_model_cpu_offload()`` 는 호출 시점에 존재하는 components(unet, vae,
    text_encoder 등)에만 offload hook을 부착한다. IP-Adapter의 ``image_encoder``
    가 그 후에 추가되면 hook이 없어 CPU에 영구 거주 → dtype/device mismatch.

    Args:
        pipe: 메모리 최적화를 적용할 diffusers 파이프라인 인스턴스.
    """
    # 필수 — CLAUDE.md 절대 원칙
    pipe.enable_model_cpu_offload()
    logger.info("CPU offload enabled.")

    pipe.enable_vae_slicing()
    logger.info("VAE slicing enabled.")


def apply_vram_optimizations(pipe: Any) -> None:
    """후방 호환 래퍼 — 신규 호출자는 ``apply_attention_optimizations`` +
    ``apply_memory_optimizations`` 를 IP-Adapter 로딩 전/후로 나누어 호출해야 한다.

    Args:
        pipe: 최적화를 적용할 diffusers 파이프라인 인스턴스.
    """
    apply_attention_optimizations(pipe)
    apply_memory_optimizations(pipe)


def load_ip_adapter(pipe: Any, ip_config: dict) -> None:
    """파이프라인에 IP-Adapter 가중치를 로드하고 scale을 설정한다.

    Args:
        pipe: IP-Adapter를 적용할 diffusers 파이프라인 인스턴스.
        ip_config: ``config["model"]["ip_adapter"]`` 서브딕셔너리.
            ``repo``, ``subfolder``, ``weight_name``, ``scale`` 키를 포함해야 한다.
    """
    image_encoder_folder: str = ip_config.get("image_encoder_folder", "models/image_encoder")
    logger.info(
        "Loading IP-Adapter: %s/%s/%s (image_encoder=%s)",
        ip_config["repo"],
        ip_config["subfolder"],
        ip_config["weight_name"],
        image_encoder_folder,
    )
    # image_encoder_folder를 명시해야 한다 — ip-adapter-plus_sdxl_vit-h는 ViT-H/14 (1280 dim)
    # 용이지만, diffusers 기본값은 ``{subfolder}/image_encoder``이며 h94/IP-Adapter의
    # sdxl_models/image_encoder/는 ViT-bigG (1664 dim) 다. 명시 없으면 dim mismatch.
    pipe.load_ip_adapter(
        ip_config["repo"],
        subfolder=ip_config["subfolder"],
        weight_name=ip_config["weight_name"],
        image_encoder_folder=image_encoder_folder,
    )
    scale: float = float(ip_config["scale"])
    pipe.set_ip_adapter_scale(scale)
    logger.info("IP-Adapter loaded (scale=%.2f)", scale)


def setup_scheduler(pipe: Any, sampling_config: dict) -> None:
    """config의 scheduler 설정을 파이프라인에 적용한다.

    현재 ``DPMSolverMultistepScheduler`` 만 명시적으로 지원한다.
    다른 이름이 지정되면 경고를 남기고 파이프라인 기본 scheduler를 유지한다.

    Args:
        pipe: scheduler를 교체할 diffusers 파이프라인 인스턴스.
        sampling_config: ``config["sampling"]`` 서브딕셔너리.
            ``scheduler``, ``scheduler_config`` 키를 선택적으로 포함한다.

    Raises:
        ImportError: diffusers가 설치되어 있지 않을 때.
    """
    scheduler_name: str = sampling_config.get("scheduler", _DPM_SCHEDULER_NAME)

    if scheduler_name != _DPM_SCHEDULER_NAME:
        logger.warning(
            "Scheduler '%s' is not explicitly supported; keeping default.",
            scheduler_name,
        )
        return

    try:
        from diffusers import DPMSolverMultistepScheduler  # type: ignore[import-untyped]
    except ImportError as exc:
        raise ImportError("diffusers is required. Install with: pip install diffusers") from exc

    scheduler_cfg: dict = sampling_config.get("scheduler_config", {})
    overrides: dict = {}
    if "use_karras_sigmas" in scheduler_cfg:
        overrides["use_karras_sigmas"] = scheduler_cfg["use_karras_sigmas"]
    if "algorithm_type" in scheduler_cfg:
        overrides["algorithm_type"] = scheduler_cfg["algorithm_type"]

    pipe.scheduler = DPMSolverMultistepScheduler.from_config(
        pipe.scheduler.config,
        **overrides,
    )
    logger.info("Scheduler set to DPMSolverMultistepScheduler (overrides=%s)", overrides)


def run_inference_step(
    pipe: Any,
    *,
    source_rgb: Any,
    control_arg: Any,
    scale_arg: Any,
    reference: Any,
    prompt: str,
    neg_prompt: str,
    num_steps: int,
    guidance_scale: float,
    strength: float,
    ip_adapter_scale: float | None,
    generator: torch.Generator | None,
) -> Any:
    """SDXL + ControlNet + IP-Adapter 추론 호출 + 결과 PIL Image 추출.

    ``StyleUnificationPipeline._run_inference`` 의 실제 구현.
    모델 객체(``pipe``)를 직접 받으므로 파이프라인 클래스에 의존하지 않는다.

    Args:
        pipe: 로드된 StableDiffusionXLControlNetImg2ImgPipeline 인스턴스.
        source_rgb: RGB PIL Image (img2img 입력).
        control_arg: ControlNet control_image. 단일 Image 또는 리스트.
        scale_arg: controlnet_conditioning_scale. 단일 float 또는 리스트.
        reference: IP-Adapter 스타일 reference 이미지.
        prompt: 생성 프롬프트.
        neg_prompt: 네거티브 프롬프트.
        num_steps: diffusion 스텝 수.
        guidance_scale: CFG guidance scale.
        strength: img2img denoising strength.
        ip_adapter_scale: IP-Adapter scale. None이면 호출에 포함하지 않음.
        generator: 재현성 Generator. None이면 랜덤.

    Returns:
        생성된 PIL Image (RGB).
    """
    # diffusers SDXL ControlNet img2img는 width/height 미명시 시 image와
    # control_image를 다른 로직으로 보정해 latent shape mismatch가 발생할 수 있다.
    # 8의 배수로 강제 정렬한 동일 크기를 명시한다.
    from PIL import Image as _PILImage

    src_w, src_h = source_rgb.size
    width = (src_w // 8) * 8
    height = (src_h // 8) * 8
    if (width, height) != (src_w, src_h):
        source_rgb = source_rgb.resize((width, height), _PILImage.Resampling.LANCZOS)

    def _resize_control(c: Any) -> Any:
        return c.resize((width, height), _PILImage.Resampling.LANCZOS)

    control_arg_resized: Any = (
        [_resize_control(c) for c in control_arg]
        if isinstance(control_arg, list)
        else _resize_control(control_arg)
    )

    call_kwargs: dict[str, Any] = {
        "prompt": prompt,
        "negative_prompt": neg_prompt,
        "image": source_rgb,
        "control_image": control_arg_resized,
        "ip_adapter_image": reference,
        "num_inference_steps": num_steps,
        "guidance_scale": guidance_scale,
        "strength": strength,
        "controlnet_conditioning_scale": scale_arg,
        "generator": generator,
        "width": width,
        "height": height,
    }
    if ip_adapter_scale is not None:
        call_kwargs["ip_adapter_scale"] = ip_adapter_scale

    with torch.inference_mode():
        pipe_output: Any = pipe(**call_kwargs)

    generated = pipe_output.images[0]
    logger.debug("Inference done: generated size=%s", generated.size)
    return generated
