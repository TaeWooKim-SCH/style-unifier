"""그룹 A4 — HuggingFace 모델 일괄 다운로드 스크립트.

필요한 모든 모델을 로컬 캐시에 저장하고, 다운로드 성공/실패를 요약 출력한다.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# 모델 명세
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelSpec:
    """단일 HuggingFace 모델의 다운로드 명세.

    Attributes:
        key: CLI --only 필터에서 사용하는 단축 키 (예: ``"sdxl"``).
        repo_id: HuggingFace 리포지토리 ID.
        description: 사람이 읽을 수 있는 용도 설명.
        allow_patterns: 다운로드할 파일 패턴 목록. ``None`` 이면 전체.
        ignore_patterns: 제외할 파일 패턴 목록. ``allow_patterns``보다 우선.
    """

    key: str
    repo_id: str
    description: str
    allow_patterns: tuple[str, ...] | None = None
    ignore_patterns: tuple[str, ...] | None = None


_DIFFUSERS_IGNORE: tuple[str, ...] = (
    "*.onnx",
    "*.onnx_data",
    "openvino_model.*",
    "**/openvino_model.*",
    "*.msgpack",
    "**/*.msgpack",
    "*flax*",
    "**/*flax*",
    "*.fp16.safetensors",
    "**/*.fp16.safetensors",
    "*.bin",
    "**/*.bin",
    "*.ckpt",
    "**/*.ckpt",
    "*.png",
    "*.jpg",
    "*.jpeg",
    "*.gif",
)


MODELS: tuple[ModelSpec, ...] = (
    ModelSpec(
        key="sdxl",
        repo_id="stabilityai/stable-diffusion-xl-base-1.0",
        description="SDXL 베이스 모델",
        ignore_patterns=_DIFFUSERS_IGNORE + (
            "sd_xl_base_1.0.safetensors",
            "sd_xl_base_1.0_0.9vae.safetensors",
            "sd_xl_offset_example-lora_1.0.safetensors",
        ),
    ),
    ModelSpec(
        key="animagine",
        repo_id="cagliostrolab/animagine-xl-3.1",
        description="Animagine-XL-3.1 — 일러스트 풍 파인튜닝",
        ignore_patterns=_DIFFUSERS_IGNORE + (
            "animagine-xl-3.1.safetensors",
        ),
    ),
    ModelSpec(
        key="ipadapter",
        repo_id="h94/IP-Adapter",
        description="IP-Adapter — 스타일 인코딩 (SDXL plus vit-h variant만)",
        allow_patterns=(
            "sdxl_models/ip-adapter-plus_sdxl_vit-h.safetensors",
            "models/image_encoder/config.json",
            "models/image_encoder/model.safetensors",
        ),
    ),
    ModelSpec(
        key="controlnet_lineart",
        repo_id="diffusers/controlnet-canny-sdxl-1.0",
        description="ControlNet canny/lineart — 라인 컨디션 (추후 lineart-specific으로 교체 가능)",
        ignore_patterns=_DIFFUSERS_IGNORE,
    ),
    ModelSpec(
        key="controlnet_depth",
        repo_id="diffusers/controlnet-depth-sdxl-1.0",
        description="ControlNet depth — 깊이 컨디션",
        ignore_patterns=_DIFFUSERS_IGNORE,
    ),
    ModelSpec(
        key="rmbg",
        repo_id="briaai/RMBG-1.4",
        description="RMBG-1.4 — 배경 제거",
        ignore_patterns=(
            "*.onnx",
            "model.pth",
        ),
    ),
    ModelSpec(
        key="clip",
        repo_id="laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
        description="CLIP ViT-H/14 — transformers 포맷만 (open_clip 중복 제외)",
        allow_patterns=(
            "config.json",
            "model.safetensors",
            "preprocessor_config.json",
            "merges.txt",
            "vocab.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
        ),
    ),
    ModelSpec(
        key="dinov2",
        repo_id="facebook/dinov2-large",
        description="DINOv2-Large — 평가용 identity 메트릭",
        ignore_patterns=(
            "*.bin",
            "*.msgpack",
        ),
    ),
)

_KEY_TO_SPEC: dict[str, ModelSpec] = {spec.key: spec for spec in MODELS}


# ---------------------------------------------------------------------------
# 다운로드 로직
# ---------------------------------------------------------------------------


@dataclass
class DownloadResult:
    """단일 모델의 다운로드 결과."""

    spec: ModelSpec
    success: bool
    error: str | None = None
    local_dir: Path | None = None


def _download_one(
    spec: ModelSpec,
    cache_dir: Path,
    token: str | None,
    dry_run: bool,
) -> DownloadResult:
    """단일 모델을 HuggingFace에서 다운로드한다.

    Args:
        spec: 다운로드할 모델의 명세.
        cache_dir: HuggingFace 캐시 디렉토리.
        token: HuggingFace API 토큰. ``None`` 이면 캐시/HF_TOKEN 환경변수 사용.
        dry_run: ``True`` 이면 실제 다운로드 없이 명세만 출력.

    Returns:
        다운로드 결과를 담은 ``DownloadResult``.
    """
    logger.info(
        "[%s] %s — %s",
        spec.key,
        spec.repo_id,
        spec.description,
    )

    if dry_run:
        allow_str = ", ".join(spec.allow_patterns) if spec.allow_patterns else "전체"
        ignore_str = ", ".join(spec.ignore_patterns) if spec.ignore_patterns else "없음"
        logger.info("  DRY-RUN: allow=%s | ignore=%s", allow_str, ignore_str)
        return DownloadResult(spec=spec, success=True)

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        return DownloadResult(
            spec=spec,
            success=False,
            error=f"huggingface_hub import 실패: {exc}",
        )

    try:
        local_dir_str: str = snapshot_download(
            repo_id=spec.repo_id,
            cache_dir=str(cache_dir),
            token=token,
            allow_patterns=list(spec.allow_patterns) if spec.allow_patterns else None,
            ignore_patterns=list(spec.ignore_patterns) if spec.ignore_patterns else None,
        )
        local_path = Path(local_dir_str)
        logger.info("  저장 위치: %s", local_path)
        return DownloadResult(spec=spec, success=True, local_dir=local_path)
    except OSError as exc:
        logger.error("  [%s] 다운로드 실패 (OSError): %s", spec.key, exc)
        return DownloadResult(spec=spec, success=False, error=str(exc))
    except ValueError as exc:
        logger.error("  [%s] 다운로드 실패 (ValueError): %s", spec.key, exc)
        return DownloadResult(spec=spec, success=False, error=str(exc))


def _resolve_token(cli_token: str | None) -> str | None:
    """CLI 인자 → 환경변수 → None 순으로 토큰을 결정한다."""
    if cli_token:
        return cli_token
    env_token = os.environ.get("HF_TOKEN")
    if env_token:
        logger.info("HF_TOKEN 환경변수에서 토큰 사용")
        return env_token
    logger.info("토큰 없음: HuggingFace 캐시 인증(huggingface-cli login) 사용")
    return None


def _filter_models(only_keys: list[str] | None) -> list[ModelSpec]:
    """only_keys 필터를 적용해 다운로드할 모델 목록을 반환한다.

    Args:
        only_keys: 다운로드할 모델 키 목록. ``None`` 또는 빈 리스트면 전체.

    Returns:
        필터링된 ``ModelSpec`` 리스트.

    Raises:
        SystemExit: 유효하지 않은 키가 포함된 경우.
    """
    if not only_keys:
        return list(MODELS)

    valid_keys = set(_KEY_TO_SPEC.keys())
    invalid = [k for k in only_keys if k not in valid_keys]
    if invalid:
        logger.error("알 수 없는 모델 키: %s", ", ".join(invalid))
        logger.error("유효한 키: %s", ", ".join(sorted(valid_keys)))
        sys.exit(1)

    return [_KEY_TO_SPEC[k] for k in only_keys]


def run_download(
    cache_dir: Path,
    only_keys: list[str] | None,
    token: str | None,
    dry_run: bool,
) -> int:
    """모델 다운로드 메인 로직.

    Args:
        cache_dir: HuggingFace 캐시 디렉토리.
        only_keys: 특정 모델 키 목록. ``None`` 이면 전체.
        token: HuggingFace API 토큰.
        dry_run: ``True`` 이면 실제 다운로드 없이 명세만 출력.

    Returns:
        실패한 모델 수. 0이면 전체 성공.
    """
    resolved_token = _resolve_token(token)
    targets = _filter_models(only_keys)

    mode_label = "DRY-RUN" if dry_run else "다운로드"
    logger.info(
        "=== %s 시작: %d개 모델, 캐시=%s ===",
        mode_label,
        len(targets),
        cache_dir,
    )

    results: list[DownloadResult] = []
    for spec in targets:
        result = _download_one(spec, cache_dir, resolved_token, dry_run)
        results.append(result)

    # 요약
    success_count = sum(1 for r in results if r.success)
    fail_count = len(results) - success_count

    logger.info(
        "=== %s 완료: %d/%d 성공 ===",
        mode_label,
        success_count,
        len(results),
    )

    if fail_count > 0:
        logger.error("실패한 모델 (%d개):", fail_count)
        for r in results:
            if not r.success:
                logger.error("  - [%s] %s: %s", r.spec.key, r.spec.repo_id, r.error)

    if not dry_run:
        # 대략적인 캐시 크기 추정
        try:
            total_bytes = sum(
                f.stat().st_size
                for r in results
                if r.success and r.local_dir is not None
                for f in r.local_dir.rglob("*")
                if f.is_file()
            )
            total_gb = total_bytes / (1024**3)
            logger.info(
                "다운로드 완료: %d/%d 모델, 합계 ~%.1f GB", success_count, len(results), total_gb
            )
        except OSError:
            logger.info("다운로드 완료: %d/%d 모델 (크기 집계 실패)", success_count, len(results))

    return fail_count


# ---------------------------------------------------------------------------
# CLI 진입점
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="style-unifier 모델 일괄 다운로드 (그룹 A4)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="\n".join(
            [
                "사용 가능한 모델 키:",
                *[f"  {spec.key:20s} {spec.description}" for spec in MODELS],
            ]
        ),
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path.home() / ".cache" / "huggingface",
        help="HuggingFace 캐시 디렉토리 (기본: ~/.cache/huggingface)",
    )
    parser.add_argument(
        "--only",
        type=str,
        default=None,
        metavar="KEY[,KEY...]",
        help="특정 모델만 다운로드 (쉼표 구분). 예: --only sdxl,ipadapter",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="실제 다운로드 없이 어떤 모델을 받을지 출력만",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        metavar="TOKEN",
        help="HuggingFace API 토큰 (미지정 시 HF_TOKEN 환경변수 → 로그인 캐시 순으로 탐색)",
    )
    return parser


if __name__ == "__main__":
    parser = _build_parser()
    args = parser.parse_args()

    only_keys: list[str] | None = None
    if args.only:
        only_keys = [k.strip() for k in args.only.split(",") if k.strip()]

    exit_code = run_download(
        cache_dir=args.cache_dir,
        only_keys=only_keys,
        token=args.token,
        dry_run=args.dry_run,
    )
    sys.exit(exit_code)
