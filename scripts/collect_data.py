"""CC0 게임 에셋 수집 및 자동 태깅 스크립트.

로컬에 미리 다운로드된 디렉토리를 스캔하고, 라이선스를 검증하며,
open_clip 기반 CLIP 분류기로 카테고리를 자동 태깅한다.
결과는 manifest.json으로 저장된다.

주의: URL 다운로드는 지원하지 않는다. Kenney.nl은 패키지 단위 zip으로만
배포되므로, 사용자가 수동으로 다운로드·압축 해제한 디렉토리를 --source로
지정해야 한다.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from PIL import Image
from tqdm import tqdm

from src.utils.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# 상수
# ---------------------------------------------------------------------------

SUPPORTED_EXTENSIONS: frozenset[str] = frozenset({".png", ".jpg", ".jpeg", ".webp"})

LICENSE_FILENAMES: tuple[str, ...] = ("LICENSE.txt", "license.txt", "LICENSE", "license.json")

# CLIP 분류 후보. character / object / item만 프로젝트 도메인, 나머지는 reject.
CLIP_CATEGORIES: list[str] = ["character", "object", "item", "background", "ui"]
DOMAIN_CATEGORIES: frozenset[str] = frozenset({"character", "object", "item"})

KNOWN_CC0_KEYWORDS: tuple[str, ...] = (
    "cc0",
    "creative commons zero",
    "public domain",
    "cc-zero",
)
KNOWN_CCBY_KEYWORDS: tuple[str, ...] = (
    "cc-by",
    "cc by",
    "attribution",
    "creative commons attribution",
)


# ---------------------------------------------------------------------------
# 라이선스 유틸
# ---------------------------------------------------------------------------


def _detect_license_from_text(text: str) -> str | None:
    """라이선스 텍스트에서 라이선스 종류를 감지한다.

    Args:
        text: 라이선스 파일의 내용.

    Returns:
        감지된 라이선스 식별자 ("CC0", "CC-BY") 또는 None.
    """
    lower = text.lower()
    for kw in KNOWN_CC0_KEYWORDS:
        if kw in lower:
            return "CC0"
    for kw in KNOWN_CCBY_KEYWORDS:
        if kw in lower:
            return "CC-BY"
    return None


def _find_license_in_dir(directory: Path) -> str | None:
    """디렉토리(또는 상위 2레벨)에서 라이선스 파일을 찾아 라이선스 문자열을 반환한다.

    Args:
        directory: 검색을 시작할 디렉토리.

    Returns:
        감지된 라이선스 식별자 또는 None.
    """
    # 현재 디렉토리 → 부모 → 조부모 순서로 검색 (최대 2레벨 상위)
    search_dirs = [directory, directory.parent, directory.parent.parent]
    for search_dir in search_dirs:
        for fname in LICENSE_FILENAMES:
            candidate = search_dir / fname
            if candidate.is_file():
                try:
                    text = candidate.read_text(encoding="utf-8", errors="ignore")
                except OSError as exc:
                    logger.warning("Failed to read license file %s: %s", candidate, exc)
                    continue
                detected = _detect_license_from_text(text)
                if detected is not None:
                    return detected
                logger.debug("License file found at %s but type unrecognized", candidate)
    return None


# ---------------------------------------------------------------------------
# 이미지 스캔
# ---------------------------------------------------------------------------


def scan_images(source: Path) -> list[dict]:
    """소스 디렉토리를 재귀적으로 스캔하여 이미지 메타데이터 목록을 반환한다.

    각 이미지에 대해 크기, 알파 채널 여부, 파일 크기, 라이선스 정보를 수집한다.
    라이선스 파일이 없는 디렉토리의 이미지는 경고 후 제외된다.

    Args:
        source: 스캔할 루트 디렉토리.

    Returns:
        각 이미지의 메타데이터 dict 목록. 키: path, width, height,
        has_alpha, file_size_bytes, license.

    Raises:
        FileNotFoundError: source 디렉토리가 존재하지 않을 때.
    """
    if not source.is_dir():
        raise FileNotFoundError(f"Source directory does not exist: {source}")

    logger.info("Scanning images in: %s", source)

    image_files = [
        p for p in source.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    logger.info("Found %d image files", len(image_files))

    items: list[dict] = []
    # 라이선스 캐시 (디렉토리별 중복 탐색 방지)
    license_cache: dict[Path, str | None] = {}
    warned_dirs: set[Path] = set()

    for img_path in tqdm(image_files, desc="Scanning", unit="img"):
        parent = img_path.parent
        if parent not in license_cache:
            license_cache[parent] = _find_license_in_dir(parent)

        license_str = license_cache[parent]
        if license_str is None:
            if parent not in warned_dirs:
                logger.warning(
                    "No license file found in %s; skipping items as license unverified",
                    parent,
                )
                warned_dirs.add(parent)
            continue

        try:
            with Image.open(img_path) as img:
                width, height = img.size
                has_alpha = img.mode in ("RGBA", "LA", "PA")
        except OSError as exc:
            logger.warning("Cannot open image %s: %s", img_path, exc)
            continue

        items.append(
            {
                "path": str(img_path),
                "width": width,
                "height": height,
                "has_alpha": has_alpha,
                "file_size_bytes": img_path.stat().st_size,
                "license": license_str,
                "category": None,
                "category_confidence": None,
            }
        )

    logger.info("Valid images after license scan: %d", len(items))
    return items


# ---------------------------------------------------------------------------
# 라이선스 필터
# ---------------------------------------------------------------------------


def filter_by_license(items: list[dict], license_filter: str) -> list[dict]:
    """라이선스 필터 기준에 맞는 항목만 반환한다.

    Args:
        items: scan_images()의 반환 목록.
        license_filter: "CC0", "CC-BY", "any" 중 하나.

    Returns:
        필터링된 항목 목록.

    Raises:
        ValueError: license_filter가 지원되지 않는 값일 때.
    """
    allowed = {"CC0", "CC-BY", "any"}
    if license_filter not in allowed:
        raise ValueError(f"license_filter must be one of {allowed}, got {license_filter!r}")

    if license_filter == "any":
        logger.info("License filter: any — keeping all %d items", len(items))
        return items

    filtered = [it for it in items if it["license"] == license_filter]
    logger.info("License filter '%s': %d → %d items", license_filter, len(items), len(filtered))
    return filtered


# ---------------------------------------------------------------------------
# 자동 태깅 (CLIP)
# ---------------------------------------------------------------------------


def auto_tag_items(items: list[dict], clip_model: str) -> None:
    """CLIP 임베딩으로 각 이미지의 카테고리를 in-place 태깅한다.

    카테고리 후보: ["character", "object", "item", "background", "ui"].
    background / ui로 분류된 항목은 도메인 외(reject)로 기록된다.

    Args:
        items: scan_images() 반환 목록. category 필드를 채운다 (in-place).
        clip_model: open_clip 모델 이름 (예: "ViT-B-32").
    """
    if not items:
        logger.info("No items to tag — skipping CLIP auto-tagging")
        return

    # open_clip은 --auto-tag 시에만 lazy import
    try:
        import open_clip  # type: ignore[import-untyped]
        import torch
    except ImportError as exc:
        raise ImportError(
            "open_clip_torch is required for --auto-tag. Install via: pip install open-clip-torch"
        ) from exc

    from src.utils.device import get_device

    device = get_device()
    logger.info("Loading CLIP model '%s' on %s for auto-tagging", clip_model, device)

    try:
        model, _, preprocess = open_clip.create_model_and_transforms(
            clip_model, pretrained="openai"
        )
    except (RuntimeError, ValueError, OSError) as exc:
        logger.warning(
            "Failed to load CLIP model '%s' with pretrained='openai': %s. Trying laion2b_s34b_b79k...",
            clip_model,
            exc,
        )
        model, _, preprocess = open_clip.create_model_and_transforms(
            clip_model, pretrained="laion2b_s34b_b79k"
        )

    model = model.to(device)
    model.eval()

    tokenizer = open_clip.get_tokenizer(clip_model)
    texts = [f"a {cat} game asset" for cat in CLIP_CATEGORIES]
    with torch.inference_mode():
        text_tokens = tokenizer(texts).to(device)
        text_feats = model.encode_text(text_tokens)
        text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)

    tagged = 0
    rejected = 0
    for item in tqdm(items, desc="Auto-tagging", unit="img"):
        img_path = Path(item["path"])
        try:
            with Image.open(img_path).convert("RGB") as img:
                preprocessed = preprocess(img)  # type: ignore[operator]
                img_tensor = preprocessed.unsqueeze(0).to(device)  # type: ignore[union-attr]
        except OSError as exc:
            logger.warning("Cannot open %s for tagging: %s", img_path, exc)
            continue

        with torch.inference_mode():
            img_feats = model.encode_image(img_tensor)
            img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)
            sims = (img_feats @ text_feats.T).squeeze(0)

        best_idx = int(sims.argmax().item())
        best_cat = CLIP_CATEGORIES[best_idx]
        confidence = float(sims[best_idx].item())

        item["category"] = best_cat
        item["category_confidence"] = round(confidence, 4)

        if best_cat in DOMAIN_CATEGORIES:
            tagged += 1
        else:
            rejected += 1
            logger.debug(
                "Rejected (out-of-domain): %s → %s (%.3f)", img_path.name, best_cat, confidence
            )

    logger.info(
        "Auto-tagging complete: %d tagged (in-domain), %d rejected (out-of-domain)",
        tagged,
        rejected,
    )


# ---------------------------------------------------------------------------
# Manifest 저장
# ---------------------------------------------------------------------------


def _compute_stats(items: list[dict]) -> dict:
    """아이템 목록에서 카테고리별 통계를 계산한다."""
    stats: dict[str, int] = {"total": len(items)}
    for cat in CLIP_CATEGORIES:
        stats[cat] = sum(1 for it in items if it.get("category") == cat)
    return stats


def save_manifest(
    items: list[dict],
    out_path: Path,
    source: Path,
    license_filter: str,
) -> None:
    """수집 결과를 manifest JSON 파일로 저장한다.

    Args:
        items: 저장할 이미지 메타데이터 목록.
        out_path: manifest.json 저장 경로.
        source: 원본 소스 디렉토리.
        license_filter: 적용한 라이선스 필터.
    """
    manifest = {
        "version": "1.0",
        "source": str(source),
        "license_filter": license_filter,
        "items": items,
        "stats": _compute_stats(items),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Manifest saved → %s (%d items)", out_path, len(items))


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main(args: argparse.Namespace) -> int:
    """데이터 수집 스크립트의 메인 진입점.

    Args:
        args: argparse로 파싱된 CLI 인자.

    Returns:
        프로세스 종료 코드. 0 = 성공, 1 = 실패.
    """
    t_start = time.monotonic()

    # 1. 소스 결정
    source_str: str = args.source
    if source_str.startswith("http://") or source_str.startswith("https://"):
        logger.warning(
            "URL 다운로드 미지원. Kenney.nl은 zip 단위 배포이므로 수동 다운로드 후 --source로 지정하세요."
        )
        return 1

    source = Path(source_str).expanduser().resolve()
    if not source.is_dir():
        logger.error("Source path does not exist or is not a directory: %s", source)
        return 1

    out_dir = Path(args.out_dir).expanduser().resolve()
    manifest_path = Path(args.manifest).expanduser().resolve()

    logger.info("Source: %s", source)
    logger.info("Out dir: %s", out_dir)
    logger.info("Manifest: %s", manifest_path)
    logger.info("Max items: %d, License filter: %s", args.max_items, args.license_filter)
    if args.dry_run:
        logger.info("[DRY-RUN] No files will be written")

    # 2. 스캔
    items = scan_images(source)

    # 3. 라이선스 필터
    items = filter_by_license(items, args.license_filter)

    # 4. 상한 적용
    if len(items) > args.max_items:
        logger.info("Trimming %d → %d items (--max-items)", len(items), args.max_items)
        items = items[: args.max_items]

    # 5. 자동 태깅 (선택)
    if args.auto_tag:
        auto_tag_items(items, args.clip_model)

    # 6. 저장 / dry-run 출력
    stats = _compute_stats(items)
    elapsed = time.monotonic() - t_start

    if args.dry_run:
        logger.info("[DRY-RUN] Stats: %s", json.dumps(stats))
        logger.info("[DRY-RUN] Elapsed: %.1fs — no files written", elapsed)
    else:
        out_dir.mkdir(parents=True, exist_ok=True)
        save_manifest(items, manifest_path, source, args.license_filter)
        logger.info(
            "Done in %.1fs — total=%d, character=%d, object=%d, item=%d",
            elapsed,
            stats.get("total", 0),
            stats.get("character", 0),
            stats.get("object", 0),
            stats.get("item", 0),
        )

    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="CC0 게임 에셋 수집 및 CLIP 자동 태깅 스크립트",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--source",
        required=True,
        help=(
            "데이터 소스 경로 (로컬 디렉토리). "
            "URL은 미지원 — Kenney zip을 수동 해제 후 경로를 지정하세요."
        ),
    )
    parser.add_argument(
        "--out-dir",
        default="data/raw/kenney",
        help="출력 디렉토리 (manifest.json 포함).",
    )
    parser.add_argument(
        "--manifest",
        default="data/raw/kenney/manifest.json",
        help="결과 manifest JSON 저장 경로.",
    )
    parser.add_argument(
        "--max-items",
        type=int,
        default=200,
        help="처리할 최대 이미지 수.",
    )
    parser.add_argument(
        "--license-filter",
        choices=["CC0", "CC-BY", "any"],
        default="CC0",
        help="허용할 라이선스 종류.",
    )
    parser.add_argument(
        "--auto-tag",
        action="store_true",
        help="CLIP 분류기로 카테고리 자동 태깅 (character / object / item / 도메인 외).",
    )
    parser.add_argument(
        "--clip-model",
        default="ViT-B-32",
        help="open_clip 모델 이름 (--auto-tag 사용 시).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="파일 시스템 변경 없이 스캔 및 통계만 출력.",
    )

    parsed_args = parser.parse_args()
    sys.exit(main(parsed_args))
