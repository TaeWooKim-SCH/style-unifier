"""평가셋 페어 구축 스크립트.

collect_data.py로 생성된 manifest.json을 읽어 source / reference 페어를
샘플링하고, data/eval/ 디렉토리에 이미지를 복사(또는 심볼릭 링크)한다.
pairs.json에 페어 메타데이터를 저장한다.

페어링 전략:
- random: 전체 이미지에서 무작위 페어
- category_match: source와 reference가 같은 카테고리
- stratified: 캐릭터 50% / 사물 25% / 아이템 25% 분포 보장 (기본)
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

from PIL import Image
from scripts._pair_strategies import sample_pairs
from tqdm import tqdm

from src.utils.logging import get_logger
from src.utils.repro import set_seed

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# 유틸
# ---------------------------------------------------------------------------


def _load_manifest(manifest_path: Path) -> dict:
    """manifest.json을 로드한다.

    Args:
        manifest_path: manifest.json 파일 경로.

    Returns:
        파싱된 manifest dict.

    Raises:
        FileNotFoundError: manifest 파일이 없을 때.
        ValueError: JSON 파싱 실패 시.
    """
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    try:
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in manifest: {manifest_path}") from exc


def _group_by_category(items: list[dict]) -> dict[str, list[dict]]:
    """항목 목록을 카테고리별로 분류한다.

    category가 None이거나 도메인 외(background / ui)인 항목은 제외된다.

    Args:
        items: manifest의 items 목록.

    Returns:
        카테고리 이름 → 항목 목록 매핑.
    """
    groups: dict[str, list[dict]] = {}
    domain_cats = {"character", "object", "item"}
    for it in items:
        cat = it.get("category")
        if not isinstance(cat, str) or cat not in domain_cats:
            continue
        groups.setdefault(cat, []).append(it)
    return groups


def _make_pair(source_item: dict, ref_item: dict, pair_id: int) -> dict:
    """페어 메타데이터 dict를 생성한다."""
    return {
        "id": pair_id,
        "source": source_item["path"],
        "reference": ref_item["path"],
        "source_category": source_item.get("category"),
        "reference_category": ref_item.get("category"),
        # 페어의 대표 카테고리: source 기준
        "category": source_item.get("category"),
    }


# ---------------------------------------------------------------------------
# 파일 배치
# ---------------------------------------------------------------------------


def _copy_or_link(src_path: Path, dst_path: Path, use_symlink: bool) -> None:
    """이미지를 복사하거나 심볼릭 링크로 연결한다.

    Args:
        src_path: 원본 파일 경로.
        dst_path: 대상 파일 경로.
        use_symlink: True면 심볼릭 링크, False면 복사.
    """
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    if use_symlink:
        if dst_path.exists() or dst_path.is_symlink():
            dst_path.unlink()
        dst_path.symlink_to(src_path.resolve())
    else:
        shutil.copy2(src_path, dst_path)


def _ensure_png(path: Path, dst_path: Path) -> None:
    """이미지를 PNG(RGBA) 형식으로 변환하여 저장한다.

    원본이 .png가 아닌 경우 PIL로 변환한다.

    Args:
        path: 원본 이미지 경로.
        dst_path: 저장할 PNG 경로.
    """
    if path.suffix.lower() == ".png":
        shutil.copy2(path, dst_path)
        return
    with Image.open(path) as img:
        img.save(dst_path, format="PNG")


def deploy_pairs(
    pairs: list[tuple[dict, dict]],
    out_dir: Path,
    use_symlink: bool,
    dry_run: bool,
) -> list[dict]:
    """페어를 out_dir에 배치하고 pairs 메타데이터 목록을 반환한다.

    파일은 `source_NNN.png` / `reference_NNN.png` (NNN = 3자리 제로패딩)로 저장된다.

    Args:
        pairs: (source_item, reference_item) 튜플 목록.
        out_dir: 출력 디렉토리.
        use_symlink: True면 symlink, False면 copy.
        dry_run: True면 파일 변경 없이 계획만 출력.

    Returns:
        각 페어의 메타데이터 dict 목록.
    """
    pair_records: list[dict] = []

    for idx, (src_item, ref_item) in enumerate(
        tqdm(pairs, desc="Deploying pairs", unit="pair"), start=1
    ):
        src_dst = out_dir / f"source_{idx:03d}.png"
        ref_dst = out_dir / f"reference_{idx:03d}.png"

        record = _make_pair(src_item, ref_item, idx)
        record["source_deployed"] = str(src_dst)
        record["reference_deployed"] = str(ref_dst)
        pair_records.append(record)

        if dry_run:
            logger.debug(
                "[DRY-RUN] pair %03d: %s → %s | %s → %s",
                idx,
                src_item["path"],
                src_dst,
                ref_item["path"],
                ref_dst,
            )
            continue

        src_path = Path(src_item["path"])
        ref_path = Path(ref_item["path"])

        if use_symlink:
            _copy_or_link(src_path, src_dst, use_symlink=True)
            _copy_or_link(ref_path, ref_dst, use_symlink=True)
        else:
            _ensure_png(src_path, src_dst)
            _ensure_png(ref_path, ref_dst)

    return pair_records


# ---------------------------------------------------------------------------
# Pairs JSON 저장
# ---------------------------------------------------------------------------


def save_pairs_json(
    pair_records: list[dict],
    out_dir: Path,
    strategy: str,
    seed: int,
) -> None:
    """페어 메타데이터를 pairs.json으로 저장한다.

    Args:
        pair_records: deploy_pairs()의 반환 목록.
        out_dir: 저장할 디렉토리.
        strategy: 사용한 페어링 전략.
        seed: 사용한 랜덤 시드.
    """
    payload = {
        "n_pairs": len(pair_records),
        "strategy": strategy,
        "seed": seed,
        "pairs": pair_records,
    }
    out_path = out_dir / "pairs.json"
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("pairs.json saved → %s", out_path)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main(args: argparse.Namespace) -> int:
    """평가셋 페어 구축 스크립트의 메인 진입점.

    Args:
        args: argparse로 파싱된 CLI 인자.

    Returns:
        프로세스 종료 코드. 0 = 성공, 1 = 실패.
    """
    t_start = time.monotonic()

    manifest_path = Path(args.manifest).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()

    # 재현성 시드 설정 (torch는 lazy — 여기서는 random만 사용)
    set_seed(args.seed)

    logger.info("Manifest: %s", manifest_path)
    logger.info("Out dir: %s", out_dir)
    logger.info("n_pairs=%d, strategy=%s, seed=%d", args.n_pairs, args.strategy, args.seed)
    if args.dry_run:
        logger.info("[DRY-RUN] No files will be written")

    # 1. manifest 로드
    try:
        manifest = _load_manifest(manifest_path)
    except (FileNotFoundError, ValueError) as exc:
        logger.error("%s", exc)
        return 1

    items: list[dict] = manifest.get("items", [])
    logger.info("Loaded %d items from manifest", len(items))

    # 자동 태깅이 된 항목만 사용 (category가 있는 것)
    tagged = [it for it in items if it.get("category") is not None]
    logger.info("Items with category tag: %d", len(tagged))

    groups = _group_by_category(tagged)
    for cat, grp in groups.items():
        logger.info("  %s: %d items", cat, len(grp))

    total_domain = sum(len(g) for g in groups.values())
    if total_domain < 2:
        logger.error(
            "Need at least 2 domain-category items, got %d. Run collect_data.py --auto-tag first.",
            total_domain,
        )
        return 1

    # 2. 페어 샘플링
    try:
        pairs = sample_pairs(tagged, groups, args.n_pairs, args.strategy)
    except ValueError as exc:
        logger.error("Sampling failed: %s", exc)
        return 1
    logger.info("Sampled %d pairs", len(pairs))

    # 3. 파일 배치
    if not args.dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)

    use_symlink = args.symlink
    pair_records = deploy_pairs(pairs, out_dir, use_symlink=use_symlink, dry_run=args.dry_run)

    # 4. pairs.json 저장
    if not args.dry_run:
        save_pairs_json(pair_records, out_dir, args.strategy, args.seed)

    elapsed = time.monotonic() - t_start
    logger.info(
        "%s %d pairs in %.1fs → %s",
        "[DRY-RUN] Would deploy" if args.dry_run else "Deployed",
        len(pair_records),
        elapsed,
        out_dir,
    )
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="평가셋 source/reference 페어 구축 스크립트",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--manifest",
        default="data/raw/kenney/manifest.json",
        help="collect_data.py로 생성된 manifest.json 경로.",
    )
    parser.add_argument(
        "--out-dir",
        default="data/eval",
        help="페어 이미지 및 pairs.json을 저장할 디렉토리.",
    )
    parser.add_argument(
        "--n-pairs",
        type=int,
        default=30,
        help="생성할 페어 수.",
    )
    parser.add_argument(
        "--strategy",
        choices=["random", "category_match", "stratified"],
        default="stratified",
        help=(
            "페어링 전략. "
            "random: 무작위 | "
            "category_match: 같은 카테고리 | "
            "stratified: 캐릭터 50%/사물 25%/아이템 25% 분포 보장."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="재현성을 위한 랜덤 시드.",
    )

    symlink_group = parser.add_mutually_exclusive_group()
    symlink_group.add_argument(
        "--copy",
        dest="symlink",
        action="store_false",
        default=False,
        help="이미지를 복사한다 (기본).",
    )
    symlink_group.add_argument(
        "--symlink",
        dest="symlink",
        action="store_true",
        help="이미지를 심볼릭 링크로 연결한다.",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="파일 변경 없이 페어 계획만 출력.",
    )

    parsed_args = parser.parse_args()
    sys.exit(main(parsed_args))
