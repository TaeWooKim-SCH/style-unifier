"""build_eval_pairs.py의 페어 샘플링 전략 모음.

세 가지 전략을 제공한다: random, category_match, stratified.
이 모듈은 build_eval_pairs.py에서만 import된다.
"""

from __future__ import annotations

import random

from src.utils.logging import get_logger

logger = get_logger(__name__)

# stratified 전략에서의 카테고리별 목표 비율
STRATIFIED_RATIOS: dict[str, float] = {
    "character": 0.50,
    "object": 0.25,
    "item": 0.25,
}


def sample_random(items: list[dict], n_pairs: int) -> list[tuple[dict, dict]]:
    """전체 이미지 풀에서 무작위로 n_pairs 개의 (source, reference) 페어를 샘플링한다.

    source == reference 동일 파일 페어는 허용하지 않는다.

    Args:
        items: 전체 이미지 항목 목록.
        n_pairs: 생성할 페어 수.

    Returns:
        (source_item, reference_item) 튜플 목록.

    Raises:
        ValueError: 이미지가 2장 미만이어서 페어를 만들 수 없을 때.
    """
    if len(items) < 2:
        raise ValueError(f"Need at least 2 items for pairing, got {len(items)}")

    pairs: list[tuple[dict, dict]] = []
    attempts = 0
    max_attempts = n_pairs * 20

    while len(pairs) < n_pairs and attempts < max_attempts:
        src, ref = random.sample(items, 2)
        if src["path"] != ref["path"]:
            pairs.append((src, ref))
        attempts += 1

    if len(pairs) < n_pairs:
        logger.warning(
            "Could only generate %d/%d random pairs (insufficient unique items)",
            len(pairs),
            n_pairs,
        )
    return pairs


def sample_category_match(groups: dict[str, list[dict]], n_pairs: int) -> list[tuple[dict, dict]]:
    """source와 reference가 같은 카테고리인 페어를 샘플링한다.

    Args:
        groups: 카테고리 이름 → 항목 목록 매핑.
        n_pairs: 생성할 페어 수.

    Returns:
        (source_item, reference_item) 튜플 목록.

    Raises:
        ValueError: 2개 이상의 아이템이 있는 카테고리가 없을 때.
    """
    all_cats = [cat for cat, grp in groups.items() if len(grp) >= 2]
    if not all_cats:
        raise ValueError("No category has >= 2 items for category_match sampling")

    pairs: list[tuple[dict, dict]] = []
    attempts = 0
    max_attempts = n_pairs * 20

    while len(pairs) < n_pairs and attempts < max_attempts:
        cat = random.choice(all_cats)
        grp = groups[cat]
        src, ref = random.sample(grp, 2)
        if src["path"] != ref["path"]:
            pairs.append((src, ref))
        attempts += 1

    if len(pairs) < n_pairs:
        logger.warning(
            "Could only generate %d/%d category_match pairs",
            len(pairs),
            n_pairs,
        )
    return pairs


def sample_stratified(groups: dict[str, list[dict]], n_pairs: int) -> list[tuple[dict, dict]]:
    """캐릭터 50% / 사물 25% / 아이템 25% 분포로 페어를 샘플링한다.

    각 카테고리에서 source를 뽑고, reference는 전체 풀에서 무작위 선택
    (source와 같은 파일이 아닌 것).

    Args:
        groups: 카테고리 이름 → 항목 목록 매핑.
        n_pairs: 생성할 총 페어 수.

    Returns:
        (source_item, reference_item) 튜플 목록.
    """
    # 카테고리별 목표 개수 계산
    targets: dict[str, int] = {}
    remaining = n_pairs
    cats = list(STRATIFIED_RATIOS.keys())

    for i, cat in enumerate(cats):
        if i == len(cats) - 1:
            targets[cat] = remaining
        else:
            targets[cat] = round(n_pairs * STRATIFIED_RATIOS[cat])
            remaining -= targets[cat]

    all_items = [it for grp in groups.values() for it in grp]
    pairs: list[tuple[dict, dict]] = []

    for cat, count in targets.items():
        cat_items = groups.get(cat, [])
        if not cat_items:
            logger.warning("Category '%s' has no items — skipping %d pairs", cat, count)
            continue

        generated = 0
        attempts = 0
        max_attempts = count * 30

        while generated < count and attempts < max_attempts:
            src = random.choice(cat_items)
            ref = random.choice(all_items)
            if src["path"] != ref["path"]:
                pairs.append((src, ref))
                generated += 1
            attempts += 1

        if generated < count:
            logger.warning(
                "Stratified: only %d/%d pairs for category '%s'",
                generated,
                count,
                cat,
            )

    random.shuffle(pairs)
    return pairs


def sample_pairs(
    items: list[dict],
    groups: dict[str, list[dict]],
    n_pairs: int,
    strategy: str,
) -> list[tuple[dict, dict]]:
    """전략에 따라 (source, reference) 페어 목록을 생성한다.

    Args:
        items: 전체 이미지 항목 목록 (random 전략용).
        groups: 카테고리별 그룹 (category_match / stratified용).
        n_pairs: 생성할 페어 수.
        strategy: "random" | "category_match" | "stratified".

    Returns:
        (source_item, reference_item) 튜플 목록.

    Raises:
        ValueError: strategy가 지원되지 않는 값일 때.
    """
    if strategy == "random":
        return sample_random(items, n_pairs)
    if strategy == "category_match":
        return sample_category_match(groups, n_pairs)
    if strategy == "stratified":
        return sample_stratified(groups, n_pairs)
    raise ValueError(f"Unsupported strategy: {strategy!r}. Choose random/category_match/stratified")
