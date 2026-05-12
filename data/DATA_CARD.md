# DATA_CARD.md

> 평가 데이터셋의 출처, 라이선스, 라벨링 방법, 알려진 편향을 기록한다.
> 평가 실행 전 이 문서를 읽어 데이터셋 상태를 파악할 것.
>
> **Last updated**: 2026-05-12

---

## 1. 개요

| 항목 | 내용 |
|------|------|
| 목적 | style-unifier 파이프라인의 정량 평가 |
| 도메인 | 벡터/일러스트 스타일의 정적 2D 게임 에셋 (캐릭터, 사물, 아이템) |
| 총 평가 페어 수 | 30 세트 (target) |
| 수집 방법 | 수동 다운로드 (Kenney.nl) + CLIP 자동 태깅 + 수동 검수 |
| 라이선스 기준 | CC0 우선 (재배포·상업 이용 모두 허용) |
| 페어 전략 | stratified (캐릭터 50% / 사물 25% / 아이템 25%) |

---

## 2. 데이터 출처

### 2.1 주 출처: Kenney.nl

| 항목 | 내용 |
|------|------|
| URL | https://kenney.nl |
| 라이선스 | CC0 1.0 Universal (Public Domain) |
| 재배포 가능 여부 | 가능 (CC0는 귀속 의무 없음) |
| 취득 방법 | 패키지별 zip 수동 다운로드 → 압축 해제 |
| 에셋 유형 | 캐릭터, 사물, 아이템, 배경(도메인 외 제외) |

Kenney.nl의 모든 에셋은 CC0로 배포된다.
단, **픽셀 아트 패키지는 본 프로젝트 도메인 외**로 수집에서 제외한다.

> **미해결 질문**: Kenney.nl 에셋의 CC0 라이선스를 각 패키지별로 실제 확인 필요.
> `collect_data.py`가 `LICENSE.txt` 존재 여부로 검증하며, 없으면 해당 디렉토리를 제외한다.

### 2.2 보조 출처 (확장 시)

| 출처 | 라이선스 | 비고 |
|------|---------|------|
| OpenGameArt.org | CC0, CC-BY 혼재 | license_filter=CC0 강제 필요 |
| Itch.io CC0 에셋 | CC0 명시 항목만 | 개별 확인 필요 |
| 본인 작업물 | 작성자 소유 | before/after 데모용 |

### 2.3 수집 제외 기준

다음 항목은 수집 단계에서 **자동 제외**된다.

- 라이선스 파일(`LICENSE.txt` 등)이 없는 디렉토리의 이미지
- CC0 / CC-BY 이외의 라이선스 (기본 `--license-filter CC0`)
- CLIP 분류기가 `background` 또는 `ui`로 분류한 이미지 (도메인 외)

---

## 3. 카테고리 정의

본 프로젝트의 카테고리는 평가 시 카테고리별 성능 분석을 위한 것으로,
**추론 파이프라인에는 카테고리별 분기 로직이 없다** (동일 흐름으로 처리).

| 카테고리 | 레이블 | 정의 | 예시 |
|---------|-------|------|------|
| 캐릭터 | `character` | 사람, 동물, 몬스터 등 생물 개체 | 플레이어, 적, NPC, 동물 |
| 사물 | `object` | 환경 오브젝트, 소품 | 상자, 통, 가구, 장식물 |
| 아이템 | `item` | 플레이어가 획득·장비하는 물건 | 무기, 방어구, 포션, 열쇠, 보석 |
| (도메인 외) | `background` | 배경, 타일맵 | 풀밭, 하늘, 건물 외벽 |
| (도메인 외) | `ui` | UI 요소 | 버튼, 아이콘, 슬롯 프레임 |

`background`와 `ui`로 분류된 항목은 CLIP 자동 태깅 단계에서 **reject** 처리되어
평가셋에 포함되지 않는다.

---

## 4. 자동 태깅 방법

### 4.1 CLIP 기반 카테고리 분류

| 항목 | 내용 |
|------|------|
| 라이브러리 | `open-clip-torch==2.24.0` |
| 기본 모델 | `ViT-B-32` (pretrained: openai) |
| 텍스트 프롬프트 | `"a {category} game asset"` (5개 카테고리) |
| 분류 방법 | 이미지 임베딩 ↔ 텍스트 임베딩 cosine similarity argmax |
| 결과 기록 | manifest.json의 `category`, `category_confidence` 필드 |

분류기는 단순 cosine argmax이므로 **정밀도 보장이 되지 않는다**.
이 때문에 수동 검수 절차(Section 5)가 필요하다.

### 4.2 한계

- ViT-B-32는 고속이지만 소형 모델이라 정밀 분류 정확도가 낮을 수 있다.
- `"a {category} game asset"` 프롬프트는 단순하다. 실제 에셋이 여러 카테고리에 걸쳐 있으면 오분류 가능.
- 픽셀 아트 에셋을 명시적으로 걸러내는 필터가 없다 (수동 검수에서 처리).

---

## 5. 수동 검수 절차

CLIP 자동 태깅 이후, 평가셋에 포함되는 이미지 중 **50장을 표본 추출**하여 수동 검수한다.

### 5.1 검수 대상

- stratified 기준: 캐릭터 25장, 사물 12장, 아이템 13장 (총 50장)
- 자동 태깅 confidence가 낮은 순으로 우선 검수

### 5.2 검수 기준

각 이미지에 대해 다음을 확인한다.

| 항목 | 기준 | 조치 |
|------|------|------|
| 카테고리 정확성 | CLIP 태그가 실제 에셋과 일치하는가 | 오분류면 수동 수정 또는 제외 |
| 도메인 일치 | 벡터/일러스트 스타일인가 | 픽셀 아트·포토리얼이면 제외 |
| 해상도 | 64×64 이상인가 | 미달이면 제외 |
| RGBA 여부 | 투명 배경이 있는가 (권장) | RGB도 허용하나 RGBA 우선 |
| 라이선스 | CC0 표시가 올바른가 | 의심스러우면 제외 |

### 5.3 검수 결과 기록

검수 완료 후 `data/raw/kenney/review_log.csv`에 기록:

```
path,original_category,corrected_category,excluded,reason
data/raw/kenney/...,character,object,false,""
data/raw/kenney/...,item,background,true,"background asset, out of domain"
```

---

## 6. 합성 페어 전략

### 6.1 페어 구성 원리

"동일 에셋의 다른 스타일 버전" 같은 자연 paired 데이터는 수집이 매우 어렵다.
본 프로젝트는 **합성 paired 데이터**를 생성한다.

- **source**: 변환할 원본 에셋 (CC0 수집 이미지)
- **reference**: 목표 스타일 예시 (같은 풀에서 다른 이미지)
- source ≠ reference (동일 파일 자기 자신과 페어링 금지)

### 6.2 전략별 비교

| 전략 | 설명 | 사용 시나리오 |
|------|------|-------------|
| `random` | 전체 풀에서 무작위 | 빠른 초기 실험 |
| `category_match` | source와 reference 같은 카테고리 | 카테고리 내 스타일 일관성 평가 |
| `stratified` (기본) | 캐릭터 50% / 사물 25% / 아이템 25% | 최종 평가셋 |

### 6.3 stratified 분포

총 30 페어 기준:

| 카테고리 | 목표 비율 | 페어 수 |
|---------|---------|--------|
| 캐릭터 | 50% | 15 |
| 사물 | 25% | 7 |
| 아이템 | 25% | 8 |
| **합계** | **100%** | **30** |

### 6.4 파일 명명 규칙

```
data/eval/
├── source_001.png       # 1번 페어의 source (RGBA)
├── reference_001.png    # 1번 페어의 reference (RGBA)
├── source_002.png
├── reference_002.png
...
├── source_030.png
├── reference_030.png
└── pairs.json           # 페어 메타데이터
```

---

## 7. 평가셋 구성 (data/eval/)

### 7.1 pairs.json 스키마

```json
{
  "n_pairs": 30,
  "strategy": "stratified",
  "seed": 42,
  "pairs": [
    {
      "id": 1,
      "source": "data/raw/kenney/.../char_001.png",
      "reference": "data/raw/kenney/.../item_007.png",
      "source_category": "character",
      "reference_category": "item",
      "category": "character",
      "source_deployed": "data/eval/source_001.png",
      "reference_deployed": "data/eval/reference_001.png"
    }
  ]
}
```

### 7.2 manifest.json 스키마 (data/raw/kenney/manifest.json)

```json
{
  "version": "1.0",
  "source": "/path/to/raw/kenney",
  "license_filter": "CC0",
  "items": [
    {
      "path": "data/raw/kenney/characters/char_001.png",
      "width": 256,
      "height": 256,
      "has_alpha": true,
      "file_size_bytes": 12345,
      "license": "CC0",
      "category": "character",
      "category_confidence": 0.87
    }
  ],
  "stats": {
    "total": 150,
    "character": 80,
    "object": 40,
    "item": 30,
    "background": 0,
    "ui": 0
  }
}
```

---

## 8. 알려진 편향

| 편향 | 설명 | 영향 |
|------|------|------|
| **픽셀 아트 비중** | Kenney.nl은 픽셀 아트 비중이 높다. CLIP 분류기가 이를 character/object/item으로 잘못 분류할 수 있다. | 수동 검수에서 픽셀 아트 적극 제외 필요 |
| **스타일 단일화** | Kenney.nl 에셋은 평면적·선명한 색상의 특정 일러스트 스타일이 다수 | 평가셋이 특정 스타일 편향 가능 → OpenGameArt 등 보완 권장 |
| **합성 페어의 한계** | source/reference가 "같은 에셋의 다른 버전"이 아닌 무관 이미지 | ground truth가 없어 정성 평가 필수 |
| **해상도 편차** | 에셋별 해상도가 32px ~ 512px까지 매우 다양 | 파이프라인이 1024 이하 제한, OOM 시 768로 리사이즈 |
| **카테고리 모호성** | "검" = item이지만 크게 렌더링하면 character처럼 보임 | CLIP confidence 낮을 때 수동 검수 우선 |

---

## 9. 데이터 수집 재실행 방법

### 9.1 Kenney.nl 에셋 수동 다운로드

1. https://kenney.nl/assets 접속
2. 카테고리별 패키지 zip 다운로드 (CC0 확인)
3. `data/raw/kenney/` 하위에 압축 해제
4. 각 패키지 디렉토리에 `LICENSE.txt` (CC0 명시) 존재 확인

### 9.2 이미지 스캔 및 태깅

```bash
# 기본 실행 (CC0 필터, 자동 태깅 포함)
python scripts/collect_data.py \
  --source data/raw/kenney \
  --out-dir data/raw/kenney \
  --manifest data/raw/kenney/manifest.json \
  --max-items 200 \
  --license-filter CC0 \
  --auto-tag \
  --clip-model ViT-B-32

# dry-run으로 사전 확인
python scripts/collect_data.py \
  --source data/raw/kenney \
  --dry-run
```

> 주의: `--auto-tag`는 CLIP 모델 로드가 필요하므로 Linux(RTX 4070)에서 실행 권장.
> macOS에서는 `--dry-run` 또는 `--auto-tag` 없이 스캔만 실행 가능.

### 9.3 평가 페어 구축

```bash
# stratified 30 페어 생성 (기본)
python scripts/build_eval_pairs.py \
  --manifest data/raw/kenney/manifest.json \
  --out-dir data/eval \
  --n-pairs 30 \
  --strategy stratified \
  --seed 42

# dry-run으로 페어 계획 확인
python scripts/build_eval_pairs.py \
  --manifest data/raw/kenney/manifest.json \
  --dry-run
```

### 9.4 재현성 보장

- `--seed 42` 고정 (변경 금지. 변경 시 이 문서에 명시)
- `manifest.json`은 git에 포함하지 않는다 (`data/` 전체 gitignore)
- 재실행 시 동일 seed → 동일 pairs.json 생성됨

---

## 10. 버전 이력

| 날짜 | 내용 |
|------|------|
| 2026-05-12 | 초안 작성. Kenney.nl CC0 기준, stratified 30 세트, CLIP ViT-B-32 태깅 |
