# AI 기반 게임 에셋 스타일 일관성 도구

> 여러 출처에서 모은 정적 2D 게임 에셋(캐릭터·사물·아이템)을 reference 이미지 기준으로 **배치 일관성을 보장**하며 스타일 통일하는 로컬 도구  
> **Technical Design Document (TDD)** — 본인 개발 참고용  
> Last updated: 2026-05-12 (차별화 narrative 재구성 + 시간 기반 일정 폐기 + 배치 일관성/속성 제어/Region masking 신설)

---

## 목차

1. [프로젝트 요약](#1-프로젝트-요약)
2. [범위 정의 (Scope)](#2-범위-정의-scope)
3. [시스템 아키텍처](#3-시스템-아키텍처)
4. [Vision Task 매핑](#4-vision-task-매핑)
5. [모듈 상세 설계](#5-모듈-상세-설계)
6. [데이터셋](#6-데이터셋)
7. [평가 설계](#7-평가-설계)
8. [개발 로드맵 (15주)](#8-개발-로드맵-15주)
9. [기술 의사결정 기록 (ADR)](#9-기술-의사결정-기록-adr)
10. [리스크와 Open Questions](#10-리스크와-open-questions)
11. [참고 자료](#11-참고-자료)

---

## 1. 프로젝트 요약

### 1.1 한 줄 정의

**인디 게임 개발자가 여러 출처에서 조달한 정적 2D 게임 에셋(캐릭터, 사물, 아이템 등)의 스타일을, reference 이미지 기준으로 자동 통일시키는 로컬 도구. 단일 이미지 변환의 품질이 아니라 N개 에셋 사이의 일관성, 형태 보존, 재현성을 보장한다.**

### 1.2 핵심 제약 (순서 중요)

1. **형태 보존**: 원본 에셋의 디자인 자체는 바꾸지 않는다 (prompt 부탁이 아닌 ControlNet + alpha mask 수학적 제약).
2. **배치 일관성**: N개 에셋이 함께 입력되면, 출력들 사이의 palette·lineart·shading 분포 편차를 최소화한다.
3. **스타일만 이식**: 팔레트, 라인, 그림자, 텍스처만 reference에 맞춘다.
4. **속성 단위 제어**: 자연어로 표현하기 어려운 정밀도 (palette only / lineart only / shading only / full).
5. **도메인 특화**: 포토리얼이 아닌 벡터/일러스트 스타일의 정적 2D 에셋 대상.
6. **게임 엔진 친화**: 투명 배경(알파 채널) 유지.
7. **재현성**: 같은 config + seed → bit-exact 재생성.

### 1.3 차별화 narrative (상용 모델 대비 포지셔닝)

GPT Image 2.0 등 범용 이미지 생성 모델이 등장한 환경에서, 본 도구의 차별점은 **raw 단일 이미지 품질이 아니다.** 상용 모델이 구조적으로 풀지 않는 세 영역에서 정량적 우위를 만든다.

| 차원 | GPT Image 2.0 / 범용 img2img | Style Unifier |
| --- | --- | --- |
| N개 에셋의 일관성 | 각 이미지를 독립 생성 → palette·linewidth·shading 흔들림 | reference의 통계를 추출해 모든 출력에 강제 |
| 형태 보존 | 프롬프트 "디자인 바꾸지 마" — 권유 | ControlNet + alpha mask로 수학적 제약 |
| 속성 단위 제어 | 자연어 — 거친 제어 | palette/lineart/shading 토글 + 강도 슬라이더 |
| 재현성 | 같은 입력에도 매번 다른 출력 | seed + config 고정으로 bit-exact 재생성 |
| 프라이버시·비용 | 에셋을 외부 서버로 업로드, 장당 과금 | 로컬 GPU, 무제한 재실험 |

본 도구는 GPT Image 2.0을 **경쟁 대상이 아니라 평가 baseline으로 통합**한다 (Section 7.2 ablation의 `A_GPT_Image` 열). 핵심 narrative는 "품질은 비슷하거나 약간 낮지만, 일관성·보존성·재현성은 우위" 다.

### 1.4 성공 기준

본 프로젝트는 시간 기반 일정 대신 **의존 순서 기반 게이트**로 진행한다 (Section 8 참조). 종료 시점의 성공 기준은:

**정량 (vs GPT Image 2.0 baseline)**:
- [ ] **σ_palette** (배치 내 표준편차): GPT Image 대비 −50% 이상
- [ ] **σ_linewidth** (배치 내 표준편차): GPT Image 대비 −50% 이상
- [ ] **DINOv2 Identity** (원본 대비): GPT Image 대비 +15% 이상
- [ ] **LPIPS Structure** (원본 대비): GPT Image 대비 −30% 이상
- [ ] **CLIP Style Similarity**: 상용 모델과 ±5% 이내 (품질 동급 narrative)
- [ ] **재현성**: 동일 config·seed → bit-exact 재생성 (상용 모델은 구조상 불가)

**정성**:
- [ ] 본인 개발 게임에 실제 적용한 before/after 샘플 10–20세트
- [ ] 인디 개발자 5–10명 대상 사용자 스터디 결과 확보 (head-to-head with GPT Image 포함)
- [ ] GitHub 공개 가능한 코드베이스 + README + 자동 leaderboard + 데모

### 1.5 CV 프로젝트로서의 정체성

본 프로젝트는 **이미지 입력 → 이미지 출력**의 vision pipeline으로, 전통적 CV 주제(Semantic Segmentation, Edge Detection, Color Clustering, Alpha Matting)와 현대 CV 주제(Visual Representation Learning, Conditional Diffusion Generation, Spatial Conditioning)를 통합한다. 평가 단계에서는 Perceptual IQA의 다양한 메트릭(LPIPS, CLIP, DINOv2)에 더해, **배치 일관성을 측정하는 자체 메트릭(σ_palette, σ_linewidth, σ_shading)** 을 제안한다. 상세 매핑은 [Section 4](#4-vision-task-매핑) 참조.

---

## 2. 범위 정의 (Scope)

### 2.1 In Scope (반드시 한다)

| 항목      | 내용                                                                       |
| --------- | -------------------------------------------------------------------------- |
| 입력      | 정적 2D 단일 객체 이미지 (캐릭터, 사물, 아이템, 무기, 도구 등) (PNG, RGBA) |
| Reference | 단일 이미지 1장 (필수), 다중 이미지 (선택)                                 |
| 출력      | 스타일 변환된 이미지 (PNG, RGBA)                                           |
| UI        | Gradio 웹 인터페이스 + Python CLI                                          |
| 평가      | 자동화된 평가 파이프라인 (4개 이상 메트릭)                                 |

대상 도메인은 **벡터/일러스트 스타일의 정적 2D 객체**다. 캐릭터(사람, 동물, 몬스터), 사물(상자, 통, 가구), 아이템(무기, 도구, 장비, 음식, 포션) 등이 모두 포함된다. 현재 파이프라인에는 카테고리별 분기 로직이 없으며, 동일한 RMBG → lineart → CLIP → SDXL+ControlNet+IP-Adapter → alpha 복원 → 팔레트 정제 흐름이 모든 카테고리에 작동한다.

### 2.2 Out of Scope (한 학기 내 안 한다)

도메인이 다르거나 추가 기술이 필요해 본 프로젝트 범위에서 제외하는 항목:

| 항목                    | 사유                                                                               |
| ----------------------- | ---------------------------------------------------------------------------------- |
| **픽셀 아트**           | Retro Diffusion, PixelLab 등 특화 도구 영역. 그리드 정합·색 제한 등 별개 처리 필요 |
| **UI/버튼**             | 텍스트 보존 (OCR-aware) 추가 처리가 별도 필요                                      |
| **배경/타일맵**         | 반복 텍스처 일관성, seamless 보장 등 별개 메트릭과 처리 필요                       |
| **애니메이션/이펙트**   | 시간축 일관성, 반투명 처리 등 별개 연구 주제                                       |
| **Unity 플러그인**      | C# 통합에 추가 3-4주 필요                                                          |
| **커스텀 LoRA 학습**    | 데이터 준비 + 학습 + 튜닝에 3-4주 필요. 공개 체크포인트로 대체                     |
| **SaaS 배포**           | 인프라 구축 + 운영은 별도 과제                                                     |
| **다중 객체 동시 변환** | 단일 객체만                                                                        |

### 2.3 Baseline으로 승격된 항목 (이전 Stretch였던 것)

Claude Code로 구현 시간이 절약되므로, 이전 plan에서 stretch였던 다음 항목들은 **baseline 기능**으로 승격한다:

- **배치 일관성 모듈** (`src/encoding/batch_consistency.py`) — N개 에셋 동시 입력 + 통계 강제. 차별화의 핵심.
- **속성 분리 제어** (`src/postprocessing/attribute_control.py`) — palette/lineart/shading 토글 + 강도.
- **Region masking** (`src/postprocessing/region_mask.py`) — 마스크 영역만 변환.
- **Multi-reference 가중치 블렌딩** (`src/encoding/multi_ref.py`) — uniform mean + CLIP softmax weighted.
- **GPT Image 2.0 baseline 통합** (`src/baselines/gpt_image.py`) — 평가 자동화에 외부 baseline 포함.

### 2.4 Stretch Goals (여전히 야심 옵션)

GPU 시간·우선순위에 따라 선택적 진행:

- **Test-time scale 자동 튜닝**: 변환 후 LPIPS/DINOv2 측정 → scale 조정 → 재생성 loop.
- **Iterative refinement loop**: 약한 속성 식별 → 해당 속성만 재생성 (self-reviewer).
- **개인 LoRA 파인튜닝**: 사용자 본인 에셋 20–50장으로 경량 LoRA 학습. ADR-002를 부분적으로 완화 (Section 9 ADR-002 참조).
- **Style Bible 산출물**: Reference → 통계 JSON으로 저장. 재사용 가능한 아티스트 친화 산출물.
- **자동 leaderboard**: ablation + 외부 baseline 메트릭을 README에 자동 갱신.

---

## 3. 시스템 아키텍처

### 3.1 전체 파이프라인

```
┌─────────────────┐
│  사용자 입력     │
│ - 원본 에셋(N장) │
│ - Reference     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐      ┌─────────────────┐
│  Module 1:      │      │  Module 2:      │
│  Preprocessing  │      │  Style Encoding │
│  - BG 분리      │      │  - IP-Adapter   │
│  - Lineart 추출  │      │  - CLIP 임베딩   │
│  - 팔레트 추출   │      └────────┬────────┘
└────────┬────────┘               │
         │                        │
         └──────────┬─────────────┘
                    ▼
         ┌──────────────────────┐
         │  Module 3:           │
         │  Conditional Gen     │
         │  - SDXL + ControlNet │
         │  - 일러스트 체크포인트 │
         └──────────┬───────────┘
                    ▼
         ┌──────────────────────┐
         │  Module 4:           │
         │  Postprocessing      │
         │  - 알파 복원          │
         │  - 팔레트 정제         │
         │  - 품질 검증          │
         └──────────┬───────────┘
                    ▼
         ┌──────────────────────┐
         │  출력: 변환된 에셋     │
         │  + 품질 메트릭         │
         └──────────────────────┘
```

### 3.2 디렉토리 구조

```
style-unifier/
├── README.md
├── requirements.txt
├── pyproject.toml
├── configs/
│   ├── default.yaml          # 기본 설정
│   └── experiments/          # 실험별 설정
│       ├── baseline.yaml
│       ├── with_lineart.yaml
│       └── multi_ref.yaml
├── src/
│   ├── preprocessing/
│   │   ├── bg_removal.py
│   │   ├── lineart.py
│   │   └── palette.py
│   ├── encoding/
│   │   ├── ip_adapter_wrapper.py
│   │   ├── multi_ref.py
│   │   ├── batch_consistency.py    # §5.5.1 후처리 일관성 baseline
│   │   └── shared_attention.py     # §5.5.2 StyleAligned 응용 (그룹 D5, 시도 중)
│   ├── generation/
│   │   ├── pipeline.py       # 메인 생성 파이프라인 (BaselineWrapper 상속)
│   │   └── controlnet.py
│   ├── postprocessing/
│   │   ├── alpha_restore.py
│   │   ├── palette_quantize.py
│   │   ├── attribute_control.py    # §5.6 속성 분리 제어
│   │   ├── region_mask.py          # §5.7 영역 제한 변환
│   │   └── quality_check.py
│   ├── evaluation/
│   │   ├── metrics.py        # CLIP, LPIPS, DINOv2, palette EMD (§7.1)
│   │   ├── consistency.py    # σ_palette, σ_linewidth, σ_shading (§7.2)
│   │   └── run_eval.py
│   ├── baselines/                  # §5.8 외부 baseline (OOP - ABC + 다형성)
│   │   ├── base.py                 # BaselineWrapper ABC (§5.8.0)
│   │   ├── gpt_image.py            # GPTImageBaseline (§5.8.1, ADR-008)
│   │   └── ipadapter_naive.py      # IPAdapterNaiveBaseline (§5.8.2)
│   └── utils/
│       ├── io.py
│       ├── logging.py
│       ├── device.py
│       ├── repro.py                # seed 고정 헬퍼
│       └── experiment.py           # experiments/NNN/ 자동 디렉토리 + 메타 기록
├── scripts/
│   ├── download_models.py
│   ├── collect_data.py       # CC0 에셋 수집
│   └── run_batch.py
├── app/
│   └── gradio_app.py
├── data/
│   ├── raw/                  # 수집 원본
│   ├── processed/            # 전처리 후
│   └── eval/                 # 평가셋
├── experiments/
│   ├── 001_baseline/
│   ├── 002_ipadapter_only/
│   └── ...
└── tests/
    ├── test_preprocessing.py
    ├── test_pipeline.py
    └── fixtures/
```

### 3.3 설정 파일 예시

```yaml
# configs/default.yaml

model:
  base_checkpoint: "cagliostrolab/animagine-xl-3.1"
  vae: "madebyollin/sdxl-vae-fp16-fix"

  ip_adapter:
    repo: "h94/IP-Adapter"
    subfolder: "sdxl_models"
    weight_name: "ip-adapter-plus_sdxl_vit-h.safetensors"
    scale: 0.7

  controlnet:
    - type: "lineart"
      repo: "diffusers/controlnet-canny-sdxl-1.0" # canny로 임시 대체 가능
      scale: 0.9
    - type: "depth"
      repo: "diffusers/controlnet-depth-sdxl-1.0"
      scale: 0.5
      enabled: false

sampling:
  scheduler: "DPMSolverMultistepScheduler"
  scheduler_config:
    use_karras_sigmas: true
    algorithm_type: "sde-dpmsolver++"
  steps: 25
  cfg_scale: 6.0
  denoising_strength: 0.55
  seed: null # null이면 랜덤

preprocessing:
  bg_removal_model: "briaai/RMBG-1.4"
  target_size: 1024
  preserve_aspect_ratio: true

postprocessing:
  palette_quantize: true
  palette_k: 12
  alpha_feather: 1 # px

evaluation:
  metrics:
    - clip_style_similarity
    - lpips_structure
    - palette_distance
    - dino_identity
```

---

## 4. Vision Task 매핑

이 프로젝트는 단일 CV 알고리즘이 아니라 **여러 vision task를 통합하는 시스템**이다. 각 모듈이 어떤 전통적/현대적 vision task에 해당하는지 명시적으로 정리한다. 프로젝트의 CV적 정체성을 드러내고, 평가자(교수 등)가 vision pipeline으로서의 구성을 즉시 파악할 수 있도록 하기 위함이다.

### 4.1 전체 매핑 테이블

추론 파이프라인을 구성하는 9개 vision task와, 별도로 평가 단계의 IQA를 정리한다.

| 단계 | 모듈 위치               | Vision Task 분류                                              | 사용 모델/기법                     |
| ---- | ----------------------- | ------------------------------------------------------------- | ---------------------------------- |
| 1    | 5.1.1 배경 분리         | **Semantic Segmentation** (Binary, FG-BG) / **Image Matting** | RMBG-1.4, BiRefNet                 |
| 2    | 5.1.2 Lineart 추출      | **Edge Detection**                                            | Canny, HED, PiDiNet, lineart_anime |
| 3    | 5.1.3 팔레트 추출       | **Color Quantization / Clustering**                           | K-means in RGB                     |
| 4    | 5.2.1 IP-Adapter 인코딩 | **Visual Feature Extraction** (Image Encoder)                 | CLIP ViT-H/14                      |
| 5    | 5.2.2 IP-Adapter 주입   | **Cross-Attention Conditioning**                              | IP-Adapter projection layer        |
| 6    | 5.3 ControlNet 주입     | **Spatial Conditioning**                                      | ControlNet (lineart, depth)        |
| 7    | 5.3 생성 파이프라인     | **Conditional Diffusion**                                     | SDXL Diffusion                     |
| 8    | 5.4.1 알파 복원         | **Alpha Channel Restoration**                                 | Feathering, Gaussian blur          |
| 9    | 5.4.2 팔레트 양자화     | **Color Transfer / Palette Mapping**                          | Nearest-neighbor matching          |

**평가 단계 (추론 외)**:

| 단계 | 모듈 위치       | Vision Task 분류                              | 사용 메트릭                           |
| ---- | --------------- | --------------------------------------------- | ------------------------------------- |
| E1   | 7.1 정량 메트릭 | **Perceptual Image Quality Assessment (IQA)** | LPIPS, CLIP sim, DINOv2, Palette EMD  |
| E2   | 7.2 일관성 메트릭 | **Batch Consistency Measurement** (제안)    | σ_palette, σ_linewidth, σ_shading     |
| E3   | 7.3 외부 baseline | **Head-to-head comparison**                | GPT Image 2.0 / IP-Adapter naive 출력 비교 |

평가 단계는 추론 파이프라인의 일부가 아니라 개발·검증 단계에서만 사용되는 별도 phase다. 그래서 위 9개와 분리해 표기한다. E2의 **σ_consistency 계열**은 본 프로젝트가 제안하는 자체 메트릭으로, "N개 출력이 한 가족인가"를 정량화한다.

### 4.1.1 매핑 결정의 근거

처음 검토 시점에는 일부 task를 더 일반적 명칭으로 표기했지만(Image-to-Image Translation, Feature Aggregation 등), 이를 다음과 같이 정밀화했다.

- **Cross-Attention Conditioning (단계 5)**: IP-Adapter의 본질은 CLIP 임베딩을 SDXL의 cross-attention layer에 주입하는 것이다. "Feature Aggregation"은 multi-reference에서나 의미 있고 single-reference baseline에서는 일어나지 않는다.
- **Spatial Conditioning (단계 6)** vs **Conditional Diffusion (단계 7)**: ControlNet과 SDXL이 서로 다른 task로 명확히 분리된다. ControlNet은 공간적(어디에 무엇이 있어야 하는가) 제어, SDXL은 denoising 자체.
- **Alpha Channel Restoration (단계 8)**: "Alpha Compositing"은 Porter-Duff 합성 연산을 가리키는 용어인데, 이 프로젝트는 합성이 아니라 원본 알파를 복원하는 작업이다.
- **Perceptual IQA**: 평가 단계로 분리. 후처리(이미지를 가공하는 단계)와 평가(품질을 측정하는 단계)는 별개 phase이며, 추론 시에는 평가가 일어나지 않는다.

### 4.2 Vision Task 관점 파이프라인

시스템 전체를 vision task 용어로 다시 그리면:

```
         원본 에셋 (RGBA Image)
                ↓
    [Semantic Segmentation]       ← 배경/전경 분리
                ↓
         Foreground Mask (α)
                ↓
    [Edge Detection]              ← 구조 조건 신호
                ↓
         Lineart Map
                ↓
    [Color Clustering]            ← 팔레트 분석
                ↓
         Palette (k colors)

   Reference Image (RGB)
                ↓
    [Visual Feature Extraction]   ← CLIP encoder
                ↓
         Style Embedding
                ↓
    [Cross-Attention Conditioning] ← IP-Adapter projection
                ↓
         Injected Style Tokens

                ↓ (모든 신호 통합)
    ┌──────────────────────────────────────────┐
    │  생성 단계                                │
    │   - [Spatial Conditioning]                │
    │       → ControlNet (lineart guide)        │
    │   - [Conditional Diffusion]               │
    │       → SDXL denoising loop               │
    └──────────────────────────────────────────┘
                ↓
         Generated RGB Image
                ↓
    [Alpha Channel Restoration]   ← 원본 mask 복원
                ↓
    [Color Transfer]              ← 팔레트 양자화
                ↓
         Output (RGBA Image)

         === 추론 파이프라인 종료 ===

         === 평가 단계 (개발 시점만) ===
         [Perceptual IQA]          ← LPIPS / CLIP / DINOv2 / Palette
```

### 4.3 세부 설명

#### 4.3.1 전처리 단계 (전통적 CV 기법)

**Semantic Segmentation (배경 분리)**

RMBG-1.4와 BiRefNet은 **pixel-wise binary classification** (foreground vs background)을 수행한다. 엄밀히는 hard segmentation이 아닌 soft alpha(연속 확률)를 출력하므로 **image matting**에 더 가깝다. Matting은 segmentation의 일반화된 형태로, 반투명 영역(머리카락, 경계 안티에일리어싱)까지 다룰 수 있다.

- **CV 교과서 위치**: Image Segmentation, Alpha Matting
- **대표 논문**: U-Net (Ronneberger 2015), DeepLab 계열, MODNet (matting), BiRefNet (2024)

**Edge Detection (Lineart 추출)**

고전 CV의 핵심 주제 중 하나. 본 프로젝트는 고전(Canny, 1986)부터 현대(PiDiNet, lineart_anime, 2021+)까지의 스펙트럼을 모두 다룬다.

- **Canny**: Gradient 계산 → Non-maximum suppression → Double threshold → Hysteresis
- **HED (Holistically-Nested Edge Detection)**: VGG backbone, multi-scale supervision
- **PiDiNet**: Pixel difference convolution 기반 경량 edge detection
- **lineart_anime**: 일러스트 도메인 파인튜닝 버전

- **CV 교과서 위치**: Low-level Vision, Feature Detection
- **대표 논문**: Canny (1986), HED (Xie & Tu 2015), PiDiNet (Su et al. 2021)

**Color Clustering (팔레트 추출)**

엄격히는 unsupervised learning 기법이지만 CV의 **color space analysis**로 분류된다. RGB 공간에서 k-means를 수행한다.

- **CV 교과서 위치**: Color Vision, Image Representation
- **대표 기법**: K-means, Mean-shift (color segmentation), Median cut

#### 4.3.2 인코딩 단계 (Representation Learning + Conditioning)

**Visual Feature Extraction (CLIP image encoder)**

CLIP의 vision transformer가 reference 이미지를 고차원 embedding으로 변환. Pretrained vision model을 **feature extractor**로 활용하는 전형적 패턴.

- **CV 교과서 위치**: Visual Representation Learning, Vision Transformers
- **대표 논문**: ViT (Dosovitskiy et al. 2020), CLIP (Radford et al. 2021)

**Cross-Attention Conditioning (IP-Adapter projection)**

CLIP에서 추출한 임베딩을 SDXL의 cross-attention layer가 받을 수 있는 형태로 변환·주입. 텍스트 임베딩이 SDXL에 주입되는 것과 동일한 메커니즘으로 이미지 임베딩을 주입하는 것이 IP-Adapter의 핵심 기여다.

- **CV 교과서 위치**: Conditional Generation, Cross-Modal Conditioning
- **대표 논문**: IP-Adapter (Ye et al. 2023), Stable Diffusion (Rombach et al. 2022)

#### 4.3.3 생성 단계 (Modern Generative Vision)

본 프로젝트의 **핵심 CV 단계**. 두 가지 conditioning 메커니즘이 한 diffusion 생성 과정에 결합된다.

**Spatial Conditioning (ControlNet)**

ControlNet은 **공간적 조건 신호**(lineart, depth, pose 등)를 diffusion 모델에 주입해 생성 결과의 구조를 제어한다. 본 프로젝트에서는 lineart를 주 조건으로, depth를 보조 조건으로 사용해 원본 형태를 강하게 보존한다.

- **CV 교과서 위치**: Conditional Generation, Geometric Reasoning
- **대표 논문**: ControlNet (Zhang et al. 2023), T2I-Adapter (Mou et al. 2023)

**Conditional Diffusion (SDXL)**

SDXL의 denoising loop. text + image(IP-Adapter) + spatial(ControlNet) 다중 조건을 동시에 받아 latent space에서 점진적으로 노이즈를 제거하며 이미지를 생성한다. Diffusion model 기반 conditional generation은 현대 CV 학계의 주류 흐름.

- **CV 교과서 위치**: Generative Models, Conditional Image Generation
- **대표 논문**: DDPM (Ho et al. 2020), Latent Diffusion (Rombach et al. 2022), SDXL (Podell et al. 2023)

#### 4.3.4 후처리 단계 (Image Processing)

**Alpha Channel Restoration**

생성된 RGB 결과에 전처리에서 추출한 원본 alpha mask를 다시 적용해 RGBA로 복원. 경계 영역은 Gaussian blur로 부드럽게 feathering. Compositing 연산이 아니라 알파 채널의 재부착에 가깝다.

- **CV 교과서 위치**: Image Processing, Alpha Matting
- **관련 기법**: Gaussian filtering, Laplacian pyramid blending

**Color Transfer**

Reference에서 추출한 k-색 팔레트 방향으로 결과 이미지의 색상을 부드럽게 양자화. Reinhard et al. (2001)의 LAB 색공간 기반 기법부터 neural color transfer까지 이어지는 세부 주제.

- **CV 교과서 위치**: Color Vision, Image Enhancement
- **대표 논문**: Reinhard et al. (2001), Neural Color Transfer (He et al. 2019)

#### 4.3.5 평가 단계 (Perceptual IQA, 추론 외)

**Perceptual Image Quality Assessment**

생성 결과의 품질을 정량적으로 평가. 이 단계는 **추론 파이프라인의 일부가 아니다**. 개발 중 ablation 실험과 사용자 스터디에서만 사용된다. Perceptual IQA는 인간의 지각적 판단을 모방하는 메트릭들을 가리키는 분야 명칭으로, 본 프로젝트의 4가지 메트릭이 모두 여기에 속한다.

- **LPIPS**: VGG 기반 perceptual distance
- **CLIP Style Similarity**: CLIP 임베딩 cosine
- **DINOv2 Identity**: self-supervised ViT feature similarity
- **Palette Distance**: 색 분포 간 EMD

- **CV 교과서 위치**: Image Quality Assessment, Perceptual Metrics
- **대표 논문**: SSIM (Wang et al. 2004), LPIPS (Zhang et al. 2018), FID (Heusel et al. 2017), DINOv2 (Oquab et al. 2023)

### 4.4 본 프로젝트의 CV적 정체성 요약

본 프로젝트가 다루는 vision task의 **폭**:

**전통적 CV (4개 영역)**: Semantic Segmentation, Edge Detection, Color Clustering, Alpha Channel Restoration

**현대 CV (4개 영역)**: Visual Representation Learning, Cross-Attention Conditioning, Spatial Conditioning, Conditional Diffusion

**세부 전문 분야 (1개 영역)**: Color Transfer

**평가 단계 (별도 phase)**: Perceptual IQA (4가지 메트릭으로 구현)

**Input/Output modality (평가 기준 핵심)**:

- Input: Image (multiple)
- Output: Image (RGBA) + optional text reports (품질 검증 결과)
- **"이미지 입력 → 이미지 출력" 평가 기준을 정면으로 충족**

**학부 프로젝트로서의 포지셔닝**: 각 vision task를 from-scratch로 구현하지는 않지만, **여러 vision task를 통합하고 도메인(정적 2D 게임 에셋)에 최적화하는 system-level CV 연구**로 분류된다. 이는 현대 CV 학계의 주류 흐름(foundation model 활용 + 응용)과 일치한다.

---

## 5. 모듈 상세 설계

### 5.1 Module 1: Preprocessing

#### 5.1.1 배경 분리 (bg*removal.py) — \_Semantic Segmentation / Image Matting*

**목적**: 게임 에셋의 투명 배경을 보존하고, 생성 과정에서 배경이 오염되지 않도록 분리.

**인터페이스**:

```python
from PIL import Image
from typing import Tuple
import numpy as np

def remove_background(
    image: Image.Image,
    model_name: str = "briaai/RMBG-1.4",
) -> Tuple[Image.Image, np.ndarray]:
    """
    입력 이미지에서 배경을 분리한다.

    Args:
        image: RGB 또는 RGBA PIL Image.
        model_name: HuggingFace 모델 식별자.

    Returns:
        foreground: RGBA 이미지. 배경 영역은 alpha=0.
        mask: (H, W) float32 in [0, 1]. 전경 확률.
    """
    ...
```

**구현 노트**:

- 입력이 이미 투명 배경이면 (alpha 채널 존재) 그 mask를 그대로 사용.
- 불투명 배경만 있는 에셋은 RMBG-1.4로 자동 분리.
- 품질 부족 시 BiRefNet으로 업그레이드 가능 (단, 추론 속도 느림).

#### 5.1.2 Lineart 추출 (lineart.py) — _Edge Detection_

**목적**: ControlNet에 입력할 구조 조건 신호 생성.

```python
def extract_lineart(
    image: Image.Image,
    detector: str = "lineart_anime",  # or "lineart_realistic", "canny"
    threshold_low: int = 100,
    threshold_high: int = 200,
) -> Image.Image:
    """
    벡터/일러스트 스타일에 적합한 라인 맵을 추출한다.

    벡터 스타일은 라인이 명확하므로 lineart_anime detector가
    일반 canny보다 결과 품질이 높다. 검증 필요.
    """
    ...
```

**대안 비교**:

| Detector          | 장점            | 단점                          |
| ----------------- | --------------- | ----------------------------- |
| canny             | 가볍고 빠름     | 텍스처 디테일까지 엣지로 잡음 |
| lineart_anime     | 일러스트에 최적 | 추가 모델 로드 필요           |
| lineart_realistic | 선명한 라인     | 일러스트엔 과도함             |
| softedge (HED)    | 자연스러운 경계 | 너무 부드러워 형태 모호       |

**초기 선택**: `lineart_anime`. 대체 시 `canny`로 fallback.

#### 5.1.3 팔레트 추출 (palette.py) — _Color Clustering_

```python
from sklearn.cluster import KMeans

def extract_palette(
    image: Image.Image,
    k: int = 12,
    ignore_alpha: bool = True,
) -> np.ndarray:
    """
    이미지의 주요 색상을 k-means로 추출한다.

    Returns:
        (k, 3) uint8 array. RGB.
    """
    arr = np.array(image.convert("RGBA"))
    if ignore_alpha:
        mask = arr[..., 3] > 128
        pixels = arr[mask][..., :3]
    else:
        pixels = arr.reshape(-1, 4)[..., :3]

    if len(pixels) < k:
        return np.array([[0, 0, 0]] * k, dtype=np.uint8)

    kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
    kmeans.fit(pixels)
    return kmeans.cluster_centers_.astype(np.uint8)
```

### 5.2 Module 2: Style Encoding

#### 5.2.1 IP-Adapter Wrapper (ip*adapter_wrapper.py) — \_Visual Feature Extraction + Cross-Attention Conditioning*

**목적**: Reference 이미지에서 스타일 embedding을 추출해 생성 파이프라인에 주입.

```python
from diffusers import StableDiffusionXLPipeline
from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor

class StyleEncoder:
    def __init__(
        self,
        ip_adapter_repo: str = "h94/IP-Adapter",
        subfolder: str = "sdxl_models",
        weight_name: str = "ip-adapter-plus_sdxl_vit-h.safetensors",
        device: str = "cuda",
    ):
        self.image_processor = CLIPImageProcessor()
        self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            "h94/IP-Adapter",
            subfolder="models/image_encoder",
        ).to(device)
        # IP-Adapter weights는 pipeline에 load_ip_adapter()로 적용

    def encode_single(self, reference: Image.Image) -> torch.Tensor:
        """단일 reference → (1, seq_len, dim) embedding."""
        ...

    def encode_multi(
        self,
        references: List[Image.Image],
        strategy: str = "mean",  # "mean" | "weighted" | "attention"
    ) -> torch.Tensor:
        """
        다중 reference → 통합된 embedding.

        strategy:
          - "mean": 단순 평균. 베이스라인.
          - "weighted": 각 reference와 타겟의 CLIP 유사도로 가중 평균.
          - "attention": self-attention으로 통합 (실험적).
        """
        ...
```

#### 5.2.2 Multi-Reference 전략 (multi_ref.py) — *Feature Aggregation* (그룹 D4)

> **참고**: Single-reference baseline에서는 이 단계가 동작하지 않는다. Multi-reference 사용 시에만 의미가 있는 task다. Section 4.1의 Vision Task 매핑에서 Single-reference 기준 9개 task를 카운트할 때 이 항목은 제외된다.

이전 plan에서는 stretch였으나, Claude Code 도입으로 **그룹 D의 baseline 기능**으로 승격되었다. `mean`과 `weighted` 두 전략 모두 구현한다.

```python
def weighted_multi_ref(
    references: List[torch.Tensor],  # 각각 (seq, dim)
    target: Optional[torch.Tensor] = None,  # 변환 대상의 CLIP embedding
) -> torch.Tensor:
    """
    Target과 각 reference의 CLIP 유사도로 가중 평균.
    Target 없으면 uniform mean과 동일.
    """
    if target is None or len(references) == 1:
        return torch.stack(references).mean(dim=0)

    sims = torch.stack([
        cosine_similarity(ref.mean(0), target.mean(0), dim=0)
        for ref in references
    ])
    weights = torch.softmax(sims / 0.1, dim=0)  # temperature 0.1

    weighted = sum(w * ref for w, ref in zip(weights, references))
    return weighted
```

### 5.3 Module 3: Conditional Generation

#### 5.3.1 메인 파이프라인 (pipeline.py) — _Spatial Conditioning + Conditional Diffusion_

```python
from diffusers import StableDiffusionXLControlNetImg2ImgPipeline, ControlNetModel
from typing import List, Optional
import torch

class StyleUnificationPipeline:
    def __init__(self, config: dict):
        self.config = config

        # 1. ControlNet 로드
        controlnets = [
            ControlNetModel.from_pretrained(c["repo"], torch_dtype=torch.float16)
            for c in config["model"]["controlnet"] if c.get("enabled", True)
        ]

        # 2. 베이스 파이프라인
        self.pipe = StableDiffusionXLControlNetImg2ImgPipeline.from_pretrained(
            config["model"]["base_checkpoint"],
            controlnet=controlnets,
            torch_dtype=torch.float16,
        ).to("cuda")

        # 3. IP-Adapter 로드
        self.pipe.load_ip_adapter(
            config["model"]["ip_adapter"]["repo"],
            subfolder=config["model"]["ip_adapter"]["subfolder"],
            weight_name=config["model"]["ip_adapter"]["weight_name"],
        )
        self.pipe.set_ip_adapter_scale(config["model"]["ip_adapter"]["scale"])

        # 4. Scheduler
        self._setup_scheduler()

    def transform(
        self,
        source: Image.Image,
        reference: Image.Image,
        control_images: List[Image.Image],  # [lineart, depth, ...]
        prompt: str = "high quality game asset, clean vector illustration",
        negative_prompt: str = "blurry, realistic, photo, 3d render",
        **kwargs,
    ) -> Image.Image:
        """
        단일 에셋 변환.
        """
        result = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=source,
            control_image=control_images,
            ip_adapter_image=reference,
            num_inference_steps=self.config["sampling"]["steps"],
            guidance_scale=self.config["sampling"]["cfg_scale"],
            strength=self.config["sampling"]["denoising_strength"],
            controlnet_conditioning_scale=[
                c["scale"] for c in self.config["model"]["controlnet"]
                if c.get("enabled", True)
            ],
        ).images[0]
        return result
```

#### 5.3.2 하이퍼파라미터 탐색 범위

| 파라미터                     | 탐색 범위 | 초기값 | 효과                   |
| ---------------------------- | --------- | ------ | ---------------------- |
| `ip_adapter_scale`           | 0.4 - 1.0 | 0.7    | 스타일 충실도 (↑ 강함) |
| `controlnet_scale (lineart)` | 0.6 - 1.2 | 0.9    | 형태 보존 (↑ 강함)     |
| `denoising_strength`         | 0.3 - 0.7 | 0.55   | 원본 보존 (↓ 강함)     |
| `cfg_scale`                  | 4.0 - 8.0 | 6.0    | 프롬프트 따름 (↑ 강함) |
| `steps`                      | 20 - 40   | 25     | 품질 vs 속도           |

**탐색 전략**: 초기값으로 20장 생성 → 시각 검사 → 문제 차원만 grid search.

### 5.4 Module 4: Postprocessing

#### 5.4.1 알파 복원 (alpha*restore.py) — \_Alpha Channel Restoration*

```python
def restore_alpha(
    generated: Image.Image,  # RGB
    original_mask: np.ndarray,  # (H, W), float32 in [0, 1]
    feather_px: int = 1,
) -> Image.Image:
    """
    생성 결과(RGB)에 원본 mask를 적용해 RGBA로 복원.
    경계는 약간 feather로 부드럽게.
    """
    gen_arr = np.array(generated.convert("RGB"))

    # Feathering
    from scipy.ndimage import gaussian_filter
    mask_feathered = gaussian_filter(original_mask, sigma=feather_px)

    rgba = np.concatenate([
        gen_arr,
        (mask_feathered * 255).astype(np.uint8)[..., None]
    ], axis=-1)

    return Image.fromarray(rgba, mode="RGBA")
```

#### 5.4.2 팔레트 정제 (palette*quantize.py) — \_Color Transfer*

```python
def quantize_to_palette(
    image: Image.Image,
    palette: np.ndarray,  # (k, 3) from reference
    strength: float = 0.7,  # 0 = 원본 유지, 1 = 완전 양자화
) -> Image.Image:
    """
    생성된 에셋의 색상을 reference 팔레트 방향으로 밀어낸다.
    완전 양자화는 품질 저하가 크므로 soft-snap 방식.
    """
    arr = np.array(image.convert("RGBA"))
    rgb = arr[..., :3].astype(np.float32)

    # 각 픽셀에서 가장 가까운 팔레트 색
    distances = np.linalg.norm(
        rgb[..., None, :] - palette[None, None, :, :],
        axis=-1
    )
    nearest_idx = distances.argmin(axis=-1)
    nearest_color = palette[nearest_idx].astype(np.float32)

    # Soft-snap: 원본과 nearest 간 보간
    blended = rgb * (1 - strength) + nearest_color * strength

    arr[..., :3] = np.clip(blended, 0, 255).astype(np.uint8)
    return Image.fromarray(arr, mode="RGBA")
```

#### 5.4.3 품질 검증 (quality*check.py) — \_Perceptual IQA (개발 시점만)*

**참고**: 이 모듈은 추론 파이프라인의 일부가 아니라 **개발·검증 시점**에서만 사용되는 평가 도구다. 실제 사용자가 한 장 변환할 때는 호출되지 않는다. 자세한 평가 설계는 [Section 7](#7-평가-설계) 참조.

```python
@dataclass
class QualityReport:
    clip_style_sim: float
    lpips_structure: float
    identity_cos: float
    palette_distance: float
    passed: bool
    warnings: List[str]

def verify_quality(
    source: Image.Image,
    reference: Image.Image,
    result: Image.Image,
    thresholds: dict,
) -> QualityReport:
    """
    변환 결과의 품질을 자동 검증.
    임계값 미달 시 재생성 권장 플래그.
    """
    ...
```

### 5.5 Module 5: Batch Consistency (시도 중인 architectural approach)

> ⚠️ **상태**: 이 모듈은 **실험 검증 단계의 architectural approach**다. 후처리 hack(§5.5.1)으로 안전한 baseline을 확보한 다음, **mechanism level 시도**(§5.5.2 StyleAligned 응용)를 추가한다. 실제 효과는 그룹 F1 ablation(A4 vs A5)에서 측정된 뒤 narrative에 반영. 자세한 결정 근거는 ADR-010 참조.

#### 5.5.1 후처리 기반 통계 강제 (batch_consistency.py) — *Cross-Asset Statistics Enforcement* (안전한 baseline)

**목적**: N개 source를 함께 변환한 결과에 reference 기반 palette·linewidth·shading 통계를 일괄 적용한다. 후처리에서 동작하므로 **구현·디버깅이 쉽고, 항상 동작이 보장**되는 안전한 baseline.

**한계 (정직한 명시)**: 이 방식은 GPT Image 2.0 + 외부 후처리 스크립트로도 원리상 가능하다. 즉 이것만으로는 "API-only 모델이 못 하는 것"을 주장할 수 없다. mechanism level 차별화는 §5.5.2에서 시도.

```python
from dataclasses import dataclass
import numpy as np
import torch
from PIL import Image


@dataclass
class StyleStatistics:
    """Reference에서 추출한 통계. 모든 출력에 동일하게 적용."""
    palette: np.ndarray              # (k, 3) RGB 중심
    palette_weights: np.ndarray      # (k,) 각 색의 빈도
    mean_linewidth: float            # px 단위
    linewidth_distribution: np.ndarray  # 히스토그램
    shading_histogram: np.ndarray    # 밝기 히스토그램 (LAB L*)
    ip_adapter_embedding: torch.Tensor  # 캐싱된 reference 임베딩


def extract_style_statistics(
    reference: Image.Image,
    palette_k: int = 12,
) -> StyleStatistics:
    """Reference 1장에서 모든 일관성 강제용 통계를 추출한다."""
    ...


def enforce_consistency(
    generated_images: List[Image.Image],
    style_stats: StyleStatistics,
    palette_strength: float = 0.7,
    linewidth_strength: float = 0.5,
    shading_strength: float = 0.5,
) -> List[Image.Image]:
    """
    생성된 N개 이미지에 reference 통계를 강제 적용한다.

    - palette quantization을 공유 palette로 일괄 수행
    - linewidth가 reference 평균에 가까워지도록 morphological op 보정
    - shading 히스토그램 매칭 (LAB L* 채널)
    """
    ...
```

**Pipeline 통합**:

```python
def transform_batch(
    self,
    sources: List[Image.Image],
    reference: Image.Image,
    enforce_consistency: bool = True,
    attribute_control: Optional[Dict[str, float]] = None,
) -> List[Image.Image]:
    """
    N개 source를 함께 변환. 출력 N개가 서로 일관된 스타일을 공유한다.

    핵심: reference의 ip_adapter_embedding을 1회만 계산해 모든 source에 재사용.
    그 다음 enforce_consistency=True면 후처리에서 통계 강제.
    """
    style_stats = extract_style_statistics(reference)
    raw_outputs = [
        self.transform(src, reference, _cached_embedding=style_stats.ip_adapter_embedding)
        for src in sources
    ]
    if enforce_consistency:
        return enforce_consistency_fn(raw_outputs, style_stats)
    return raw_outputs
```

**예상 효과 (실험 검증 필요)**: N개 출력의 palette 분포가 reference 중심으로 모임 → σ_palette 감소. 단, **이 자체로는 mechanism level 차별화가 아님** — 외부 후처리로도 모방 가능. 진짜 mechanism level 차이는 §5.5.2에서 시도.

#### 5.5.2 Cross-image attention (shared_attention.py) — *StyleAligned-based Mechanism-Level Consistency* ⚠️ 시도 중

> **상태**: 검증 단계 architectural approach. ADR-010 참조. 안전한 baseline(§5.5.1)이 동작한 뒤 추가.

**목적**: 후처리가 아닌 **diffusion loop 내부**에서 N개 출력의 일관성을 mechanism level로 유도한다. Hertz et al. 2023의 *StyleAligned*가 제안한 **shared self-attention** 을 SDXL + ControlNet + 게임 에셋 도메인에 적용한다.

**핵심 아이디어**: SDXL UNet의 self-attention layer에서, batch dim 0(reference 이미지)의 K(Key), V(Value)를 batch dim 1..N(생성 대상)에 broadcast한다. 즉 모든 출력의 patch가 "내 스타일을 정할 때 reference 이미지의 patch들을 참고"하게 된다.

**의사 코드**:

```python
from diffusers.models.attention_processor import AttnProcessor


class SharedKVAttnProcessor(AttnProcessor):
    """
    Self-attention K/V를 batch[0] (reference)에서 가져와 batch[1..N]에 broadcast.

    원본 StyleAligned (Hertz et al. 2023)을 SDXL + ControlNet 환경에 맞게 수정.
    Reference는 batch dim 0에 위치한다고 가정.
    """

    def __init__(self, share_layers: Optional[List[int]] = None):
        """share_layers: 어떤 attention layer에서 공유할지. None이면 전체.
        성능과 형태 보존의 trade-off를 위해 일부 layer만 공유 가능."""
        super().__init__()
        self.share_layers = share_layers

    def __call__(self, attn, hidden_states, ...):
        # 평소 Q, K, V 계산
        q = attn.to_q(hidden_states)  # (B, seq, dim)
        k = attn.to_k(hidden_states)
        v = attn.to_v(hidden_states)

        # 핵심: batch[0]의 K, V를 batch[1..N]에 broadcast
        # batch[0] (reference) 자체는 원래 K/V를 사용
        shared_k = k[0:1].expand(k.size(0), -1, -1)
        shared_v = v[0:1].expand(v.size(0), -1, -1)

        # 첫 번째 row(reference)는 원래대로, 나머지는 shared
        k = torch.cat([k[0:1], shared_k[1:]], dim=0)
        v = torch.cat([v[0:1], shared_v[1:]], dim=0)

        # 표준 attention
        attn_out = scaled_dot_product_attention(q, k, v)
        return attn.to_out(attn_out)


def apply_shared_attention(
    pipeline: StableDiffusionXLControlNetImg2ImgPipeline,
    share_layers: Optional[List[int]] = None,
):
    """Pipeline의 UNet self-attention processor를 교체."""
    processor = SharedKVAttnProcessor(share_layers=share_layers)
    pipeline.unet.set_attn_processor(processor)
```

**Pipeline 통합**:

```python
def transform_batch(
    self,
    sources: List[Image.Image],
    reference: Image.Image,
    use_shared_attention: bool = True,  # 시도 중인 architectural approach
    enforce_consistency: bool = True,    # 후처리 안전망 (§5.5.1)
    ...,
) -> List[Image.Image]:
    if use_shared_attention:
        apply_shared_attention(self.pipe)
        # Reference를 batch[0]으로 두고 sources를 batch[1..N]으로 함께 denoising
        batch_input = [reference] + sources
        raw_outputs = self.pipe(batch=batch_input, ...).images[1:]  # batch[0]은 버림
    else:
        # 기존 경로: 독립 변환 N회
        raw_outputs = [self.transform(src, reference) for src in sources]

    if enforce_consistency:
        return enforce_consistency_fn(raw_outputs, style_stats)
    return raw_outputs
```

**왜 이게 §5.5.1 후처리 hack과 본질적으로 다른가**:

| | §5.5.1 후처리 hack | §5.5.2 Shared self-attention |
|---|---|---|
| 일관성 강제 시점 | 그림이 다 그려진 *후* (palette 강제 양자화) | 그림을 그리는 *동안* (매 denoising step의 attention) |
| API-only 모델 모방 가능? | ✅ 외부 스크립트로 가능 | ❌ 구조상 불가 (모델 attention layer 접근 필요) |
| 결과 자연스러움 | 강제 양자화로 디테일 잃음 | 처음부터 같은 화풍으로 그려짐 (기대) |
| 차별화 narrative | 약함 | mechanism level (실험 검증 시 강해짐) |

**알려진 위험 (정직한 명시)**:

1. **ControlNet과 충돌 가능성** — ControlNet은 형태를 강하게 고정하는데, shared attention은 reference의 attention 패턴을 빌려옴. **Reference의 형태가 출력에 새어 들어갈 위험** (예: 캐릭터 변환인데 reference의 사물 윤곽이 나타남). `share_layers`를 일부 layer로 제한하거나, ControlNet scale을 상향해 완화 시도.
2. **VRAM 부담** — N개 이미지를 함께 denoising하므로 4070 12GB에서 **N ≤ 3~4가 한계** 예상. 더 큰 batch는 CPU offload + sequential 처리.
3. **단일 이미지 품질 저하 가능성** — Reference에 과적합되어 출력 다양성·디테일이 감소할 수 있음.
4. **첫 시도에 작동 안 할 가능성** — Attention tensor shape 디버깅, batch dim broadcasting, ControlNet과의 상호작용 디버깅 필요.

**성공 기준 (실험 검증)**:
- A4 (후처리 only) 대비 A5 (shared attention + 후처리)에서:
  - σ_palette · σ_linewidth · σ_shading 추가 감소 ≥ 20%
  - DINOv2 Identity 감소 < 5% (형태 보존이 망가지지 않음)
  - 단일 이미지 CLIP Style Similarity 감소 < 10% (품질 trade-off 허용 범위)

이 기준을 못 맞추면 §5.5.2 비활성화 후 §5.5.1만 사용. **narrative는 실험 결과에 따라 사후 결정.**

### 5.6 Module 6: Attribute Control (차별화)

#### 5.6.1 속성 분리 제어 (attribute_control.py) — *Selective Stylization*

**목적**: 자연어 prompt로는 표현 불가능한 정밀도. "팔레트만 적용하고 라인 두께는 원본 유지" 같은 selective stylization.

```python
from enum import Enum
from typing import Dict


class StyleAttribute(str, Enum):
    PALETTE = "palette"
    LINEART = "lineart"
    SHADING = "shading"


def route_scales(
    attributes_on: Dict[StyleAttribute, float],  # 각 속성의 강도 0~1
) -> Dict[str, float]:
    """
    속성별 강도를 ControlNet/IP-Adapter scale + 후처리 단계 on/off로 라우팅한다.

    예:
        {PALETTE: 1.0, LINEART: 0.0, SHADING: 0.5}
        →
        {
            "ip_adapter_scale": 0.3,    # 스타일 영향 최소화
            "controlnet_lineart_scale": 1.2,  # 원본 라인 강하게 보존
            "controlnet_depth_scale": 0.5,
            "palette_quantize_strength": 1.0, # 팔레트만 강하게
            "lineart_postprocess": False,
            "shading_match_strength": 0.5,
        }
    """
    ...
```

**Gradio UI**:

토글 4개 (`palette` / `lineart` / `shading` / `full`) + 각 속성에 강도 슬라이더 (0–100%). `full`은 모든 속성 적용 = 현재 baseline 동작.

### 5.7 Module 7: Region Masking (차별화)

#### 5.7.1 영역 제한 변환 (region_mask.py) — *Spatially-Localized Stylization*

**목적**: "얼굴은 보존, 옷만 변환" 같은 영역 제한. 사용자 마스크 입력 or 자동 마스크 (face detection / focal point).

```python
def apply_region_mask(
    source: Image.Image,
    transformed: Image.Image,
    mask: np.ndarray,  # (H, W) float32 in [0, 1]. 1 = 변환 영역, 0 = 원본 보존
    feather_px: int = 4,
) -> Image.Image:
    """
    mask가 1인 영역만 transformed를 사용하고, 0인 영역은 source 원본 유지.
    경계는 feather로 부드럽게.
    """
    ...


def auto_mask(
    source: Image.Image,
    mode: str = "face_preserve",  # "face_preserve" | "focal" | "none"
) -> np.ndarray:
    """프리셋 마스크 자동 생성."""
    ...
```

**구현 노트**:
- 1차 구현: 사용자 PNG 마스크 입력 + Gradio brush widget.
- 2차 (있으면): SAM2 통합으로 클릭 한 번에 segment.

### 5.8 Module 8: External Baselines (평가 통합)

본 모듈은 **여러 구현체가 같은 역할을 수행**하는 명백한 OOP 케이스다 (ABC + 다형성). evaluator는 baseline 구현체에 무관하게 동일한 인터페이스로 호출한다.

#### 5.8.0 BaselineWrapper ABC (baselines/base.py)

**목적**: 모든 외부/내부 baseline의 공통 인터페이스. evaluator의 ablation loop가 구체 클래스에 의존하지 않도록 분리 (Dependency Inversion + Open/Closed).

```python
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

from PIL import Image


class BaselineWrapper(ABC):
    """모든 baseline 구현체의 공통 인터페이스.

    Attributes:
        name: 결과·로그·실험 폴더에서 식별자로 사용. 예: "gpt_image_2_0", "ipadapter_naive".
        cache_dir: 결과 캐시 디렉토리. None이면 캐시 비활성화.
    """

    name: str
    cache_dir: Optional[Path] = None

    @abstractmethod
    def transform(
        self,
        source: Image.Image,
        reference: Image.Image,
    ) -> Image.Image:
        """단일 (source, reference) → 변환 결과 1장."""
        ...

    def transform_batch(
        self,
        sources: list[Image.Image],
        reference: Image.Image,
    ) -> list[Image.Image]:
        """N개 source를 동일 reference로 변환. 기본 구현은 단순 순회.

        Style Unifier 본체는 이 메서드를 override해 배치 일관성을 보장하지만,
        외부 baseline(GPT Image 등)은 호출이 독립적이라 기본 구현을 그대로 사용.
        이것이 σ_consistency 차이의 원천 (Section 7.2 참조).
        """
        return [self.transform(src, reference) for src in sources]
```

**설계 근거**:
- `transform`은 abstract — 구현체마다 반드시 다름.
- `transform_batch`는 concrete default — 외부 baseline은 그대로 쓰고, Style Unifier 본체(`StyleUnificationPipeline`)는 override해서 §5.5.1 후처리 + §5.5.2 shared attention 적용.
- ABC 유지로 evaluator가 `for baseline in baselines: baseline.transform(...)` 다형적 호출 가능.

#### 5.8.1 GPT Image 2.0 wrapper (baselines/gpt_image.py)

**목적**: 평가 자동화에서 head-to-head 비교를 위한 외부 baseline. 학기 결과물의 메인 비교 대상.

```python
from src.baselines.base import BaselineWrapper


class GPTImageBaseline(BaselineWrapper):
    """OpenAI GPT Image 2.0 API wrapper. 결과를 캐싱해 비용·재현성 확보."""

    name = "gpt_image_2_0"

    def __init__(self, cache_dir: Path = Path("data/cache/gpt_image")):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def transform(
        self,
        source: Image.Image,
        reference: Image.Image,
        prompt: str = "Apply the style of the reference image to the source asset, preserving the design.",
    ) -> Image.Image:
        """결과 캐싱 키: hash(source) + hash(reference) + prompt."""
        ...
```

`transform_batch`는 ABC의 기본 구현(단순 순회)을 그대로 사용 — API 호출이 독립적이라는 사실이 σ_consistency 약점으로 나타남.

#### 5.8.2 IP-Adapter naive baseline (baselines/ipadapter_naive.py)

ControlNet·후처리 없이 IP-Adapter만 적용한 minimal baseline. Ablation A1과 동등.

```python
from src.baselines.base import BaselineWrapper


class IPAdapterNaiveBaseline(BaselineWrapper):
    """IP-Adapter만 적용. ControlNet, 후처리, 배치 일관성 모두 비활성화."""

    name = "ipadapter_naive"

    def __init__(self, config_path: Path = Path("configs/baselines/ipadapter_naive.yaml")):
        ...

    def transform(self, source: Image.Image, reference: Image.Image) -> Image.Image:
        ...
```

---

## 6. 데이터셋

### 6.1 필요한 데이터

한 학기 프로젝트에서는 **LoRA 학습용 데이터는 생략**하고, **평가용 데이터**에 집중한다.

| 목적                | 규모       | 소스                                |
| ------------------- | ---------- | ----------------------------------- |
| 스타일 reference 풀 | 50-100세트 | Itch.io CC0, Kenney.nl, OpenGameArt |
| 변환 대상 풀        | 100-200장  | 위와 동일 + 본인 작업물             |
| Paired 평가셋       | 20-50 세트 | 합성 생성 (후술)                    |

**카테고리 구성**: 정적 2D 에셋이라는 도메인 전반을 다루므로, 한 카테고리에 치우치지 않도록 구성한다.

| 카테고리             | 비율 (목표) | 예시                           |
| -------------------- | ----------- | ------------------------------ |
| 캐릭터               | ~50%        | 사람, 동물, 몬스터             |
| 사물 / 환경 오브젝트 | ~25%        | 상자, 통, 가구, 장식           |
| 아이템 / 도구        | ~25%        | 무기, 방어구, 음식, 포션, 보석 |

본 시스템은 카테고리별 분기 로직이 없어 같은 파이프라인이 모든 카테고리에 작동한다. 다양한 카테고리 데이터로 평가하면 일반화 성능을 더 객관적으로 측정할 수 있다.

### 6.2 수집 스크립트 개요

```python
# scripts/collect_data.py

SOURCES = {
    "kenney": {
        "url_pattern": "https://kenney.nl/assets/...",
        "license": "CC0",
    },
    "opengameart": {
        "url_pattern": "https://opengameart.org/...",
        "license": "varies",  # 필터링 필요
    },
}

def collect_and_filter(
    source: str,
    min_resolution: int = 512,
    max_items: int = 500,
    style_filter: List[str] = ["vector", "illustration", "cartoon"],
):
    """
    - 라이선스 필터 (CC0, CC-BY만)
    - 해상도 필터
    - CLIP classifier로 스타일 태그 자동 부여
    - 수동 검수 대상 목록 출력
    """
    ...
```

### 6.3 평가셋 구축 전략

**문제**: "동일 에셋의 여러 스타일 버전" 같은 paired 데이터는 실제로 수집하기 매우 어렵다.

**해결**: 합성 paired 데이터 생성.

1. CC0 에셋 100장을 "원본"으로 지정.
2. 각 원본에 대해 서로 다른 style LoRA 3-5개를 적용한 버전을 생성.
3. 이 중 하나를 reference로, 다른 하나를 source로 사용.
4. "같은 원본의 다른 스타일 버전"이 ground truth 역할.

**주의**: 합성이므로 실제 아티스트 워크플로우와 차이가 있다. 마지막에는 반드시 본인 작업물이나 실제 게임 에셋으로 보충 평가.

### 6.4 데이터 라벨링 스키마

```json
{
  "asset_id": "kenney_asset_001",
  "source": "kenney",
  "license": "CC0",
  "resolution": [512, 768],
  "category": "character",
  "subcategory": "humanoid",
  "style_tags": ["flat", "cartoon", "thick_outline"],
  "palette_primary": ["#2E5C8A", "#F0E5D0", "#3A3A3A"],
  "has_alpha": true,
  "usage": ["reference_pool", "source_pool"]
}
```

**`category`**의 허용 값:

- `character`: 사람, 동물, 몬스터
- `prop`: 상자, 통, 가구, 장식 등 환경 사물
- `item`: 무기, 방어구, 음식, 포션, 보석 등 휴대 가능한 아이템

이 분류는 평가 시 카테고리별 성능 분석을 위한 것일 뿐, 추론 파이프라인에서는 사용하지 않는다.

---

## 7. 평가 설계

### 7.1 정량 메트릭

```python
# src/evaluation/metrics.py

def clip_style_similarity(generated, reference, model="ViT-L/14"):
    """CLIP image embedding cosine similarity. 높을수록 reference와 유사."""
    ...

def lpips_structure(source, generated, net="vgg"):
    """LPIPS. 낮을수록 원본 구조 보존 우수."""
    ...

def dino_identity(source, generated, model="dinov2_vitl14"):
    """DINOv2 embedding cosine. 높을수록 객체 identity 유지.

    캐릭터에 적용 시 정체성 유지를, 사물·아이템에 적용 시 인식 가능성 유지를 측정한다.
    같은 메트릭을 모든 카테고리에 사용하므로 추가 분기는 없다.
    """
    ...

def palette_distance(reference, generated, k=12):
    """Reference의 k-색 팔레트와 generated의 팔레트 간 earth mover's distance."""
    ...

def gram_matrix_distance(reference, generated, vgg_layers=[0, 5, 10, 19, 28]):
    """Neural style transfer의 Gram matrix 거리."""
    ...
```

### 7.2 일관성 메트릭 (제안 — 차별화의 핵심)

본 프로젝트의 자체 제안 메트릭. **N개 출력 사이의 분포 편차**를 측정해, GPT Image 2.0 같은 독립 생성 모델 대비 우위를 정량화한다.

```python
# src/evaluation/consistency.py

def sigma_palette(outputs: List[Image.Image], k: int = 12) -> float:
    """
    N개 출력의 palette 중심 사이 표준편차.

    각 출력에서 k-color palette를 추출 → palette 중심들을 LAB 공간으로 변환
    → 출력 간 pairwise 거리의 표준편차를 반환.
    낮을수록 N개 출력이 같은 팔레트 가족.
    """
    ...


def sigma_linewidth(outputs: List[Image.Image]) -> float:
    """
    N개 출력의 평균 라인 두께의 표준편차.

    Canny edge map의 connected component 두께를 평균낸 뒤,
    N개 평균값의 std를 반환.
    """
    ...


def sigma_shading(outputs: List[Image.Image]) -> float:
    """
    N개 출력의 LAB L* 채널 히스토그램의 평균 KL-divergence.

    Shading 분포가 일관되는지 측정.
    """
    ...
```

**해석**:
- σ_palette = 0 → N개 출력이 정확히 같은 팔레트 가족
- σ_palette ↑ → 출력들 사이에 색 분포가 흩어짐 (= "짜깁기" 룩)
- GPT Image 2.0: 각 호출이 독립적이라 σ가 크게 나옴 (예상)
- Style Unifier (배치 일관성 ON): σ가 작아짐 (구조적 보장)

### 7.3 Ablation Study 계획

| 실험 | ControlNet | IP-Adapter     | Postproc | Batch consistency | 목적                              |
| ---- | ---------- | -------------- | -------- | ----------------- | --------------------------------- |
| A0   | ❌         | ❌             | ❌       | ❌                | SDXL img2img only (최저 baseline) |
| A1   | ❌         | ✅             | ❌       | ❌                | IP-Adapter 기여도                 |
| A2   | ✅         | ❌             | ❌       | ❌                | ControlNet 기여도                 |
| A3   | ✅         | ✅             | ❌       | ❌                | 둘 다 (형태 + 스타일)             |
| A4   | ✅         | ✅             | ✅       | ❌                | 풀 시스템 (단장 모드)             |
| A5   | ✅         | ✅             | ✅       | ✅                | 풀 시스템 + 배치 일관성 (핵심)    |
| A6   | ✅         | ✅ + Multi-ref | ✅       | ✅                | Multi-reference 확장              |
| **A_GPT** | — | —          | —        | —                 | **GPT Image 2.0 외부 baseline**   |
| A_naive | — | ✅            | —        | —                 | IP-Adapter naive baseline         |

각 실험에서 평가셋 30 세트(샘플 50장 권장) 생성 → 4개 정량 메트릭 + 3개 일관성 메트릭 측정 → 평균/표준편차 기록.

**핵심 비교**:
- **A4 vs A3**: 후처리 기여도
- **A5 vs A4**: 배치 일관성 모듈 기여도 (σ_palette·σ_linewidth·σ_shading가 떨어져야 함)
- **A5 vs A_GPT**: 본 프로젝트 결과물의 메인 narrative. 단일 이미지 CLIP/LPIPS는 비슷하거나 약간 낮을 수 있으나, **σ_consistency·DINOv2 Identity·LPIPS Structure는 명확한 우위**여야 함.

**평가셋 50샘플 구성**: 카테고리별 일반화 성능을 함께 측정하기 위해 다음과 같이 분배:

**평가셋 50샘플 구성**: 카테고리별 일반화 성능을 함께 측정하기 위해 다음과 같이 분배:

- 캐릭터: 25 (50%)
- 사물 / 환경 오브젝트: 13 (~25%)
- 아이템 / 도구: 12 (~25%)

분석 시 전체 평균뿐 아니라 카테고리별 평균도 함께 기록해, 특정 카테고리에서 시스템이 약한지 점검한다.

### 7.4 사용자 스터디 (소규모)

**대상**: 인디 개발자 5–10명 (r/gamedev, 국내 인디 개발자 디스코드 서버, 본인 네트워크).

**프로토콜**:

1. **단일 이미지 비교** — 10쌍의 변환 예시 (A: Style Unifier vs B: GPT Image 2.0, 순서 랜덤). 캐릭터·사물·아이템 골고루.
   - "reference 스타일과 더 가까운 것은?" (style fidelity)
   - "원본 에셋 디자인을 더 잘 보존한 것은?" (content preservation)
   - "게임에 실제로 사용할 수 있는 것은?" (usability)
2. **배치 일관성 비교** (차별화 핵심) — 한 reference로 변환한 N=5장의 출력 세트를 A: Style Unifier vs B: GPT Image 2.0로 제시.
   - "N장이 한 게임의 에셋처럼 보이는 것은?" (batch coherence)
3. 결과는 선호도 %로 보고. 카테고리별 + 질문별 분리.

### 7.5 실전 적용 평가

**본인 개발 게임에 적용**:

- 현재 스타일 불일치가 있는 에셋 10–20장을 선정. 캐릭터·사물·아이템 모두 포함.
- 본 시스템의 **배치 일관성 모드**로 일괄 변환.
- Before/after 스크린샷 비교 (게임 화면 안에 합성).
- 수정 없이 사용 가능한 비율, 수동 보정이 필요한 비율 측정. 카테고리별 기록.

이게 **가장 설득력 있는 평가**다. 논문성보다 실용성을 보여준다.

### 7.6 자동 leaderboard

`scripts/build_leaderboard.py` — 모든 ablation + 외부 baseline의 메트릭을 표로 자동 집계, README에 자동 갱신. 프로젝트의 정량 narrative가 한눈에 보이는 단일 산출물.

---

## 8. 작업 큐 (의존 순서)

> 시간 기반 일정(주차)을 폐기한다. Claude Code로 구현 시간이 더 이상 병목이 아니므로, 진짜 병목인 **GPU 시간 · 데이터 · 사용자 평가**에 맞춘 의존 순서 기반 큐로 진행한다. 상세 추적은 [`docs/roadmap.md`](roadmap.md) 참조.

### 8.1 그룹 개요

```
A. 인프라         (선행 없음)
B. 코어 파이프라인 (선행: A)
C. 평가 인프라    (선행: B)
D. 차별화 기능    (선행: B; C와 병렬 가능)
E. 야심 옵션      (선행: D; 선택)
F. 검증           (선행: C, D)
```

### 8.2 그룹별 작업

**A. 인프라**
- A1. 저장소 구조 + `__init__.py`
- A2. 의존성 확정 (`requirements*.txt`)
- A3. 리눅스 GPU·CUDA 환경 검증
- A4. 모델 다운로드 (`scripts/download_models.py`)
- A5. 재현성 인프라 (`src/utils/{repro,experiment,logging,device}.py`)
- A6. 스모크 테스트 (`scripts/smoke_test.py`)

**B. 코어 파이프라인**
- B1. 전처리 (RMBG / lineart / palette)
- B2. 인코딩 (IP-Adapter wrapper)
- B3. 생성 (`StyleUnificationPipeline` + `configs/default.yaml`)
- B4. 후처리 (alpha restore / palette quantize)
- B5. Gradio UI 최소판

**C. 평가 인프라**
- C1. 정량 메트릭 5개 (`src/evaluation/metrics.py`)
- C2. 일관성 메트릭 3개 (`src/evaluation/consistency.py`) — σ_palette, σ_linewidth, σ_shading
- C3. 평가셋 구축 (수집 + 합성 paired 30세트 + `DATA_CARD.md`)
- C4. 외부 baseline 통합 (`src/baselines/{base,gpt_image,ipadapter_naive}.py` — base.py에 `BaselineWrapper` ABC 정의, §5.8.0)
- C5. 평가 자동화 (`scripts/run_eval.py`)

**D. 차별화 기능** (이전 stretch였던 것들 baseline으로 승격)
- D1. 배치 일관성 모듈 (`src/encoding/batch_consistency.py`) ⭐
- D2. 속성 분리 제어 (`src/postprocessing/attribute_control.py`) ⭐
- D3. Region masking (`src/postprocessing/region_mask.py`) ⭐
- D4. Multi-reference 가중치 (`src/encoding/multi_ref.py`)

**E. 야심 옵션** (선택)
- E1. Test-time scale 자동 튜닝
- E2. Iterative refinement loop
- E3. 개인 LoRA 파인튜닝 (ADR-002 부분 완화)
- E4. Style Bible 산출물

**F. 검증**
- F1. Ablation A0–A6 + A_GPT_Image + A_naive
- F2. 하이퍼파라미터 sweep
- F3. 사용자 스터디 5–10명 (head-to-head with GPT Image 포함)
- F4. 본인 게임 에셋 10–20장 적용
- F5. 자동 leaderboard

### 8.3 게이트 (체크포인트)

각 게이트에서 막히면 후속 그룹이 무의미하므로 우선순위가 가장 높다. **시간이 아닌 게이트 통과 기준으로 진행 여부 판단.**

| 게이트 | 기준 | 실패 시 대응 |
|---|---|---|
| **A** — 스모크 테스트 | reference 1장 → 변환 결과 1장 | 모델 로딩·VRAM 문제부터 해결. Colab Pro 우회 검토. |
| **B** — 코어 동작 | RGBA in/out + 시각적으로 스타일 적용·형태 보존 | ControlNet/IP-Adapter scale 재점검. |
| **C** — 평가 자동화 | 평가셋 30장에 9개 메트릭 자동 측정 + GPT Image 컬럼 | 메트릭 구현 sanity check (known-good pair). |
| **D** — 차별화 기능 통합 | 배치 일관성 + 속성 제어 + Region mask 모두 UI에서 동작 | 모듈 간 인터페이스 정리, 통합 테스트 보강. |
| **F** — 차별화 우위 입증 | A5가 A_GPT 대비 σ_consistency·LPIPS·DINOv2에서 명확 우위 | 배치 일관성 모듈 강화. 그래도 안 되면 narrative를 "본인 게임 적용 정성 데모" 중심으로 조정. |

---

## 9. 기술 의사결정 기록 (ADR)

### ADR-001: 베이스 모델로 SDXL 선택

**결정일**: 2026-04-24  
**결정**: SDXL 1.0 + 일러스트 파인튜닝 체크포인트 (Animagine 등)를 기본으로 사용.

**고려한 대안**:

- **SD 1.5**: 가볍고 커뮤니티 자산 풍부. 하지만 해상도(512) 한계.
- **SD 3 Medium**: 최신 아키텍처. 하지만 IP-Adapter 등 생태계가 아직 미성숙 (2026.04 기준 확인 필요).
- **Flux.1**: 고품질. 하지만 라이선스 제약 (Schnell 제외) + 추론 비용 큼.
- **SDXL**: 1024 네이티브 해상도, ControlNet/IP-Adapter 생태계 성숙.

**결정 근거**: 게임 에셋은 1024 정도 해상도가 실용적이며, 필요한 컨트롤 모듈(IP-Adapter + 복수 ControlNet)이 안정적으로 지원됨.

### ADR-002: LoRA 학습은 baseline scope에서 제외 (개인 LoRA는 야심 옵션으로 부분 허용)

**결정일**: 2026-04-24
**개정일**: 2026-05-12 — Claude Code 도입에 따른 부분 완화
**결정**: 커스텀 LoRA 학습은 **baseline에 포함하지 않고**, 공개 일러스트 체크포인트로 대체. 단, **그룹 E (야심 옵션)에 사용자 본인 에셋 기반 경량 개인 LoRA를 추가**한다.

**고려한 대안**:

- LoRA 학습을 baseline에 포함: 도메인 적응으로 품질 향상 기대.
- LoRA 완전 제외: 기존 체크포인트로 검증.
- **(채택) baseline은 공개 체크포인트, 개인 LoRA는 stretch (E3)**: 두 단계로 분리.

**결정 근거**:
- 데이터 수집과 검수는 사람 시간이며 Claude Code로 단축되지 않는다. 따라서 대규모 LoRA 학습은 여전히 baseline scope 밖.
- 그러나 사용자 본인 에셋 20–50장 정도의 경량 LoRA는 4070에서 몇 시간이면 학습 가능하고, 학습 스크립트 작성은 Claude Code로 빠르게 끝낸다.
- 경량 개인 LoRA는 **"내 게임 스타일을 학습한 reference"** 라는 데모를 가능하게 한다 — GPT Image 2.0이 구조적으로 못 하는 영역이라 차별화 narrative에 부합.

**적용 범위**:
- Baseline: 공개 체크포인트(Animagine-XL-3.1 등)만 사용 (ADR-001).
- Stretch (E3): 사용자 본인 에셋 기반 경량 personalization LoRA.

### ADR-003: Unity 플러그인은 scope 제외

**결정일**: 2026-04-24  
**결정**: Unity C# 플러그인은 한 학기 내에 구현하지 않음. Gradio + CLI로 충분.

**결정 근거**:

- C# 통합은 Python 서버와의 IPC 설계, Unity Editor UI 개발, 에셋 파이프라인 통합 등 별개 작업.
- 본 프로젝트의 핵심 기여는 "스타일 일관성 변환"이지 "통합 툴"이 아님.
- Unity 통합이 없어도 개발자는 Gradio UI로 변환 후 에셋 폴더에 저장 가능.

**재검토 시점**: 졸업 후 Unity Asset Store 배포를 실제로 진행할 때.

### ADR-004: 평가 데이터는 합성으로 보강

**결정일**: 2026-04-24  
**결정**: Paired 평가셋을 실제 수집하지 않고, LoRA 기반 합성으로 구축.

**결정 근거**:

- "동일 에셋의 여러 공식 스타일 버전"이라는 데이터는 존재하지 않음.
- 아티스트 협업은 시간/비용 부담.
- 합성 데이터는 실제 분포와 차이가 있지만, ablation 비교에는 충분.
- 최종 검증은 본인 게임 에셋으로 보충.

### ADR-005: 본 프로젝트의 CV 프로젝트로서의 정체성

**결정일**: 2026-04-24  
**결정**: 본 프로젝트를 "from-scratch CV 알고리즘 연구"가 아닌 **"multi-task vision pipeline 통합 시스템"**으로 포지셔닝.

**배경**: 과목의 평가 가산점 기준이 "입력 이미지 → 출력 이미지/텍스트"로 명시됨. 전통적 CV 알고리즘의 직접 구현 깊이보다 vision-based 시스템의 완성도와 입출력 modality 충족에 평가 초점.

**고려한 대안**:

- **(A) From-scratch CV 알고리즘 구현**: 예컨대 edge detector나 segmentation network를 직접 설계/학습. 깊이 있지만 한 학기에 하나를 제대로 하기도 버거움.
- **(B) Foundation model 활용 + 응용 시스템**: 사전학습 모델들을 조합해 실용적 vision pipeline 구성. 폭이 넓고 현대 CV 학계 주류 흐름과 일치.

**결정 근거**:

- Section 4에 명시된 바와 같이 본 프로젝트는 **추론 단계 9개 vision task** (Semantic Segmentation, Edge Detection, Color Clustering, Visual Feature Extraction, Cross-Attention Conditioning, Spatial Conditioning, Conditional Diffusion, Alpha Channel Restoration, Color Transfer)와 **평가 단계 1개 task** (Perceptual IQA)를 통합한다.
- 입출력 modality가 "이미지 → 이미지(+선택적 텍스트 리포트)"로 평가 기준과 정합.
- 현대 CV 학계의 흐름(Vision Transformer, CLIP, Diffusion 기반 응용 연구)과 일치.
- 한 학기 범위에서 실현 가능하면서도 여러 vision 영역을 실제로 다룸.

**재검토 조건**: 만약 담당 교수의 구체 피드백이 "foundation model 호출만으로는 부족"이라 판단되면, Stretch Goal로 옵션 A를 추가(예: 도메인 특화 segmentation 파인튜닝)하여 보강.

**관련 문서**: Section 4 (Vision Task 매핑) 전체.

### ADR-006: 도메인 범위를 "정적 2D 에셋 (캐릭터 + 사물 + 아이템)"으로 정의

**결정일**: 2026-04-24  
**결정**: 본 시스템의 적용 도메인을 캐릭터에 한정하지 않고, **벡터/일러스트 스타일의 정적 2D 객체** (캐릭터, 사물, 아이템 등)로 정의한다.

**배경**: 프로젝트 초기 단계(리깅 자동화 시점)에는 캐릭터에 한정할 수밖에 없었다. 본(skeleton) 구조와 관절 정의가 캐릭터 해부학에 종속되어 사물에는 본 자체를 박을 수 없기 때문이다. 그러나 프로젝트가 스타일 통일로 피벗된 이후 시스템 구조를 다시 검토하면, 현재 파이프라인은 카테고리에 무관하게 작동한다.

**고려한 대안**:

- **(A) 캐릭터 한정**: 리깅 시절의 가정을 그대로 유지.
- **(B) 정적 2D 에셋 (캐릭터 + 사물 + 아이템)**: 추가 처리 없이 같은 파이프라인이 작동하는 범위로 자연스럽게 확장.
- **(C) 모든 게임 에셋 (UI, 배경, 이펙트, 애니메이션 포함)**: 카테고리별 별도 처리 필요. 한 학기 범위 초과.

**결정 근거**:

- 현재 파이프라인(RMBG → lineart → CLIP → SDXL+ControlNet+IP-Adapter → alpha 복원 → 팔레트 정제)에는 **카테고리별 분기 로직이 없다.** RMBG는 모든 객체에서 배경을 분리하고, lineart는 모든 객체에서 윤곽선을 추출하며, CLIP/SDXL은 카테고리에 무관하게 작동한다.
- 평가 메트릭(LPIPS, CLIP Style, DINOv2, Palette Distance)도 카테고리에 독립적이다. DINOv2 Identity는 캐릭터에서 "정체성", 사물에서 "인식 가능성"으로 의미가 약간 달라지지만 계산식은 동일하다.
- 따라서 옵션 A(캐릭터 한정)는 리깅 시절의 유산이며, 피벗 후 정당화하기 어려운 임의 제약이다.
- 옵션 C는 별도 기술이 필요하다 (UI는 텍스트 OCR 보존, 배경은 반복 텍스처 일관성, 이펙트는 시간축·반투명). Section 2.2에 out-of-scope로 명시.

**영향**:

- 코드 변경: 없음 (파이프라인이 이미 카테고리 무관하게 작동).
- 데이터 수집(그룹 C3): 캐릭터·사물·아이템을 골고루 수집 (Section 6.1).
- 평가셋 구성(그룹 C3): 카테고리별 분포를 명시적으로 정의 (Section 7.3).
- 사용자 스터디(그룹 F3): 카테고리 다양성 반영.

**관련 문서**: Section 2 (Scope), Section 6 (데이터셋), Section 7 (평가).

### ADR-007: 차별화 narrative를 "단일 이미지 품질"에서 "배치 일관성 + 형태 보존 + 속성 제어 + 재현성"으로 전환

**결정일**: 2026-05-12
**결정**: GPT Image 2.0 등 상용 모델이 등장한 환경에서, 본 도구의 차별화 narrative를 **단일 이미지 raw 품질 경쟁에서 철수**하고, 상용 모델이 구조적으로 풀지 않는 4개 영역에 재배치한다.

**4개 영역**:
1. **N개 에셋의 일관성** — 상용 API는 호출마다 독립이라 σ가 큼. 본 도구는 reference 통계 강제로 σ를 mechanism level에서 작게 만듦.
2. **수학적 형태 보존** — 상용 모델의 "디자인 바꾸지 마"는 prompt 부탁. 본 도구는 ControlNet + alpha mask 제약.
3. **속성 단위 제어** — 자연어로는 표현 불가능한 정밀도.
4. **재현성 + 로컬 실행** — 상용 모델은 같은 입력에도 매번 다른 결과 + 외부 서버 업로드 필수.

**고려한 대안**:
- (A) 단일 이미지 raw 품질로 GPT Image와 정면 경쟁 — 솔로 개발자 + 4070 12GB + 오픈소스 SDXL로 비현실적. **기각**.
- (B) (채택) 상용 모델이 구조적으로 못 하는 영역에 재배치 — 정량 측정 가능, 학기 scope 내 실현 가능, 인디 게임 페인 포인트와 정합.

**결정 근거**:
- σ_consistency 메트릭은 본 프로젝트가 처음 제안하는 자체 메트릭으로, 학술적 contribution이 될 수 있다.
- "품질은 비슷하거나 약간 낮지만, 일관성·보존성·재현성은 우위"라는 narrative는 정량 + 정성 양쪽으로 입증 가능.
- 인디 게임 개발자의 실제 페인 포인트("짜깁기 룩")와 정확히 맞는다.

**영향**:
- Hero feature 3가지 신설 (Cross-asset consistency / Structural preservation as constraint / Granular attribute control).
- 평가 메트릭에 σ_palette, σ_linewidth, σ_shading 추가 (Section 7.2).
- Ablation에 GPT Image 2.0을 외부 baseline으로 통합 (Section 7.3 A_GPT).
- 새 모듈 추가: 5.5 batch_consistency, 5.6 attribute_control, 5.7 region_mask, 5.8 baselines.

**관련 문서**: README.md "🆚 왜 GPT Image 2.0이 아닌가" 섹션, Section 1.3, Section 7.2.

### ADR-008: GPT Image 2.0을 평가 baseline으로 통합 (경쟁 대상이 아닌 평가 동료)

**결정일**: 2026-05-12
**결정**: GPT Image 2.0을 ablation의 `A_GPT_Image` 컬럼으로 포함해, 모든 메트릭에서 head-to-head 비교를 자동 생성한다. 결과 캐싱으로 비용을 통제한다 (평가셋 30장 × 1회 ≈ $5).

**고려한 대안**:
- 상용 모델은 무시: GPT Image 등장 이후의 평가자/사용자 입장에서 "왜 이걸 안 쓰나?"라는 질문을 피할 수 없음. **기각**.
- 별도 비교 섹션만: ablation 표에 안 들어가면 narrative가 흩어짐. **기각**.
- **(채택) ablation의 한 row로 통합**: 모든 메트릭에서 직접 비교 가능. 결과 캐싱으로 비용·재현성 확보.

**결정 근거**:
- "단일 이미지 CLIP/LPIPS는 비슷하지만, σ_consistency·DINOv2·LPIPS Structure는 명확한 우위" — 이 narrative는 GPT Image 데이터가 함께 있어야만 성립.
- 결과를 캐싱하면 재실험·재현 비용이 0.
- 평가의 객관성 + 학술적 정직성.

**구현**:
- `src/baselines/gpt_image.py` — OpenAI API wrapper + 디스크 캐시 (key: hash(source) + hash(reference) + prompt).
- `scripts/run_eval.py`가 모든 baseline을 동등하게 호출.

**관련 문서**: Section 5.8, Section 7.3.

### ADR-009: 시간 기반 일정(주차)을 폐기하고 의존 순서 기반 작업 큐로 전환

**결정일**: 2026-05-12
**결정**: 기존 "15주 로드맵 (Week 1–15)"을 폐기하고, 그룹 A–F의 **의존 순서 기반 작업 큐**로 전환한다.

**고려한 대안**:
- 주차별 일정 유지: Claude Code로 구현 속도가 크게 빨라진 환경에서 주차 단위가 의미를 잃음. **기각**.
- 주차 + 게이트 혼용: 이중 추적으로 혼란만 증가. **기각**.
- **(채택) 게이트 기반 진행**: 시간이 아닌 "선행 게이트 통과" 기준으로 다음 그룹 진입.

**결정 근거**:
- Claude Code로 코딩 시간이 단축되어 주차 기반 추정이 부정확해짐.
- 진짜 병목은 (1) GPU 시간, (2) 데이터 수집·검수, (3) 사용자 평가 — 이 세 가지는 Claude Code로 단축되지 않으며, 그룹 간 의존 순서로 모델링하는 게 정확.
- 게이트 기준 진행은 품질 보장 + 진척 가시화에 더 효과적.

**영향**:
- `docs/technical_design.md` Section 8 전면 교체.
- `docs/roadmap.md` 전면 재구성 (의존 순서 기반 작업 큐).
- `README.md` "로드맵" 섹션을 그룹 A–F 트리로 교체.
- 시간 기반 체크포인트 → 게이트 기반 체크포인트로 재정의.

**관련 문서**: Section 8 전체, `docs/roadmap.md` 전체, `CLAUDE.md` 현재 상태 섹션.

### ADR-010: Mechanism-level batch consistency 시도 — StyleAligned 기반 shared self-attention 채택 (검증 단계)

**결정일**: 2026-05-12
**상태**: **시도 중인 architectural approach** — 실험 결과에 따라 narrative 확정.
**결정**: 후처리 기반 통계 강제(§5.5.1)를 안전한 baseline으로 두고, 그 위에 StyleAligned (Hertz et al. 2023) 응용인 shared self-attention(§5.5.2)을 시도한다. 결과를 ablation A4 vs A5로 측정해 사후 채택 여부 결정.

**배경**: ADR-007에서 차별화 narrative를 "배치 일관성 + 형태 보존 + 속성 제어 + 재현성"으로 재구성했다. 그러나 후처리 기반 일관성 강제만으로는 **외부 후처리 스크립트로 GPT Image도 모방 가능**해 mechanism level 차별화가 약하다는 한계가 있다. 이를 보완하기 위해 모델 내부 attention layer 수준의 시도를 추가한다.

**고려한 대안**:

- **(A) 후처리만으로 충분** — 구현 안전. 그러나 차별화 narrative 약함. **부분 채택 (안전한 baseline으로 유지)**.
- **(B) StyleAligned 응용 shared self-attention** — diffusion loop 내부에서 K/V를 reference에서 공유. mechanism level 차이를 mechanism level에서 만듦. **(병행 채택, 실험 검증 후 확정)**.
- **(C) B-LoRA 기반 style subspace decomposition** — 더 강력하지만 LoRA 학습 비용·시간 큼. **그룹 E 야심 옵션으로 별도**.
- **(D) Custom test-time optimization** — 미분 가능 consistency loss로 latent 최적화. 구현 부담 큼. **그룹 E로 별도**.

**(B) 채택 근거**:

- StyleAligned는 peer-reviewed 기법 (Hertz et al. 2023). 실재성 확보.
- `diffusers`의 `AttnProcessor` 인터페이스로 모델 가중치 학습 없이 attention behavior 교체 가능. 4070 12GB에서 실행 가능 (N ≤ 4 예상).
- API-only 상용 모델(GPT Image 등)은 attention layer 접근 권한이 없어 **구조상 모방 불가**. 차별화 narrative의 정직한 근거.

**(B)의 알려진 위험 (정직히 인정)**:

1. **ControlNet과 attention 공유의 충돌** — Reference의 형태가 새어 들어갈 가능성. 형태 보존 제약과 정면 충돌. `share_layers`를 일부 layer로 제한해 완화 시도하되, 실측해야 안다.
2. **결과 품질 우위가 실험 전엔 미지수** — 게임 에셋 도메인 + ControlNet 조합에서 StyleAligned 효과가 원 논문(일반 이미지) 수준으로 재현될 보장 없음.
3. **VRAM 한계** — N ≥ 5 배치는 OOM 위험.
4. **구현 디버깅 부담** — Attention tensor shape, batch dim broadcasting, ControlNet hook과의 상호작용. 첫 시도에 작동 보장 없음.

**의사결정 프로토콜 (실험 결과 기반)**:

그룹 F1 ablation 결과에서:

- A5(shared attention + 후처리)가 A4(후처리 only) 대비 **σ_consistency 추가 감소 ≥ 20%** 이고 **DINOv2 Identity 감소 < 5%** → **채택**. README의 hero feature #1을 "mechanism level batch consistency"로 강화.
- 위 기준 미충족 → **§5.5.2 비활성화**, §5.5.1만 사용. README의 hero feature #1을 "후처리 기반 일관성 강제" 로 약화 (정직). Fallback narrative로 "본인 게임 적용 정성 데모 + 재현성 + 로컬 실행" 강화.

**구현 순서 (안전한 진행)**:

그룹 B (코어 파이프라인) → 그룹 C (평가 인프라) → 그룹 D1 (§5.5.1 후처리 baseline) → **D5 (§5.5.2 shared attention)** → 그룹 F1 (실측). D5 이전 게이트가 흔들리면 절대 진행하지 않는다.

**학술적 contribution 표현 (정직한 수준)**:

"학회 논문 수준의 novelty"라고 주장하지 않는다. 본 프로젝트의 contribution은 **"StyleAligned shared attention을 게임 에셋 도메인 + ControlNet과 결합한 적용 사례 + σ_consistency 메트릭 제안"** 수준이다. 학기 프로젝트·portfolio에는 적합하지만, 학회 publication을 노린 것은 아니다.

**관련 문서**: §5.5.2, README.md hero features 섹션, `docs/roadmap.md` 그룹 D5, ADR-007 (차별화 narrative).

---

## 10. 리스크와 Open Questions

### 10.1 주요 리스크

| 리스크 | 확률 | 영향 | 대응 |
| --- | --- | --- | --- |
| 공개 일러스트 체크포인트 품질 부족 | 중 | 고 | 여러 체크포인트 비교 (Animagine, Counterfeit, Pony 계열). 최악의 경우 stretch goal이었던 LoRA 학습(E3)으로 전환. |
| IP-Adapter가 스타일과 내용을 분리 못함 (reference의 오브젝트가 source에 섞임) | 중 | 고 | InstantStyle 방식(특정 레이어만 주입)으로 완화. 실패 시 IP-Adapter scale 낮추고 ControlNet 의존도 높임. |
| ControlNet이 형태를 과도하게 고정해 스타일 변환이 제한 | 중 | 중 | `controlnet_scale` 감소, lineart 대신 softedge 사용. |
| VRAM 부족 (12GB GPU) | 중 | 중 | `enable_model_cpu_offload()`, sequential CPU offload, 해상도 축소(1024 → 768). |
| **σ_consistency 우위가 약함** | 중 | 고 | 배치 일관성 모듈(D1)의 통계 강제 강도 상향. shared attention(D5) 시도로 mechanism level 강화. 그래도 부족하면 narrative를 "본인 게임 정성 데모" 중심으로 조정. |
| **Shared attention(§5.5.2)이 ControlNet과 충돌해 형태 보존 무너짐** | 중 | 고 | `share_layers`를 일부 attention layer로 제한. ControlNet scale 상향. 그래도 충돌하면 §5.5.2 비활성화하고 §5.5.1 후처리만 사용 (ADR-010 의사결정 프로토콜). |
| **Shared attention 시도가 첫 실행에 작동 안 함** | 고 | 중 | 디버깅 부담을 그룹 D5 게이트에서 흡수. 그룹 B/C/D1-D4 baseline은 별도 진행해 fallback narrative 보장. |
| **GPT Image 2.0이 우리가 측정한 모든 메트릭에서 동등하거나 우수** | 저 | 매우 고 | σ_consistency·DINOv2·LPIPS Structure 중 하나라도 명확 우위 확보 시 narrative 유지. 셋 다 진다면 프로젝트의 존재 의의 재정의 필요. **F1 ablation 결과로 조기 판단**. |
| GPT Image API 정책 변경·접근 불가 | 저 | 중 | 결과를 한 번에 캐싱해 디스크에 보존. API가 막혀도 캐시로 평가 재현 가능. |
| 데이터 수집 시간 과다 소요 | 고 | 중 | 엄격한 타임박스. 부족하면 기존 공개 데이터셋(Danbooru 등, 게임 에셋 아니지만 일러스트) 보조 활용. |
| 사용자 스터디 참가자 모집 실패 | 중 | 중 | 온라인 커뮤니티 외에 교내 게임 개발 동아리, 트위터/디스코드 활용. |
| 생성 품질이 "데모용 체리픽 샘플"에 그침 | 중 | 고 | 평가셋 전체에 대한 랜덤 샘플 평가 의무화. 체리픽 없이 평균 품질 측정. |

### 10.2 Open Questions (해결 필요)

- [ ] `lineart_anime` detector가 벡터 스타일 에셋에도 잘 동작하는가? **그룹 B 진행 중 검증**.
- [ ] σ_palette·σ_linewidth·σ_shading 메트릭이 인간 지각의 "한 가족" 판단과 잘 상관되는가? **그룹 F 사용자 스터디에서 검증** (메트릭 score vs 사람 batch coherence 점수 상관).
- [ ] 배치 일관성 모듈의 통계 강제가 단일 이미지 품질을 얼마나 떨어뜨리는가 (trade-off)? **그룹 F A4 vs A5 비교에서 측정**.
- [ ] Multi-reference "mean" 전략이 reference 스타일을 어색하게 섞는가? **그룹 D4에서 검증**.
- [ ] 팔레트 quantization이 일러스트 스타일에 도움이 되는가, 오히려 품질 저하인가? **그룹 F A3 vs A4 비교에서 검증**.
- [ ] **GPT Image 2.0이 어떤 메트릭에서 우리보다 더 우수한가?** 그 영역은 차별화 narrative에서 정직하게 인정. **그룹 F1 핵심 출력**.
- [ ] 본인 게임의 현재 스타일이 본 시스템으로 유의미하게 통일되는가? **그룹 F4에서 검증. 이게 가장 중요한 질문**.

### 10.3 해결된 질문 (Log)

_(개발 진행하면서 Open Questions에서 여기로 이동)_

---

## 11. 참고 자료

### 11.1 핵심 논문

- **IP-Adapter** (Ye et al., 2023): "IP-Adapter: Text Compatible Image Prompt Adapter for Text-to-Image Diffusion Models"
- **ControlNet** (Zhang et al., 2023): "Adding Conditional Control to Text-to-Image Diffusion Models"
- **InstantStyle** (Wang et al., 2024): "InstantStyle: Free Lunch towards Style-Preserving in Text-to-Image Generation"
- **B-LoRA** (Frenkel et al., 2024): "Implicit Style-Content Separation using B-LoRA"
- **StyleAligned** (Hertz et al., 2023): "Style Aligned Image Generation via Shared Attention"

**서베이 우선순위**: IP-Adapter → ControlNet → InstantStyle → (선택) B-LoRA, StyleAligned

### 11.2 도구 및 라이브러리

- `diffusers` (HuggingFace): 메인 프레임워크
- `transformers`: CLIP, DINOv2
- `controlnet_aux`: lineart, canny, depth 추출기
- `rembg` / `briaai/RMBG-1.4`: 배경 제거
- `lpips`: LPIPS 메트릭
- `open_clip_torch`: CLIP 메트릭
- `gradio`: UI

### 11.3 데이터 소스

- [Kenney.nl](https://kenney.nl) — CC0, 고품질
- [OpenGameArt.org](https://opengameart.org) — CC0/CC-BY 혼재, 필터링 필수
- [Itch.io Free Assets](https://itch.io/game-assets/free) — 라이선스 개별 확인

### 11.4 관련 제품 (경쟁 환경 모니터링)

- **OpenAI GPT Image 2.0** — 범용 이미지 생성. **본 프로젝트의 메인 외부 baseline** (Section 7.3 A_GPT). 차별화 narrative의 핵심 비교 대상.
- Scenario.gg — 신규 에셋 생성, 스타일 변환 일부 지원
- Layer.ai — 팀 협업 기반 에셋 생성
- Retro Diffusion — 픽셀 아트 특화
- PixelLab — 픽셀 아트 특화

---

## 부록 A: 환경 구축 명령어

```bash
# Python 3.10+ 권장
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

pip install -r requirements.txt

# 모델 다운로드 (HuggingFace Hub 필요 시 HF_TOKEN 설정)
python scripts/download_models.py

# 스모크 테스트
python -m src.generation.pipeline --smoke_test

# Gradio 실행
python app/gradio_app.py
```

## 부록 B: requirements.txt (초기안)

```
torch>=2.1.0
torchvision
diffusers>=0.27.0
transformers>=4.38.0
accelerate
safetensors
controlnet-aux
rembg
pillow
numpy
scipy
scikit-learn
scikit-image
lpips
open-clip-torch
gradio>=4.0
pyyaml
tqdm
```

---

_이 문서는 살아있는 문서다. 개발 진행에 따라 Open Questions, ADR, 작업 큐 게이트 통과 상태가 갱신된다._
