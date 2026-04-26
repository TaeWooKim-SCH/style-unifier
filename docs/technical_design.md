# AI 기반 게임 에셋 스타일 일관성 도구

> 정적 2D 게임 에셋(캐릭터·사물·아이템)을 reference 이미지 기준으로 스타일 통일하는 도구  
> **Technical Design Document (TDD)** — 본인 개발 참고용  
> Last updated: 2026-04-26 (Vision Task 매핑 정확성 개선 및 도메인 범위 일반화)

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

**인디 게임 개발자가 여러 소스에서 조달한 정적 2D 게임 에셋(캐릭터, 사물, 아이템 등)의 스타일을, reference 이미지 기준으로 자동 통일시키는 도구.**

### 1.2 핵심 제약 (순서 중요)

1. **형태 보존**: 원본 에셋의 디자인 자체는 바꾸지 않는다.
2. **스타일만 이식**: 팔레트, 라인, 그림자, 텍스처만 reference에 맞춘다.
3. **도메인 특화**: 포토리얼이 아닌 벡터/일러스트 스타일의 정적 2D 에셋 대상.
4. **게임 엔진 친화**: 투명 배경(알파 채널) 유지.

### 1.3 성공 기준 (한 학기 종료 시점)

- [ ] Baseline(IP-Adapter 단독) 대비 정량 지표 개선 (LPIPS, CLIP style similarity)
- [ ] 본인 개발 게임에 실제 적용한 before/after 샘플 10세트 이상
- [ ] 인디 개발자 5-10명 대상 사용자 스터디 결과 확보
- [ ] GitHub 공개 가능한 코드베이스 + README + 데모

### 1.4 CV 프로젝트로서의 정체성

본 프로젝트는 **이미지 입력 → 이미지 출력**의 vision pipeline으로, 전통적 CV 주제(Semantic Segmentation, Edge Detection, Color Clustering, Alpha Matting)와 현대 CV 주제(Visual Representation Learning, Conditional Diffusion Generation, Spatial Conditioning)를 통합한다. 평가 단계에서는 Perceptual IQA의 다양한 메트릭(LPIPS, CLIP, DINOv2)을 활용한다. 상세 매핑은 [Section 4](#4-vision-task-매핑) 참조.

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

### 2.3 Stretch Goals (여유 있으면 시도)

- Multi-reference 지원 (3.3절 참조)
- 간단한 Unity import helper 스크립트 (플러그인 아님)
- 자체 수집 에셋 기반 경량 LoRA 파인튜닝

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
│   │   └── multi_ref.py
│   ├── generation/
│   │   ├── pipeline.py       # 메인 생성 파이프라인
│   │   └── controlnet.py
│   ├── postprocessing/
│   │   ├── alpha_restore.py
│   │   ├── palette_quantize.py
│   │   └── quality_check.py
│   ├── evaluation/
│   │   ├── metrics.py        # CLIP, LPIPS, etc.
│   │   └── run_eval.py
│   └── utils/
│       ├── io.py
│       └── logging.py
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
| E    | 7.1 정량 메트릭 | **Perceptual Image Quality Assessment (IQA)** | LPIPS, CLIP sim, DINOv2, Palette Dist |

평가 단계는 추론 파이프라인의 일부가 아니라 개발·검증 단계에서만 사용되는 별도 phase다. 그래서 위 9개와 분리해 표기한다.

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

#### 5.2.2 Multi-Reference 전략 (multi*ref.py) — \_Feature Aggregation (Stretch Goal)*

> **참고**: Single-reference baseline에서는 이 단계가 동작하지 않는다. Multi-reference 사용 시(Week 9 stretch goal)에만 의미가 있는 task다. Section 4.1의 Vision Task 매핑에서 Single-reference 기준 9개 task를 카운트할 때 이 항목은 제외된다.

**Stretch goal**. 한 학기 내에는 "mean" strategy까지만 확정 구현, "weighted"는 시간 있으면 추가.

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

### 7.2 Ablation Study 계획

| 실험 | ControlNet | IP-Adapter     | Postproc | 목적                              |
| ---- | ---------- | -------------- | -------- | --------------------------------- |
| A0   | ❌         | ❌             | ❌       | SDXL img2img only (최저 baseline) |
| A1   | ❌         | ✅             | ❌       | IP-Adapter 기여도                 |
| A2   | ✅         | ❌             | ❌       | ControlNet 기여도                 |
| A3   | ✅         | ✅             | ❌       | 둘 다 (형태 + 스타일)             |
| A4   | ✅         | ✅             | ✅       | 풀 시스템                         |
| A5   | ✅         | ✅ + Multi-ref | ✅       | Stretch goal                      |

각 실험에서 50샘플 생성 → 4개 정량 메트릭(CLIP, LPIPS, DINOv2, Palette) 측정 → 평균/표준편차 기록.

**평가셋 50샘플 구성**: 카테고리별 일반화 성능을 함께 측정하기 위해 다음과 같이 분배:

- 캐릭터: 25 (50%)
- 사물 / 환경 오브젝트: 13 (~25%)
- 아이템 / 도구: 12 (~25%)

분석 시 전체 평균뿐 아니라 카테고리별 평균도 함께 기록해, 특정 카테고리에서 시스템이 약한지 점검한다.

### 7.3 사용자 스터디 (소규모)

**대상**: 인디 개발자 5-10명 (r/gamedev, 국내 인디 개발자 디스코드 서버, 본인 네트워크).

**프로토콜**:

1. 10쌍의 변환 예시 제시 (A: baseline vs B: 본 시스템, 순서는 랜덤). 카테고리는 캐릭터·사물·아이템을 골고루 섞어 시스템의 일반화를 평가에 반영.
2. 각 쌍에 대해 다음 3개 질문:
   - "reference 스타일과 더 가까운 것은?" (style fidelity)
   - "원본 에셋 디자인을 더 잘 보존한 것은?" (content preservation)
   - "게임에 실제로 사용할 수 있는 것은?" (usability)
3. 결과는 선호도 %로 보고. 카테고리별 결과도 함께 보고하면 어떤 도메인에서 강한지 파악 가능.

### 7.4 실전 적용 평가

**본인 개발 게임에 적용**:

- 현재 스타일 불일치가 있는 에셋 10-20장을 선정. 캐릭터·사물·아이템을 모두 포함.
- 본 시스템으로 일괄 변환.
- Before/after 스크린샷 비교.
- 수정 없이 사용 가능한 비율, 수동 보정이 필요한 비율 측정. 카테고리별로 분리해 기록.

이게 **가장 설득력 있는 평가**다. 논문성보다 실용성을 보여준다.

---

## 8. 개발 로드맵 (15주)

### 8.1 주차별 체크리스트

#### Phase 1: Foundation (1-3주차)

**Week 1** — 환경 구축

- [ ] 저장소 초기화, 디렉토리 구조 생성
- [ ] `requirements.txt`, `pyproject.toml` 작성
- [ ] GPU 환경 확인 (VRAM 12GB+ 필요)
- [ ] SDXL, IP-Adapter, ControlNet 모델 로컬 캐싱
- [ ] 기본 generation 스모크 테스트 (reference 1장으로 1장 변환)

**Week 2** — Baseline 구현

- [ ] `StyleUnificationPipeline` 클래스 최소 기능 구현
- [ ] IP-Adapter 단독 변환 → 결과 샘플 확보
- [ ] ControlNet(lineart) 추가 → 형태 보존 개선 확인
- [ ] Gradio UI 기본 화면 (input, reference, output)

**Week 3** — 전처리 파이프라인

- [ ] BG removal 모듈
- [ ] Lineart 추출 모듈
- [ ] 팔레트 추출 모듈
- [ ] 단위 테스트 작성

#### Phase 2: Data & Evaluation (4-6주차)

**Week 4** — 데이터 수집

- [ ] Kenney.nl CC0 에셋 스크레이핑 (합법적 방법으로만)
- [ ] 라이선스 필터링 스크립트
- [ ] 자동 태깅 (CLIP classifier)
- [ ] 수동 검수 50장

**Week 5** — 평가셋 & 메트릭

- [ ] 합성 paired 평가셋 30세트 구축
- [ ] 5개 평가 메트릭 구현 및 단위 테스트
- [ ] 평가 자동화 스크립트 (`run_eval.py`)

**Week 6** — Baseline 평가

- [ ] A0 ~ A3 ablation 실행
- [ ] 결과 분석, 실패 케이스 수집
- [ ] Week 7 이후 개선 방향 결정

#### Phase 3: Refinement (7-10주차)

**Week 7** — Postprocessing

- [ ] 알파 복원, 팔레트 양자화, 품질 검증 모듈
- [ ] A4 실험 (풀 시스템) 평가

**Week 8** — 하이퍼파라미터 탐색

- [ ] `ip_adapter_scale`, `controlnet_scale`, `denoising_strength` grid search
- [ ] 최적 조합 확정 + `configs/best.yaml` 저장

**Week 9** — Multi-reference (Stretch)

- [ ] 단순 mean 전략 구현
- [ ] A5 실험
- [ ] 시간 부족 시 이 주차 skip 가능

**Week 10** — 버그 수정 & 안정화

- [ ] 엣지 케이스 (작은 에셋, 긴 에셋, 단색 reference 등)
- [ ] 메모리 누수 확인 (긴 배치 실행 시)
- [ ] 에러 핸들링 강화

#### Phase 4: Validation & Delivery (11-15주차)

**Week 11** — 사용자 스터디 준비

- [ ] 설문 폼 작성 (Google Forms / Typeform)
- [ ] 샘플 20세트 준비 (baseline vs full)
- [ ] 참가자 모집 (5-10명 목표)

**Week 12** — 사용자 스터디 진행

- [ ] 데이터 수집
- [ ] 결과 분석

**Week 13** — 본인 게임 적용

- [ ] 실제 게임 에셋 10-20장 변환
- [ ] Before/after 스크린샷 제작
- [ ] 실패 케이스 기록

**Week 14** — 문서화

- [ ] README 작성
- [ ] API 문서 (docstring → Sphinx or mkdocs)
- [ ] 실험 결과 정리 (tables, figures)

**Week 15** — 발표 & 마무리

- [ ] 발표 자료
- [ ] 데모 영상
- [ ] GitHub 공개
- [ ] 최종 보고서

### 8.2 중간 체크포인트

| 시점       | 기준                                         | 실패 시 대응                                               |
| ---------- | -------------------------------------------- | ---------------------------------------------------------- |
| Week 3 말  | Baseline 1장 변환 성공                       | 모델 로딩/인프라 문제 해결. 불가 시 Colab Pro 사용         |
| Week 6 말  | Ablation A0-A3 결과 확보                     | 결과가 기대 이하면 ControlNet 세팅 재검토                  |
| Week 10 말 | Full system 품질이 baseline 대비 유의미 개선 | 개선 없으면 multi-ref 포기하고 postprocessing 강화 집중    |
| Week 13 말 | 본인 게임에 실제 적용 가능한 품질            | 불가능하면 "부분 자동화 + 수동 보정" 워크플로우로 포지셔닝 |

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

### ADR-002: LoRA 학습은 한 학기 scope에서 제외

**결정일**: 2026-04-24  
**결정**: 커스텀 LoRA 학습은 하지 않고, 공개 일러스트 체크포인트로 대체.

**고려한 대안**:

- LoRA 학습 포함: 도메인 적응으로 품질 향상 기대.
- LoRA 제외: 기존 체크포인트로 검증.

**결정 근거**: 한 학기 15주에서 LoRA 학습은 최소 3-4주를 소요하며, 데이터 품질 이슈로 실패 가능성이 높다. 먼저 기존 체크포인트로 baseline을 확실히 만든 뒤, 여유가 있으면 stretch goal로 이동.

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
- 데이터 수집(Week 4): 캐릭터·사물·아이템을 골고루 수집 (Section 6.1).
- 평가셋 구성(Week 5): 카테고리별 분포를 명시적으로 정의 (Section 7.2).
- 사용자 스터디(Week 11-12): 카테고리 다양성 반영.

**관련 문서**: Section 2 (Scope), Section 6 (데이터셋), Section 7 (평가).

---

## 10. 리스크와 Open Questions

### 10.1 주요 리스크

| 리스크                                                                        | 확률 | 영향 | 대응                                                                                                        |
| ----------------------------------------------------------------------------- | ---- | ---- | ----------------------------------------------------------------------------------------------------------- |
| 공개 일러스트 체크포인트 품질 부족                                            | 중   | 고   | 여러 체크포인트 비교 (Animagine, Counterfeit, Pony 계열). 최악의 경우 stretch goal이었던 LoRA 학습으로 전환 |
| IP-Adapter가 스타일과 내용을 분리 못함 (reference의 오브젝트가 source에 섞임) | 중   | 고   | InstantStyle 방식(특정 레이어만 주입)으로 완화. 실패 시 IP-Adapter scale 낮추고 ControlNet 의존도 높임      |
| ControlNet이 형태를 과도하게 고정해 스타일 변환이 제한                        | 중   | 중   | `controlnet_scale` 감소, lineart 대신 softedge 사용                                                         |
| VRAM 부족 (12GB GPU)                                                          | 중   | 중   | `enable_model_cpu_offload()`, sequential CPU offload, 해상도 축소                                           |
| 데이터 수집 시간 과다 소요                                                    | 고   | 중   | Week 4에 엄격히 타임박스. 부족하면 기존 공개 데이터셋(Danbooru 등, 게임 에셋 아니지만 일러스트) 보조 활용   |
| 사용자 스터디 참가자 모집 실패                                                | 중   | 중   | 온라인 커뮤니티 외에 교내 게임 개발 동아리, 트위터/디스코드 활용                                            |
| 생성 품질이 "데모용 체리픽 샘플"에 그침                                       | 중   | 고   | Week 10까지 랜덤 샘플 평가 의무화. 체리픽 없이 평균 품질 측정                                               |

### 10.2 Open Questions (해결 필요)

- [ ] `lineart_anime` detector가 벡터 스타일 에셋에도 잘 동작하는가? Week 2에 검증.
- [ ] Multi-reference의 "mean" 전략이 각 reference 스타일을 섞어 어색한 혼합을 만드는가? Week 9에 검증.
- [ ] 팔레트 quantization이 일러스트 스타일에 도움이 되는가, 오히려 품질 저하인가? Week 7에 검증.
- [ ] 본인 게임의 현재 스타일이 본 시스템으로 유의미하게 통일되는가? Week 13에 검증. **이게 가장 중요한 질문**.

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

_이 문서는 살아있는 문서다. 개발 진행에 따라 Open Questions, ADR, 주차별 체크리스트가 갱신된다._
