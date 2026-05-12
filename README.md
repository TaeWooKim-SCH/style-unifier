# 🎨 Style Unifier

> **여러 출처에서 모은 2D 게임 에셋을, 한 가족처럼 보이게 통일시키는 로컬 AI 도구**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![PyTorch 2.1](https://img.shields.io/badge/PyTorch-2.1-EE4C2C.svg)](https://pytorch.org/)
[![Status: Alpha](https://img.shields.io/badge/status-alpha-orange.svg)]()

<!-- 데모 이미지 placeholder. 베이스라인 + 일관성 모듈 완성 후 실제 before/after로 교체 -->
<p align="center">
  <!-- <img src="docs/assets/demo_banner_placeholder.png" alt="Demo: N개 에셋이 동일한 스타일로 통일되는 before/after 예시" width="800"/> -->
  <br>
  <em>(데모 이미지는 코어 파이프라인 + 배치 일관성 모듈 완성 후 업데이트 예정)</em>
</p>

---

## 🎯 해결하려는 문제

인디 게임 개발자는 에셋을 여러 경로로 조달합니다. 외주 아티스트, Unity Asset Store 구매, 개발자 본인 작업물, AI 생성 이미지. 그 결과는 **스타일 불일치**입니다.

- 🎨 아티스트마다 라인 두께, 그림자 방향, 색감이 제각각
- 🧩 에셋 스토어에서 구매한 것들은 스타일이 섞여 있음
- ⏳ 전부 다시 그리려면 현실적으로 불가능

Steam 인디 게임 리뷰에서 **에셋이 짜깁기처럼 보인다**는 비판이 흔한 이유입니다. 시각적 완성도가 낮으면 게임 자체 품질이 아무리 좋아도 판매가 어렵습니다.

---

## 🆚 왜 GPT Image 2.0 / Scenario / 일반 img2img가 아닌가

상용 이미지 생성 모델은 **한 장을 멋지게 만드는 데** 최적화되어 있습니다. Style Unifier는 그 싸움을 하지 않습니다. 대신 **상용 모델이 구조적으로 풀지 못하는 세 가지 문제**에 집중합니다.

| | GPT Image 2.0 / 일반 img2img | Style Unifier |
|---|---|---|
| **N개 에셋의 일관성** | 각 호출이 독립 → 라인 두께·팔레트·그림자가 흔들림 | (baseline) Reference 통계를 후처리로 강제. (시도) StyleAligned 응용 attention 공유로 diffusion 내부 일관성 유도 — 검증 단계 |
| **형태 보존** | 프롬프트 "디자인 바꾸지 마" — 권유에 불과 | ControlNet + alpha mask로 **수학적 제약** |
| **속성 단위 제어** | 자연어로만 — 거친 제어 | 팔레트만 / 라인만 / 그림자만 토글 + scale 슬라이더 |
| **재현성** | 같은 입력에도 매번 다른 출력 | seed + config 고정으로 bit-exact 재생성 |
| **프라이버시·비용** | 에셋을 외부 서버로 업로드, 장당 과금 | 로컬 GPU에서 실행, 무제한 재실험 |

> 단일 이미지의 raw 품질에서는 상용 모델이 더 좋을 수 있습니다. Style Unifier가 푸는 문제는 **"한 장을 잘 만드는 것"이 아니라 "N장이 일관되게 보이는 것"** 입니다.

---

## ⭐ Hero Features

### 1. Cross-Asset Consistency (배치 일관성)
N개 에셋을 함께 입력하면, 그들 사이의 **palette·lineart·shading 분포 편차 σ**를 최소화한다. 단일 이미지가 reference와 얼마나 닮았느냐가 아니라, **출력들 사이의 일관성**을 보장한다.

**두 가지 메커니즘으로 접근**:
- **(baseline)** Reference에서 추출한 통계(palette·linewidth·shading)를 N개 출력에 후처리로 일괄 적용. 항상 동작 보장.
- **(시도 중인 architectural approach)** Hertz et al. 2023의 *StyleAligned* 를 SDXL + ControlNet + 게임 에셋 도메인에 적용해, diffusion loop 내부의 self-attention K/V를 reference에서 공유. 후처리가 아닌 **모델이 그림을 그리는 과정 자체**에서 일관성을 유도. **실험 검증 단계** — 결과에 따라 narrative 강화/약화. 자세한 설계 결정은 [ADR-010](docs/technical_design.md#adr-010) 참조.

### 2. Structural Preservation as Constraint (수학적 형태 보존)
ControlNet(lineart)로 원본 구조를 강하게 잠그고, 원본 alpha mask를 후처리에서 재부착해 **실루엣 일치 + 디자인 정체성**을 동시에 유지한다. DINOv2 identity score로 정량 검증.

### 3. Granular Attribute Control (속성 단위 제어)
`palette only` / `lineart only` / `shading only` / `full` 토글 + 각 속성에 대한 강도 슬라이더. 자연어 프롬프트로는 표현하기 어려운 정밀도.

---

## 🧠 어떻게 동작하는가

<!-- <p align="center">
  <img src="docs/assets/pipeline_diagram_placeholder.png" alt="Vision pipeline: 원본 에셋 + reference → segmentation, edge detection, CLIP encoding → SDXL + ControlNet + IP-Adapter → alpha restoration → 최종 출력" width="700"/>
</p> -->

추론 시 9개의 vision task를 통합하고, 별도 평가 단계에서 Perceptual IQA + 일관성 메트릭을 수행합니다.

| 단계   | Vision Task                  | 모델/기법                   |
| ------ | ---------------------------- | --------------------------- |
| 전처리 | Semantic Segmentation        | RMBG-1.4                    |
| 전처리 | Edge Detection               | lineart_anime / Canny       |
| 전처리 | Color Clustering             | K-means                     |
| 인코딩 | Visual Feature Extraction    | CLIP ViT-H/14               |
| 인코딩 | Cross-Attention Conditioning | IP-Adapter projection layer |
| 생성   | Spatial Conditioning         | ControlNet (lineart, depth) |
| 생성   | Conditional Diffusion        | SDXL + Animagine-XL-3.1     |
| 후처리 | Alpha Channel Restoration    | Feathering + Mask 재부착    |
| 후처리 | Color Transfer               | Palette quantization        |

**평가 단계** (개발·검증 시점만 사용, 추론 외):

| 단계 | Vision Task    | 사용 메트릭                                                                |
| ---- | -------------- | -------------------------------------------------------------------------- |
| 평가 | Perceptual IQA | LPIPS, CLIP Style Sim, DINOv2 Identity, Palette EMD                        |
| 평가 | Consistency    | σ_palette, σ_linewidth, σ_shading (배치 내 표준편차)                       |
| 평가 | External baseline | GPT Image 2.0 / IP-Adapter naive — 동일 입력에 대한 head-to-head 비교   |

핵심 차별점은 **형태 보존 + 배치 일관성을 정량 보장**한다는 점입니다. 상세 기술 설계는 [docs/technical_design.md](docs/technical_design.md) 참조.

---

## 🚀 Quick Start

> **⚠️ 현재 알파 단계**. 코어 파이프라인 완성 후 베타 공개 예정.
> 진행 상황은 [docs/roadmap.md](docs/roadmap.md) 참조.

### 요구사항

- **실행 환경**: Linux + NVIDIA GPU (VRAM 12GB+)
- **개발 환경**: Linux / macOS / Windows (Python 3.11)

### 설치

```bash
# 저장소 클론
git clone https://github.com/TaewooKim-SCH/style-unifier.git
cd style-unifier

# Python 3.11 가상환경
python3.11 -m venv venv
source venv/bin/activate    # Linux/macOS
# venv\Scripts\activate      # Windows

# 의존성 설치
pip install --upgrade pip
pip install -r requirements.txt

# HuggingFace 인증 (토큰 필요)
huggingface-cli login

# 모델 다운로드 (~20GB)
python scripts/download_models.py

# 검증
python scripts/smoke_test.py
```

상세한 OS별 설치 가이드: [SETUP.md](SETUP.md)

### 사용법 (코어 파이프라인 완성 후 가능)

#### CLI — 단일 변환

```bash
python -m style_unifier \
    --source path/to/asset.png \
    --reference path/to/style_reference.png \
    --output path/to/output.png
```

#### CLI — 배치 일관성 모드

```bash
python -m style_unifier batch \
    --sources path/to/assets/*.png \
    --reference path/to/style_reference.png \
    --output-dir path/to/output/ \
    --enforce-consistency
```

#### Gradio UI

```bash
python app/gradio_app.py
# 브라우저: http://127.0.0.1:7860
```

#### Python API

```python
from style_unifier import StyleUnificationPipeline
from PIL import Image

pipeline = StyleUnificationPipeline.from_config("configs/default.yaml")

# 단일 변환
result = pipeline.transform(
    source=Image.open("asset.png"),
    reference=Image.open("target_style.png"),
)
result.save("output.png")

# 배치 일관성 모드 — N개 에셋이 서로 일관되게 변환됨
results = pipeline.transform_batch(
    sources=[Image.open(p) for p in asset_paths],
    reference=Image.open("target_style.png"),
    enforce_consistency=True,
    control={"palette": 1.0, "lineart": 1.0, "shading": 0.7},  # 속성별 강도
)
```

---

## 📊 측정 가능한 차별화 목표

상용 모델과 raw 품질을 직접 비교하는 대신, **상용 모델이 풀지 않는 메트릭**으로 우위를 측정합니다.

| 메트릭                            | 목표                                                | 비교 대상                                |
| --------------------------------- | --------------------------------------------------- | ---------------------------------------- |
| **σ_palette** (배치 내)           | GPT Image 2.0 baseline 대비 **−50% 이상**           | 같은 reference로 N장 생성 후 분산 측정   |
| **σ_linewidth** (배치 내)         | GPT Image 2.0 baseline 대비 **−50% 이상**           | 동일                                     |
| **DINOv2 Identity** (원본 대비)   | GPT Image 2.0 baseline 대비 **+15% 이상**           | 원본 디자인 정체성 보존도               |
| **LPIPS Structure** (원본 대비)   | GPT Image 2.0 baseline 대비 **−30% 이상**           | 형태 보존 우위                           |
| **CLIP Style Similarity**         | 상용 모델과 ±5% 이내                                | "품질은 비슷, 일관성·보존성 우위" narrative |
| **재현성**                        | 동일 config·seed로 **bit-exact 재생성**             | 상용 모델은 불가능                       |
| **로컬 비용**                     | 장당 $0 (GPU 전기료 외)                             | GPT Image 2.0 장당 $0.04–0.19            |

추가로 **인디 개발자 5–10명 사용자 스터디** + **본인 게임 에셋 10–20장 적용 데모**로 실용성 검증.

---

## 🎮 타겟 사용자

### 이런 분들에게 유용합니다

- 여러 소스에서 에셋을 조달하는 **인디 게임 개발자**
- 외주 아티스트 간 스타일을 통일하고 싶은 **소규모 게임 스튜디오**
- AI 생성 이미지를 게임 에셋화하려는 **1인 개발자**
- 에셋을 외부 서버로 보내고 싶지 않은 **IP 민감 스튜디오**
- 동일 config로 6개월 후 같은 결과를 재생성해야 하는 **장기 운영 프로젝트**

### 현재 지원 범위

본 도구는 **벡터/일러스트 스타일의 정적 2D 객체** 도메인을 다룹니다. 카테고리별 분기 로직 없이 동일한 파이프라인이 모든 카테고리에 작동합니다.

**도메인 내 (지원)**:

- ✅ **캐릭터** (사람, 동물, 몬스터)
- ✅ **사물** (상자, 통, 가구, 장식)
- ✅ **아이템** (무기, 방어구, 음식, 포션, 보석)
- ✅ RGBA 형식, 단일 객체

**도메인 외 (별개 처리 필요)**:

- ❌ **픽셀 아트** — 그리드 정합, 색 제한 등 별도 처리. [Retro Diffusion](https://www.retrodiffusion.ai/) 추천
- ❌ **UI/버튼** — 텍스트 OCR 보존 추가 처리 필요
- ❌ **배경/타일맵** — 반복 텍스처 일관성, seamless 보장 별도 처리
- ❌ **애니메이션/이펙트** — 시간축 일관성, 반투명 처리 별개 연구 주제
- ❌ **포토리얼 스타일** — 베이스 모델 도메인 mismatch

---

## 🗺️ 작업 큐 (의존 순서)

시간 기반 일정 대신 **의존 순서**로 진행합니다. 각 그룹은 선행 그룹이 완료된 뒤 시작됩니다. 상세는 [docs/roadmap.md](docs/roadmap.md) 참조.

```
A. 인프라
   환경 셋업 · 모델 다운로드 · 스모크 테스트 · 재현성 인프라

B. 코어 파이프라인
   전처리(RMBG · lineart · palette) · StyleUnificationPipeline
   (IP-Adapter + ControlNet) · 후처리(alpha · palette quantize)

C. 평가 인프라
   5개 메트릭 + 합성 paired 평가셋
   σ_consistency 메트릭 (배치 일관성)
   GPT Image 2.0 baseline 통합

D. 차별화 기능
   배치 일관성 모듈 (후처리 baseline) · 속성 분리 제어
   Region masking · Multi-reference 가중치
   D5. Cross-image attention (StyleAligned 응용) ⚠️ 시도 중인 architectural approach

E. 야심 옵션 (선택)
   Test-time scale 자동 튜닝 · Iterative refinement loop
   개인 LoRA 파인튜닝 · Style Bible 산출물

F. 검증
   Ablation (A0–A5 + A_GPT_Image) · 사용자 스터디 5–10명
   본인 게임 에셋 10–20장 적용 · 자동 leaderboard
```

릴리즈 이후 계획 (현 scope 외):

- 🔜 Unity Asset Store 플러그인
- 🔜 SaaS 웹 서비스
- 🔜 배경·UI·이펙트 에셋 지원 확장

---

## 🏗️ 기술 스택

- **언어**: Python 3.11
- **프레임워크**: PyTorch 2.1, Diffusers, Transformers
- **베이스 모델**: SDXL + Animagine-XL-3.1
- **컨트롤**: IP-Adapter-Plus, ControlNet (lineart, depth)
- **UI**: Gradio
- **평가**: LPIPS, CLIP, DINOv2, palette EMD + 자체 σ_consistency 메트릭

전체 아키텍처 상세: [docs/technical_design.md](docs/technical_design.md)

---

## 📂 프로젝트 구조

```
style-unifier/
├── src/                    # 핵심 코드
│   ├── preprocessing/      # BG 제거, Lineart, 팔레트
│   ├── encoding/           # IP-Adapter, Multi-ref, Batch consistency
│   ├── generation/         # SDXL + ControlNet 파이프라인
│   ├── postprocessing/     # Alpha 복원, 팔레트 정제, Attribute control
│   ├── evaluation/         # 평가 메트릭 + 일관성 메트릭
│   └── baselines/          # GPT Image 2.0 등 외부 baseline wrapper
├── app/                    # Gradio UI
├── scripts/                # 데이터 수집, 모델 다운로드, leaderboard
├── configs/                # YAML 설정
├── experiments/            # 실험 결과 (NNN_name/)
├── tests/                  # pytest
├── docs/                   # 문서
│   ├── technical_design.md
│   └── roadmap.md
├── .claude/agents/         # Claude Code 에이전트 정의
├── CLAUDE.md               # Claude Code 컨텍스트
├── STYLE.md                # 코드 스타일
└── SETUP.md                # 환경 구축
```

---

## 🤝 기여하기

현재는 **개인 연구 프로젝트** 단계라 외부 PR은 받지 않습니다. 다만:

- 🐛 **버그·이슈 제보**: Issues 탭에서 자유롭게
- 💡 **기능 제안**: Discussions 탭에서 환영
- 🎨 **데이터 제공**: CC0 라이선스로 기여 가능한 게임 에셋이 있다면 연락 주세요

정식 기여 가이드는 1.0 릴리즈 이후 공개 예정.

---

## 📜 라이선스

이 프로젝트는 [MIT 라이선스](LICENSE)를 따릅니다.

자유롭게 사용·수정·배포 가능하며, 상업적 사용도 허용됩니다. 단, 원 저작권 표시는 유지해주세요.

### 의존 모델 라이선스 주의

본 도구는 다음 사전학습 모델들을 사용합니다. **각 모델의 라이선스를 개별 확인**하세요:

- SDXL: [CreativeML Open RAIL++-M License](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/blob/main/LICENSE.md)
- Animagine-XL-3.1: [Fair-AI-Public-License-1.0-SD](https://huggingface.co/cagliostrolab/animagine-xl-3.1)
- IP-Adapter: [Apache 2.0](https://github.com/tencent-ailab/IP-Adapter)
- ControlNet: [OpenRAIL](https://github.com/lllyasviel/ControlNet)

**특히 상업적 사용 시** 각 모델의 라이선스 조건 재확인 필수.

---

## 📚 인용

학술 목적으로 이 도구를 사용한 경우, 아래 형식으로 인용해주세요 (1.0 릴리즈 후 arXiv 논문 업데이트 예정):

```bibtex
@software{style_unifier_2026,
  author = {Kim, Taewoo},
  title  = {Style Unifier: Local, Reproducible, Consistency-Guaranteed Style Unification for 2D Game Assets},
  year   = {2026},
  url    = {https://github.com/TaewooKim-SCH/style-unifier}
}
```

---

## 🔗 관련 링크

- 📖 [기술 설계서](docs/technical_design.md) — 전체 아키텍처와 설계 결정
- 🗓️ [작업 큐 & 진행 상황](docs/roadmap.md) — 의존 순서 기반 진행 추적
- 🛠️ [환경 구축 가이드](SETUP.md) — OS별 설치 방법
- 💻 [코드 스타일 가이드](STYLE.md) — 기여자용

### 유사 / 비교 대상

- [OpenAI GPT Image 2.0](https://openai.com/index/) — 범용 이미지 생성 (head-to-head baseline)
- [Scenario.gg](https://www.scenario.gg) — 게임 에셋 신규 생성 SaaS
- [Retro Diffusion](https://www.retrodiffusion.ai/) — 픽셀 아트 특화
- [PixelLab](https://www.pixellab.ai/) — 픽셀 아트 AI 편집기
- [IP-Adapter](https://github.com/tencent-ailab/IP-Adapter) — 본 도구가 활용하는 핵심 기술

---

## 📬 연락

- GitHub Issues: 버그·기능 제안
- Email: `zop1234@hanmail.net`

---

<p align="center">
  <strong>🎨 한 장이 아니라 N장이 한 가족처럼. 로컬에서, 결정론적으로. 🎨</strong>
  <br><br>
  <sub>Built with ☕ and a lot of RTX 4070 VRAM anxiety.</sub>
</p>
