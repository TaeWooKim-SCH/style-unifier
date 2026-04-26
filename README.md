# 🎨 Style Unifier

> **2D 게임 에셋의 스타일을 참조 이미지 기준으로 자동 통일시키는 AI 도구**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![PyTorch 2.1](https://img.shields.io/badge/PyTorch-2.1-EE4C2C.svg)](https://pytorch.org/)
[![Status: Alpha](https://img.shields.io/badge/status-alpha-orange.svg)]()
[![Week 0/15](https://img.shields.io/badge/progress-Week%200%2F15-lightgrey.svg)]()

<!-- 데모 이미지 placeholder. Week 2 이후 실제 before/after로 교체 -->
<p align="center">
  <!-- <img src="docs/assets/demo_banner_placeholder.png" alt="Demo: 여러 소스에서 조달한 에셋이 하나의 스타일로 통일되는 before/after 예시" width="800"/> -->
  <br>
  <em>(데모 이미지는 Week 2 베이스라인 완성 후 업데이트 예정)</em>
</p>

---

## 🎯 해결하려는 문제

인디 게임 개발자는 에셋을 여러 경로로 조달합니다. 외주 아티스트, Unity Asset Store 구매, 개발자 본인 작업물, AI 생성 이미지. 그 결과는 **스타일 불일치**입니다.

- 🎨 아티스트마다 라인 두께, 그림자 방향, 색감이 제각각
- 🧩 에셋 스토어에서 구매한 것들은 스타일이 섞여 있음
- ⏳ 전부 다시 그리려면 현실적으로 불가능

Steam 인디 게임 리뷰에서 **에셋이 짜깁기처럼 보인다**는 비판이 흔한 이유입니다. 시각적 완성도가 낮으면 게임 자체 품질이 아무리 좋아도 판매가 어렵습니다.

**Style Unifier는 이 문제를 해결합니다.** 기준이 되는 reference 이미지 한 장을 주면, 나머지 에셋들의 스타일을 그 기준으로 자동 변환합니다. **원본 디자인은 그대로 보존**한 채 스타일만 이식합니다.

---

## 🧠 어떻게 동작하는가

<!-- <p align="center">
  <img src="docs/assets/pipeline_diagram_placeholder.png" alt="Vision pipeline: 원본 에셋 + reference → segmentation, edge detection, CLIP encoding → SDXL + ControlNet + IP-Adapter → alpha restoration → 최종 출력" width="700"/>
</p> -->

Vision pipeline으로 9개의 추론 단계 task를 통합하고, 별도의 평가 단계에서 Perceptual IQA를 수행합니다:

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

**평가 단계** (개발·검증 시점만 사용, 추론 파이프라인 외):

| 단계 | Vision Task    | 사용 메트릭                                     |
| ---- | -------------- | ----------------------------------------------- |
| 평가 | Perceptual IQA | LPIPS, CLIP Style Sim, DINOv2, Palette Distance |

핵심 차별점은 **형태 보존 강제**입니다. 범용 img2img 모델은 reference의 디자인을 섞어버리지만, 이 도구는 ControlNet으로 원본 구조를 강하게 제약해서 **원본 디자인은 그대로, 스타일만 변환**을 달성합니다.

상세한 기술 설계는 [docs/technical_design.md](docs/technical_design.md) 참조.

---

## 🚀 Quick Start

> **⚠️ 현재 Week 0 (개발 초기 단계)**. 전체 기능은 Week 15(2026-08)에 1.0 릴리즈 예정.
> 베타 사용을 원한다면 로드맵에서 현재 진행 상황을 먼저 확인하세요.

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

### 사용법 (예정, Week 2 베이스라인 완성 후 가능)

#### CLI

```bash
python -m style_unifier \
    --source path/to/asset.png \
    --reference path/to/style_reference.png \
    --output path/to/output.png
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

# 캐릭터, 사물, 아이템 모두 동일한 인터페이스로 처리 가능
source = Image.open("asset.png")
reference = Image.open("target_style.png")

result = pipeline.transform(source, reference)
result.save("output.png")
```

---

## 📊 성과 (로드맵)

프로젝트 종료 시점(Week 15) 달성 목표:

- ✅ Baseline 대비 **CLIP Style Similarity +15% 이상 개선**
- ✅ **LPIPS −30% 이상 개선** (범용 img2img 대비 형태 보존 우위)
- ✅ 인디 개발자 **5-10명 대상 사용자 스터디** 결과 확보
- ✅ 실제 게임 에셋 **10-20장에 적용한 before/after** 데모

현재 진행 상황은 [docs/roadmap.md](docs/roadmap.md) 참조.

---

## 🗺️ 로드맵

| Phase             | 주차       | 목표                                 | 상태       |
| ----------------- | ---------- | ------------------------------------ | ---------- |
| Foundation        | Week 1-3   | 환경 구축, Baseline                  | 🔵 진행 중 |
| Data & Evaluation | Week 4-6   | 데이터, 평가셋, Ablation             | ⏸️ 대기    |
| Refinement        | Week 7-10  | Postprocessing, HP 탐색              | ⏸️ 대기    |
| Validation        | Week 11-15 | 사용자 스터디, 게임 적용, 1.0 릴리즈 | ⏸️ 대기    |

릴리즈 이후 계획:

- 🔜 Unity Asset Store 플러그인 (2026 Q4)
- 🔜 SaaS 웹 서비스 (2027 Q1)
- 🔜 배경·UI·이펙트 에셋 지원 확장

---

## 🎮 타겟 사용자

### 이런 분들에게 유용합니다

- 여러 소스에서 에셋을 조달하는 **인디 게임 개발자**
- 외주 아티스트 간 스타일을 통일하고 싶은 **소규모 게임 스튜디오**
- AI 생성 이미지를 게임 에셋화하려는 **1인 개발자**
- 벡터/일러스트 스타일의 **2D 게임 제작자** (캐릭터·사물·아이템 모두 지원)

### 현재 지원 범위

본 도구는 **벡터/일러스트 스타일의 정적 2D 객체** 도메인을 다룹니다. 카테고리별 분기 로직 없이 동일한 파이프라인이 모든 카테고리에 작동합니다.

**도메인 내 (지원)**:

- ✅ **캐릭터** (사람, 동물, 몬스터)
- ✅ **사물** (상자, 통, 가구, 장식)
- ✅ **아이템** (무기, 방어구, 음식, 포션, 보석)
- ✅ RGBA 형식, 단일 객체

**도메인 외 (별개 처리 필요)**:

- ❌ **픽셀 아트** — 그리드 정합, 색 제한 등 별도 처리 필요. [Retro Diffusion](https://www.retrodiffusion.ai/) 추천
- ❌ **UI/버튼** — 텍스트 OCR 보존이 추가로 필요
- ❌ **배경/타일맵** — 반복 텍스처 일관성, seamless 보장 별도 처리
- ❌ **애니메이션/이펙트** — 시간축 일관성, 반투명 처리 별개 연구 주제
- ❌ **포토리얼 스타일** — 베이스 모델 도메인이 다름

---

## 🏗️ 기술 스택

- **언어**: Python 3.11
- **프레임워크**: PyTorch 2.1, Diffusers, Transformers
- **베이스 모델**: SDXL + Animagine-XL-3.1
- **컨트롤**: IP-Adapter-Plus, ControlNet (lineart)
- **UI**: Gradio
- **평가**: LPIPS, CLIP, DINOv2

전체 아키텍처 상세: [docs/technical_design.md](docs/technical_design.md)

---

## 📂 프로젝트 구조

```
style-unifier/
├── src/                    # 핵심 코드
│   ├── preprocessing/      # BG 제거, Lineart, 팔레트
│   ├── encoding/           # IP-Adapter, Multi-ref
│   ├── generation/         # SDXL + ControlNet 파이프라인
│   ├── postprocessing/     # Alpha 복원, 팔레트 정제
│   └── evaluation/         # 평가 메트릭
├── app/                    # Gradio UI
├── scripts/                # 데이터 수집, 모델 다운로드
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
  title  = {Style Unifier: AI-Based Style Unification for 2D Game Assets},
  year   = {2026},
  url    = {https://github.com/TaewooKim-SCH/style-unifier}
}
```

---

## 🔗 관련 링크

- 📖 [기술 설계서](docs/technical_design.md) — 전체 아키텍처와 설계 결정
- 🗓️ [개발 로드맵](docs/roadmap.md) — 주차별 진행 상황
- 🛠️ [환경 구축 가이드](SETUP.md) — OS별 설치 방법
- 💻 [코드 스타일 가이드](STYLE.md) — 기여자용

### 유사 프로젝트 / 참고

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
  <strong>🎨 인디 게임 개발의 비주얼 완성도를 한 단계 높입니다. 🎨</strong>
  <br><br>
  <sub>Built with ☕ and a lot of RTX 4070 VRAM anxiety.</sub>
</p>
