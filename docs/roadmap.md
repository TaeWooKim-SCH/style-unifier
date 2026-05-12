# docs/roadmap.md

> **살아있는 문서**. 진행 상황 변화 시 갱신.
> CLAUDE.md의 "현재 상태" 섹션이 이 문서를 요약 참조함.

이 문서는 **시간 기반 일정이 아닌, 의존 순서 기반 작업 큐**다. Claude Code로 구현하기 때문에 코딩 시간은 더 이상 병목이 아니며, 진짜 병목은 GPU 시간·데이터 수집·사용자 평가다. 따라서 각 항목을 "몇 주차에 한다"가 아니라 "선행 조건이 충족되면 한다"로 추적한다.

---

## 🎯 현재 상태 대시보드

> 작업 단위 진행이 바뀔 때마다 갱신.

- **현재 상태**: 베타 직전 (B/C/D 코드 완성, 실제 추론·평가·ablation 검증 대기)
- **현재 그룹**: **B+C+D 코드 작성 완료** — macOS mock 260개 테스트 통과
- **다음 게이트**: (1) B 게이트 — Linux pipeline.transform RGBA 시각 검사, (2) C 게이트 — `python scripts/run_eval.py` 30페어×9메트릭, (3) F1 ablation A4 vs A5로 D5 채택/폐기 판정 (ADR-010)
- **Blocker**: 없음

### 진행 중 작업

- [x] B1 전처리 (`src/preprocessing/{bg_removal,lineart,palette}.py`)
- [x] B2 인코딩 (`src/encoding/ip_adapter_wrapper.py` — StyleEncoder, D 진입 시 pipeline에 재연결)
- [x] B3 파이프라인 (`src/generation/{pipeline,pipeline_loader}.py` — 분리됨, 500줄 원칙 준수)
- [x] B4 후처리 (`src/postprocessing/{alpha_restore,palette_quantize}.py`)
- [x] B5 Gradio UI (`app/gradio_app.py` — 3패널 최소판)
- [x] `configs/default.yaml` (palette_strength, lineart_detector 키 포함)
- [x] C1 메트릭 5종 (`src/evaluation/metrics.py`, lazy 모델 로딩)
- [x] C2 일관성 메트릭 3종 (`src/evaluation/consistency.py`, σ_palette/linewidth/shading, LAB)
- [x] C4 BaselineWrapper ABC + GPTImageBaseline + IPAdapterNaiveBaseline
- [x] C5 평가 자동화 (`scripts/run_eval.py` + `_eval_helpers.py`, reference-aware chunk)
- [x] C3 데이터 수집 스크립트 (`scripts/collect_data.py` + `build_eval_pairs.py` + `_pair_strategies.py` + `data/DATA_CARD.md`)
- [x] D1 배치 일관성 후처리 baseline (`src/encoding/batch_consistency.py`, §5.5.1)
- [x] D2 attribute control (`src/postprocessing/attribute_control.py`)
- [x] D3 region mask (`src/postprocessing/region_mask.py`)
- [x] D4 multi-reference (`src/encoding/multi_ref.py`)
- [x] D5 shared attention 시도 (`src/encoding/shared_attention.py`, §5.5.2, ADR-010 ⚠️ Linux 실측 + F1 ablation 판정 대기)
- [x] 단위 테스트 260개 통과 (3.31s, mock 기반, 22 skipped는 macOS skimage/torchvision 의존성)
- [ ] Linux 머신에서 실제 추론·평가 검증
- [ ] Pipeline 통합 — `transform_batch(use_shared_attention, enforce_consistency)`, attribute/region mask 연결
- [ ] 그룹 A 잔여: GPU·드라이버 검증, 모델 다운로드 ~20GB

### 미해결 질문

- [ ] Kenney.nl CC0 에셋 라이선스 실제 확인 (재배포 가능?)
- [ ] Animagine-XL-3.1 외 일러스트 체크포인트 후보 (Counterfeit, Pony 등) 비교 필요?
- [ ] GPT Image 2.0 baseline 비용 — 평가셋 30장 1회 + 캐싱이면 ~$5 추정. 확인 필요.

---

## 📊 그룹별 진행 현황

| 그룹 | 내용 | 상태 | 진행률 |
|---|---|---|---|
| **A. 인프라** | 환경·재현성·스모크 테스트 | 🔵 진행 중 | 80% (A3·A6-full 잔여) |
| **B. 코어 파이프라인** | 전처리 + 생성 + 후처리 | 🔵 진행 중 | 90% (Linux 실측 검증 잔여) |
| **C. 평가 인프라** | 메트릭 + 평가셋 + GPT Image baseline | 🔵 진행 중 | 90% (Linux 실측 + 실데이터 수집 잔여) |
| **D. 차별화 기능** | 배치 일관성 + 속성 제어 + Region mask + Multi-ref + Shared attention | 🔵 진행 중 | 90% (Pipeline 통합 + Linux 검증 잔여) |
| **E. 야심 옵션** | Test-time 튜닝 / Iterative refine / 개인 LoRA / Style Bible | ⏸️ 대기 | 0% |
| **F. 검증** | Ablation + 사용자 스터디 + 게임 적용 + leaderboard | ⏸️ 대기 | 0% |

**상태 범례**: ⏸️ 대기 / 🔵 진행 중 / ✅ 완료 / ⚠️ 지연 / ❌ 포기

---

## 🧱 작업 큐 (의존 순서)

각 그룹은 **선행 그룹의 게이트 통과 후 시작**한다. 그룹 내 항목은 대체로 순서대로 진행하되, 독립적인 것은 병렬화 가능.

---

### A. 인프라 (선행 없음)

**목적**: Linux GPU에서 코드를 안정적으로 돌릴 수 있는 상태 + 재현 가능한 실험 기록 체계 확립.

#### A1. 저장소 구조

- [x] `src/{preprocessing,encoding,generation,postprocessing,evaluation,utils,baselines}/__init__.py`
- [x] `tests/`, `configs/`, `experiments/`, `scripts/`, `app/` 디렉토리
- [x] `.gitignore`, `.gitattributes` 확인 (RGBA·체크포인트 제외, LF 강제)

#### A2. 의존성

- [x] `requirements.txt` (공통) + `requirements-cuda.txt` (리눅스 추론용) 확정
- [x] `requirements-dev.txt` (ruff, pyright, pytest)
- [ ] 리눅스에서 venv + PyTorch CUDA 설치 → `torch.cuda.is_available()` 확인 (Linux 머신에서 수행)

#### A3. 환경 검증 (Linux 머신에서 수행)

- [ ] NVIDIA 드라이버, CUDA 12.1, `nvidia-smi` 확인
- [ ] `enable_model_cpu_offload()` 동작 검증 (12GB VRAM)
- [ ] HuggingFace 토큰 발급, 환경변수 설정

#### A4. 모델 다운로드

- [x] `scripts/download_models.py` — SDXL, Animagine-XL-3.1, IP-Adapter, ControlNet (canny, depth), RMBG-1.4, CLIP ViT-H/14, DINOv2 (eval용)
- [ ] 다운로드 검증 (~20GB) — Linux 머신에서 실행

#### A5. 재현성 인프라

- [x] `src/utils/repro.py` — seed 고정 헬퍼 (torch, numpy, random, CUDA) + `set_deterministic`, `make_generator`
- [x] `src/utils/experiment.py` — 실험별 디렉토리 자동 생성 (max+1 ID), config + git hash + 환경 정보 기록
- [x] `src/utils/logging.py` — 표준 logger (`STYLE_UNIFIER_LOG_LEVEL` 환경변수 지원)
- [x] `src/utils/device.py` — device 자동 선택 (cuda > mps > cpu) + `get_dtype`

#### A6. 스모크 테스트

- [x] `scripts/smoke_test.py` skeleton — 환경/utils/diffusers 임포트/SDXL 로딩(`--full`)/experiment 디렉토리 6단계
- [ ] reference 1장 → 변환 결과 1장 단계 추가 (B3 pipeline 완료 후). **이 게이트가 닫혀 있으면 인프라 문제부터 해결.**

**🚦 게이트 A**: `python scripts/smoke_test.py` → "All checks passed" 출력 + 결과 PNG 1장 생성.

> **실패 시 대응**: 모델 로딩/VRAM 문제 해결에 최우선 집중. 최악의 경우 Colab Pro로 임시 우회하고 로컬 환경은 병행 수정.

---

### B. 코어 파이프라인 (선행: A)

**목적**: 단일 reference + 단일 source → 단일 출력의 end-to-end 동작.

#### B1. 전처리 모듈

- [x] `src/preprocessing/bg_removal.py` (RMBG-1.4 wrapper, 174줄, fast path + force_remove)
- [x] `src/preprocessing/lineart.py` (controlnet_aux의 lineart_anime + canny fallback, 142줄)
- [x] `src/preprocessing/palette.py` (sklearn k-means, 결정론적, 83줄)
- [x] 각 모듈 단위 테스트 (mock + 작은 fixture, 32개)

#### B2. 인코딩

- [x] `src/encoding/ip_adapter_wrapper.py` — CLIP 이미지 임베딩 (`StyleEncoder`, IP-Adapter Plus의 `last_hidden_state`, 359줄). **현재 transform()에서 미사용 — 그룹 D 진입 시 multi-reference/shared attention 경로에서 명시적으로 연결 예정.**
- [x] 단위 테스트 19개 (lazy init, cache, encode_single/multi, weighted 가중 평균)

#### B3. 생성 파이프라인

- [x] `src/generation/pipeline.py` — `StyleUnificationPipeline` 클래스 (427줄)
- [x] `src/generation/pipeline_loader.py` — 로딩 로직 분리 (188줄)
  - [x] 베이스 SDXL + Animagine-XL-3.1 로드 (config 기반)
  - [x] IP-Adapter 로드 + scale 설정 (`load_ip_adapter`)
  - [x] ControlNet (lineart, depth) 로드 (`load_controlnets`)
  - [x] `transform(source, reference, seed, prompt, return_intermediates)` 메서드 — RGBA in → RGBA out
- [x] `configs/default.yaml` 작성 (`palette_strength`, `lineart_detector` 포함)
- [x] 통합 테스트 (lazy 가드, load_config 검증)

#### B4. 후처리

- [x] `src/postprocessing/alpha_restore.py` — 원본 mask 재부착 + feathering (115줄)
- [x] `src/postprocessing/palette_quantize.py` — soft palette snap, alpha 보존 (149줄)
- [x] 단위 테스트 20개

#### B5. Gradio UI 최소판

- [x] `app/gradio_app.py` — source / reference / output 3패널 + seed/prompt/config 컨트롤 (264줄)
- [ ] Linux 머신에서 실제 변환 동작 확인 (`python app/gradio_app.py`)

**🚦 게이트 B**:
- `pipeline.transform(source, reference)` → RGBA 출력
- 출력 형식 자동 검증 (RGBA, 차원, alpha 유효성)
- 시각 검사로 reference 스타일이 적용되었음 + 원본 형태 보존이 가시적으로 확인됨
- `pytest -m "not slow"` 통과

> **체크포인트**: 이 게이트에서 reference 스타일이 적용되지 않거나 원본 형태가 무너지면, ControlNet/IP-Adapter scale 세팅부터 재점검. 이게 실패하면 이후 그룹 전부 무의미.

---

### C. 평가 인프라 (선행: B)

**목적**: 정량 메트릭 + 평가셋 + 외부 baseline을 갖춰 ablation을 자동 실행 가능한 상태.

#### C1. 정량 메트릭

- [x] `src/evaluation/metrics.py` (430줄, lazy 캐시 4종)
  - [x] CLIP Style Similarity (ViT-L-14, open_clip) — 모델명 Linux 검증 TODO
  - [x] LPIPS Structure (VGG)
  - [x] DINOv2 Identity (`facebook/dinov2-large`, CLS 토큰)
  - [x] Palette Distance (간이 EMD = 양방향 최근접 거리 평균)
  - [x] Gram Matrix Distance (VGG-19 5개 layer)
- [x] 단위 테스트 16개 (mock 기반, 6 skipped는 macOS skimage/torchvision 의존성)

#### C2. 일관성 메트릭 (차별화의 핵심)

- [x] `src/evaluation/consistency.py` (294줄)
  - [x] **σ_palette**: LAB 공간 변환 후 평균 색 벡터의 채널별 std + L2
  - [x] **σ_linewidth**: edge mask의 `distance_transform_edt` × 2 (reviewer C1 버그 수정 — 원래 `~edge_mask` 였음)
  - [x] **σ_shading**: alpha>0 영역의 LAB L* 히스토그램 pairwise symmetric KL
- [x] 단위 테스트 18개 (N=1 edge case, 결정론, monotonic)

#### C3. 평가셋 구축

- [x] `scripts/collect_data.py` (480줄) — 로컬 스캔 + license filter (URL 다운로드는 미구현 — 사용자 수동 다운로드)
- [x] CLIP classifier 자동 태깅 (`--auto-tag`, open_clip ViT-B-32)
- [ ] 수동 검수 50장 (Linux에서 실제 Kenney 데이터 수집 후)
- [x] `scripts/build_eval_pairs.py` + `_pair_strategies.py` (366+182줄, stratified 50/25/25)
- [x] `data/DATA_CARD.md` (307줄, 출처/라이선스/스키마/편향 명시)

#### C4. 외부 baseline 통합

- [x] `src/baselines/base.py` (146줄) — `BaselineWrapper` ABC + `CachedBaselineMixin` (SHA-256 키)
- [x] `src/baselines/gpt_image.py` (216줄) — `GPTImageBaseline`, OPENAI_API_KEY 가드, `max_calls`, 캐싱. reference 미전달 한계는 Linux TODO
- [x] `src/baselines/ipadapter_naive.py` (175줄) — `IPAdapterNaiveBaseline`, ControlNet·후처리 없음
- [ ] `StyleUnificationPipeline`도 `BaselineWrapper` 상속 검토 (Linux pipeline 통합 시 함께)
- [ ] 평가셋 30장 × 외부 baseline 1회 호출 결과 캐싱 (~$5, Linux 실측)

#### C5. 평가 자동화

- [x] `scripts/run_eval.py` (489줄) + `_eval_helpers.py` (329줄) — 30페어×9메트릭, reference-aware chunk (reviewer C3 수정), summary.md 주의 사항 포함
- [x] `experiments/NNN/` 자동 디렉토리 생성 (`src.utils.init_experiment` 활용)

**🚦 게이트 C**: `python scripts/run_eval.py --config configs/default.yaml` → 평가셋 30장에 대해 6개 정량 + 3개 일관성 메트릭이 계산되고, GPT Image baseline 컬럼이 함께 출력됨.

---

### D. 차별화 기능 (선행: B 완료, C와 병렬 가능)

**목적**: GPT Image 2.0이 구조적으로 못 하는 것들을 baseline 기능으로 구현.

#### D1. 배치 일관성 모듈 — 후처리 baseline ⭐ (안전)

> §5.5.1 — 후처리 기반 통계 강제. 항상 동작 보장. **mechanism level 차별화는 아님**, 외부 후처리로 모방 가능.

- [x] `src/encoding/batch_consistency.py` (470줄)
  - [x] Reference에서 palette·lineart·shading 통계 추출 (`StyleStatistics` dataclass)
  - [x] N개 source에 동일 통계를 강제 (palette quantize 재사용, linewidth morph 보정, shading histogram matching)
  - [x] IP-Adapter 임베딩 캐싱 옵션 (`cache_embedding=False` 기본값 — reviewer I4)
- [ ] Pipeline에 `transform_batch(sources, reference, enforce_consistency=True)` 메서드 추가 (Linux 통합)
- [x] 단위 테스트 14개 (mock 기반)

#### D2. 속성 분리 제어 ⭐

- [x] `src/postprocessing/attribute_control.py` (236줄)
  - [x] `palette` / `lineart` / `shading` / `full` / `selective` 5개 모드 (`route_scales`)
  - [x] `AttributeScales` dataclass — ControlNet/IP-Adapter scale + 후처리 strength 라우팅
- [ ] Gradio UI에 토글 + scale 슬라이더 추가 (Linux 통합)
- [x] 단위 테스트 14개

#### D3. Region masking ⭐

- [x] `src/postprocessing/region_mask.py` (277줄)
  - [x] 사용자 마스크 입력 지원 (`apply_region_mask`, gaussian feathering)
  - [x] 마스크 영역 외 = 원본 보존 (alpha 채널까지 blend)
  - [x] `auto_mask("focal"/"none")` 지원 — `"face_preserve"` 는 placeholder + WARNING
- [ ] Gradio UI에 마스크 그리기 위젯 통합 (Linux 통합)
- [x] 단위 테스트 13개

#### D4. Multi-reference 가중치 블렌딩

- [x] `src/encoding/multi_ref.py` (230줄)
  - [x] `aggregate_references` — `mean` / `weighted` / `first` / `manual` 4가지
  - [x] `compute_clip_similarities` — target과 reference들의 cosine (디버깅용)
- [ ] Gradio UI에 다중 reference 입력 + 가중치 슬라이더 (Linux 통합)
- [x] 단위 테스트 13개

#### D5. Cross-image attention (StyleAligned 응용) ⚠️ **시도 중인 architectural approach**

> §5.5.2 / ADR-010 — diffusion loop 내부 mechanism level 일관성 시도. **검증 단계**, 실험 결과에 따라 채택/폐기 결정.
> **선행 조건**: D1–D4 안정적으로 동작 + 그룹 C 평가 인프라 완성 (정량 검증 가능 상태). D1이 fallback baseline 역할.

- [x] `src/encoding/shared_attention.py` (423줄)
  - [x] `SharedKVAttnProcessor` 구현 (diffusers `AttnProcessor2_0` 호환 패턴)
  - [x] batch dim 0(reference)의 K/V를 batch dim 1..N에 broadcast (`_broadcast_kv_from_reference`)
  - [x] `share_layers` 옵션 — 일부 attention layer만 공유 (apply 시점 인덱싱)
  - [x] `apply_shared_attention` / `remove_shared_attention` (원본 processor 복원 — reviewer I1)
  - [x] `validate_batch_for_shared_attention` (B<2, B>max, 해상도 일치 — reviewer I2)
- [ ] Pipeline에 `apply_shared_attention()` 토글 추가 (Linux 통합)
- [ ] `transform_batch(..., use_shared_attention=True)` 경로 — reference + sources를 함께 denoising
- [ ] VRAM 검증 (N ≤ 4 권장; OOM 시 sequential fallback) — Linux 실측
- [x] 단위 테스트 22개 (validate, broadcast 값 검증, apply/remove cycle)
- [ ] ControlNet 충돌 점검 — Linux tensor shape 디버깅 후

**🚦 게이트 D5 (실험 검증)** — 그룹 F1 ablation 결과로 판정:
- A5(shared attention + 후처리) vs A4(후처리 only)에서
  - σ_palette · σ_linewidth · σ_shading **추가 감소 ≥ 20%**
  - DINOv2 Identity **감소 < 5%** (형태 보존 망가지지 않음)
  - 단일 이미지 CLIP Style Similarity **감소 < 10%** (품질 trade-off 허용 범위)
- **충족 → 채택**, README hero feature를 "mechanism level batch consistency"로 강화.
- **미충족 → §5.5.2 비활성화**, §5.5.1 후처리만 사용. README hero feature를 "후처리 기반 일관성 강제"로 정직히 약화. Fallback narrative: "본인 게임 적용 정성 데모 + 재현성 + 로컬 실행" 강화.

> ⚠️ D5 실패는 프로젝트 실패가 아니다. D1–D4 + C + F가 살아있으면 학기 결과물로 충분. D5는 차별화 narrative를 mechanism level로 강화하려는 **추가 시도**.

**🚦 게이트 D (전체)**: D1·D2·D3 모듈이 Pipeline과 통합되어 동작 + Gradio UI에서 실사용 가능 + 통합 테스트 통과. D5는 별도 게이트 (위).

---

### E. 야심 옵션 (선택, 선행: D 완료)

> 시간·GPU·사용자 우선순위에 따라 선택적으로 진행. **모두 stretch goal**.

#### E1. Test-time scale 자동 튜닝

- [ ] 변환 후 LPIPS / DINOv2 측정 → scale 조정 → 재생성 (자동 loop)
- [ ] 수렴 조건 정의 (LPIPS 변화 < ε)

#### E2. Iterative refinement loop

- [ ] 자동 평가 → 약한 속성 식별(예: 팔레트 어긋남) → 해당 속성만 재생성
- [ ] 본 시스템의 self-reviewer

#### E3. 개인 LoRA 파인튜닝

- [ ] 사용자 본인 게임 에셋 20–50장으로 경량 LoRA 학습 스크립트
- [ ] `scripts/train_lora.py`
- [ ] **"내 게임 스타일을 학습한 reference"** 데모 — GPT Image가 못 하는 영역

#### E4. Style Bible 산출물

- [ ] Reference → palette·라인 통계·그림자 방향·shading 분포를 JSON으로 저장
- [ ] 재사용 가능한 "Style Bible" 포맷 정의
- [ ] 게임 아티스트 워크플로에 맞는 산출물

---

### F. 검증 (선행: C, D 완료)

**목적**: 차별화 narrative를 정량 + 정성 양쪽으로 입증.

#### F1. Ablation Study

- [ ] A0: SDXL img2img only — `experiments/001_a0_sdxl_img2img/`
- [ ] A1: IP-Adapter only — `experiments/002_a1_ipadapter/`
- [ ] A2: ControlNet only — `experiments/003_a2_controlnet/`
- [ ] A3: IP-Adapter + ControlNet — `experiments/004_a3_full_noproc/`
- [ ] A4: 풀 시스템 + 후처리 일관성 강제(§5.5.1) — `experiments/005_a4_full_postproc_consistency/`
- [ ] A5: A4 + Cross-image attention(§5.5.2, StyleAligned 응용) — `experiments/006_a5_shared_attention/` ⚠️ 시도
- [ ] **A_GPT**: GPT Image 2.0 baseline — `experiments/007_external_gpt_image/`
- [ ] A_naive: IP-Adapter only baseline — `experiments/008_ipadapter_naive/`
- [ ] 결과 비교 리포트 + ADR-010 의사결정 (D5 채택/폐기) 기록

> **체크포인트**: A4가 A0 대비 정량 지표에서 명확한 개선이 없으면 접근 재검토. A5 vs A4 비교로 D5 채택 판정 (게이트 D5 기준). A5가 채택되든 폐기되든, A4가 GPT Image 대비 σ_consistency 우위를 보이면 fallback narrative 가능.

#### F2. 하이퍼파라미터 탐색

- [ ] `ip_adapter_scale` sweep (0.4–1.0, step 0.1)
- [ ] `controlnet_scale` sweep (0.6–1.2)
- [ ] `denoising_strength` sweep (0.3–0.7)
- [ ] 최적 조합을 `configs/best.yaml`에 저장

#### F3. 사용자 스터디

- [ ] 설문 폼 설계 (Google Forms)
- [ ] 샘플 20세트 (Style Unifier vs GPT Image 2.0 — 순서 랜덤화)
- [ ] 참가자 5–10명 모집 (r/gamedev, 국내 인디 디스코드, 본인 네트워크)
- [ ] 3개 질문: style fidelity / content preservation / usability + **batch coherence** (N장이 한 가족인가)
- [ ] 결과 분석 + 카테고리별 보고

#### F4. 본인 게임 적용

- [ ] 현재 스타일 불일치 에셋 10–20장 선정 (캐릭터·사물·아이템 골고루)
- [ ] 배치 일관성 모드로 일괄 변환
- [ ] Before/after 스크린샷
- [ ] 실패 케이스 기록

#### F5. 자동 leaderboard

- [ ] `scripts/build_leaderboard.py` — 모든 ablation + 외부 baseline의 메트릭을 표로 자동 생성
- [ ] README에 자동 갱신 (체크인 시 또는 수동 트리거)

**🚦 게이트 F**:
- A5가 A_GPT 대비 σ_consistency·LPIPS·DINOv2에서 명확한 우위
- 본인 게임 에셋 10장 이상 before/after 데모 보유
- 사용자 스터디 정량 결과 확보

---

## 🚨 주요 게이트·체크포인트 요약

각 게이트에서 막히면 후속 그룹은 의미가 없으므로 우선순위가 가장 높다.

| 게이트 | 기준 | 실패 시 대응 |
|---|---|---|
| **A — 스모크 테스트** | reference 1장 → 변환 결과 1장 | 모델 로딩·VRAM 문제부터 해결. 최악의 경우 Colab Pro 우회. |
| **B — 코어 동작** | RGBA in/out + 시각적으로 스타일 적용·형태 보존 | ControlNet/IP-Adapter scale 재점검. |
| **C — 평가 자동화** | 평가셋 30장에 9개 메트릭 자동 측정 + GPT Image 컬럼 | 메트릭 구현 sanity check (known-good pair). |
| **D — 차별화 기능 통합** | 배치 일관성 + 속성 제어 + Region mask 모두 UI에서 동작 | 모듈 간 인터페이스 정리, 통합 테스트 보강. |
| **F — Ablation A5 우위** | σ_consistency·LPIPS·DINOv2에서 GPT Image 대비 명확 우위 | 배치 일관성 모듈 강화 (palette·lineart 통계 강제 정도 상향). 그래도 안 되면 narrative를 "본인 게임 적용 정성 데모" 중심으로 조정. |

---

## 📝 변경 로그

> 계획 변경 시 여기에 기록.

### 2026-05-12
- 시간 기반(15주차) 일정 폐기 → 의존 순서 기반 작업 큐로 전면 재구성.
- 차별화 narrative 변경: "단일 이미지 품질" → "배치 일관성 + 형태 보존 + 속성 제어 + 재현성".
- GPT Image 2.0을 baseline으로 통합 (그룹 C).
- 배치 일관성 / 속성 제어 / Region masking을 stretch가 아닌 **D그룹 baseline**으로 승격.
- 개인 LoRA 파인튜닝을 야심 옵션(E그룹)으로 추가 (기존 ADR-002에서 제외했던 것).
- **그룹 D5 추가** — StyleAligned 응용 cross-image attention 시도 (시도 중 architectural approach). ADR-010. 실험 결과에 따라 채택/폐기 결정. D1(후처리 baseline)이 fallback.

### 2026-04-24
- 프로젝트 roadmap 초안 작성.

---

## 🔄 회고

> 그룹 게이트 통과 시점에 기록. "무엇을 배웠는가", "무엇이 안 됐는가".

*(아직 기록 없음. 게이트 A 통과 시 시작.)*

### 템플릿

```markdown
### 게이트 X 통과 (YYYY-MM-DD)

**결과**:
- ✅ 완료: ...
- ⚠️ 부분 달성: ...
- ❌ 보류: ...

**배운 것**:
- ...

**다음 그룹에 반영**:
- ...
```

---

## 🔗 관련 문서

- `CLAUDE.md` — 매 세션 컨텍스트 (이 문서 요약 참조)
- `docs/technical_design.md` — Section 8 (작업 큐 상세)
- `docs/technical_design.md` — Section 9 (ADR)
- `experiments/` — 각 실험 결과

---

*Last updated: 2026-05-12 (일정 기반 → 의존 순서 기반 재구성)*
