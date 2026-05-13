# 작업 기록 (Work Log)

> **오케스트레이터(메인 Claude)** 가 한 턴의 작업을 마칠 때 `## 기록` 헤더 직후에 1개 항목으로 총정리해 추가한다.
> 시간은 한국 시간(KST, `TZ='Asia/Seoul' date '+%Y-%m-%d %H:%M KST'`), **내림차순** (최신이 위).

## 항목 형식

```markdown
### YYYY-MM-DD HH:MM KST — <한 줄 요약>

- 작업 내용 / 호출한 에이전트 / 핵심 결과 / 다음 단계를 3–6 bullet로 정리
- 변경 파일은 필요 시만 명시
```

## 기록

<!-- 새 항목은 이 줄 바로 아래에 추가 -->

### 2026-05-13 23:33 KST — 그룹 A 인프라 셋업 + 그룹 D 게이트 코드 통합 완료

- **그룹 A 인프라 셋업 (Linux RTX 4070 실측)**: venv 활성화 + PyTorch 2.1.2+cu121 + xformers + requirements.txt 설치. `nvidia-smi` 드라이버 570.211 + CUDA 12.8 확인, `torch.cuda.is_available()` True. `huggingface-cli login` 완료. 모델 다운로드 [scripts/download_models.py](scripts/download_models.py)에 `ignore_patterns` 필드 추가 + diffusers ignore 세트(`*.onnx`, `*.fp16.safetensors`, `*.bin`, `*.msgpack`, `*flax*` 등) 적용 — 첫 시도에서 SDXL 60GB·IP-Adapter 18GB·CLIP 7.4GB로 부풀어 ~130GB 받았다가, allow/ignore 패턴 정교화 후 ~38GB로 정상화. IP-Adapter는 `sdxl_models/ip-adapter-plus_sdxl_vit-h.safetensors` + `models/image_encoder/{config.json,model.safetensors}` 3파일로 제한, CLIP은 `open_clip_*` 제외 transformers 포맷만. `scripts/smoke_test.py --full` 6/6 PASS (SDXL이 cuda에 fp16 로드 성공).
- **그룹 D 게이트 코드 통합 (Plan `/home/zop1234/.claude/plans/roadmap-md-tender-liskov.md`)**:
  - **implementer Round 1**: [src/generation/pipeline.py](src/generation/pipeline.py) `transform()` 시그니처에 `scales`/`attribute_mode`/`region_mask` keyword-only 추가, `transform_batch()` 위임 메서드 추가. [src/generation/pipeline_batch.py](src/generation/pipeline_batch.py) 신규 — `run_batch_transform()` + `_shared_attention_scope` `@contextmanager` (try/finally로 `remove_shared_attention` 보장). D5 batched denoise는 v1 미구현(`_D5_BATCHED_DENOISE_IMPLEMENTED = False` 상수 + sequential fallback + WARNING 로그). D4 multi-ref는 `references[0]` fallback + WARNING.
  - **implementer Round 2**: [app/gradio_app.py](app/gradio_app.py) Single Tab에 attribute_mode Radio + ImageEditor region mask 추가. [app/gradio_batch_tab.py](app/gradio_batch_tab.py) 신규 — Batch Tab(Files×2, Gallery, enforce_consistency checkbox, attribute_mode Radio, Experimental Accordion 안 shared_attention checkbox + share_layers textbox). `_extract_region_mask_from_editor` 헬퍼로 ImageEditor dict → float32 [0,1] (H,W) mask 변환.
  - **tester**: 47개 신규 테스트 (`test_generation_pipeline_batch.py` 신규 20 + `test_generation_pipeline.py` +13 + `test_app_gradio.py` +14). mock 전략: pipeline_batch는 lazy import이므로 소스 모듈 경로 patch (`src.encoding.batch_consistency.extract_style_statistics` 등), `_shared_attention_scope`는 `patch.object`로 컨텍스트 매니저 교체. **329 tests passed, 0 regression**.
  - **reviewer**: Critical 4(macOS 절대경로 + transform/run_batch_transform/transform_batch_handler 함수 길이 STYLE.md §2.2 위반), Important 6, Nit 7 발견.
  - **implementer Round 3 (Critical/Important fix)**: C1 macOS 경로 → `Path(__file__).parent.parent / "configs" / "default.yaml"`. C2/C3/C4 함수 분리 — `pipeline.py`에 헬퍼 5종 추가했다가 616줄로 부풀어 500줄 위반 발생.
  - **implementer Round 4 (500줄 위반 fix)**: [src/generation/pipeline_helpers.py](src/generation/pipeline_helpers.py) 신규 (182줄) — `build_controlnet_args` / `pack_controlnet_args` / `read_sampling_cfg` / `resolve_effective_scales` / `apply_region_mask_if_needed` 자유 함수로 분리(언더스코어 prefix 제거해 `reportPrivateImportUsage` 차단). `_run_inference`는 `pipeline_loader.run_inference_step`으로 인라인. pipeline.py 616 → **451줄**, CLAUDE.md 500줄 원칙 준수. I1(selective fallback docstring), I2(_multi_ref_strategy dead Dropdown → Markdown 안내), I4(ValueError catch 추가) 함께 처리. 보류: I3/I5/I6/Nit 7건은 다음 plan.
- **검증**: ruff format/check 통과, pyright 0 errors, `pytest -m "not slow"` **329 passed, 회귀 0**. 파일별 LoC — pipeline.py 451 / pipeline_loader.py 248 / pipeline_batch.py 390 / pipeline_helpers.py 182 / gradio_app.py 362 / gradio_batch_tab.py 310, 모두 500줄 한계 내.
- **다음**: Linux RTX 4070에서 (1) `python app/gradio_app.py` Single Tab D2/D3 + Batch Tab D1/D5 시각 검사(B 게이트), (2) D5 batched denoise 실제 구현 + N=4 VRAM·ControlNet 충돌 점검(Linux 실측), (3) C 게이트 — Kenney/본인 에셋 수집 후 `scripts/collect_data.py` + `scripts/build_eval_pairs.py` + `scripts/run_eval.py` 30페어×9메트릭, (4) F1 ablation A0~A5 + A_GPT + A_naive → ADR-010 D5 채택/폐기 판정.

### 2026-05-13 00:22 KST — 그룹 C(평가 인프라) + D(차별화 기능) 병렬 코드 작성 완료

- **Round 1 병렬 (implementer ×2)**: C1+C2 `src/evaluation/{metrics,consistency}.py` (430+294줄, 메트릭 5종 + σ_consistency 3종, lazy 모델 로딩) / D1 `src/encoding/batch_consistency.py` (470줄, §5.5.1 후처리 baseline + `StyleStatistics`).
- **Round 2 병렬 (implementer ×2)**: D2+D3 `src/postprocessing/{attribute_control,region_mask}.py` (236+277줄, StyleAttribute enum / 5 preset / focal+face_preserve placeholder) / C4 `src/baselines/{base,gpt_image,ipadapter_naive}.py` (146+216+175줄, BaselineWrapper ABC + CachedBaselineMixin + max_calls 가드).
- **Round 3a (implementer ×2)**: D4 `src/encoding/multi_ref.py` (230줄, mean/weighted/first/manual) / C5 `scripts/run_eval.py` + `_eval_helpers.py` (489+329줄, 9 메트릭 자동화 + summary.md).
- **Round 3b (implementer ×2)**: D5 `src/encoding/shared_attention.py` (423줄, `SharedKVAttnProcessor` + apply/remove + validate_batch, ⚠️ ADR-010 검증 단계) / C3 `scripts/collect_data.py` + `build_eval_pairs.py` + `_pair_strategies.py` + `data/DATA_CARD.md` (480+366+182+307줄, license filter + CLIP auto-tag + stratified 50/25/25).
- **tester**: 144개 신규 테스트 (총 260 passed, 22 skipped는 macOS skimage/torchvision 의존성). mock 기반(`_CLIP_CACHE`, `_LPIPS_CACHE`, `StyleEncoder`, `Attention`, OpenAI API).
- **reviewer**: Critical 3, Important 7, Nit 6 발견. **Critical 전부 처리**: C1 `distance_transform_edt` 방향 버그(두 파일에서 sigma_linewidth 항상 0이었음) → `distance_transform_edt(edge_mask)` 로 수정, C2 GPT API reference 미전달 → WARNING + TODO 강화, C3 consistency chunk reference-aware 그룹화 + summary.md 주의 사항 (implementer 위임). **Important 4건 처리**: I1 D5 `remove_shared_attention`이 원본 processor 복원(implementer 위임), I2 validate_batch 해상도 검증 추가(implementer 위임), I3 CLIP 모델명 Linux TODO, I4 `cache_embedding=True` → `False` 기본값(CLAUDE.md 모델 추론 금지 원칙), I6 collect_data 중복 dry_run 조건 제거. **미처리**: I5 run_eval 함수 분리(시간 절약), I7 StyleEncoder N회 인스턴스화 docstring 강조, Nit 6건.
- **검증**: ruff All checks passed, pyright 0 errors (warning만 — 외부 스텁 부재), pytest **260 passed, 22 skipped, 0 failed** in 3.31s. regression 0.
- **다음**: Linux RTX 4070에서 (1) 그룹 A 잔여(모델 다운로드 ~20GB), (2) B 게이트 시각 검사, (3) `python scripts/run_eval.py --config configs/default.yaml`로 C 게이트, (4) Pipeline 통합 — `StyleUnificationPipeline.transform_batch()` + attribute_control 라우팅 + D5 shared attention 토글, (5) F1 ablation A0~A6 + A_GPT + A_naive → ADR-010 D5 채택/폐기 판정.

### 2026-05-12 23:08 KST — 그룹 B 코어 파이프라인 코드 작성 완료 (Linux 실측 검증 잔여)

- **B1 전처리 (implementer)**: `src/preprocessing/{bg_removal,lineart,palette}.py` (174+142+83줄). RMBG-1.4 RGBA fast path, controlnet_aux lineart_anime/canny, sklearn k-means 결정론적.
- **B4 후처리 + configs (implementer)**: `src/postprocessing/{alpha_restore,palette_quantize}.py` (115+149줄), `configs/default.yaml` (65줄). gaussian feathering, soft palette snap + alpha 보존.
- **B2 인코딩 (implementer)**: `src/encoding/ip_adapter_wrapper.py` (359줄) — `StyleEncoder` 클래스, IP-Adapter Plus `last_hidden_state` 경로, SHA-256 픽셀 캐시, `encode_multi(mean/weighted/first)`.
- **B3 파이프라인 (implementer + 분리)**: `src/generation/{pipeline,pipeline_loader}.py` (427+188줄). lazy 로딩, `transform()` RGBA in/out, `enable_model_cpu_offload`/`vae_slicing`/xformers, seed→generator, `return_intermediates` dict.
- **B5 Gradio UI (implementer)**: `app/gradio_app.py` (264줄) — 3패널, config 경로별 캐시 + evict, gr.Progress 5단계, queue safety는 향후 처리.
- **tester**: 89개 신규 테스트 (총 126개 통과, 3.18s). mock 기반(`from_pretrained`, `_load_rmbg_model`, `_load_detector`, `gaussian_filter`, `StyleUnificationPipeline.__init__` 등).
- **reviewer**: Critical 2 / Important 7 / Nit 5 발견. 처리: C1 StyleEncoder dead code 제거(_load_pipeline에서 분리, D 진입 시 재연결 명시), C2 `_postprocess`가 `(result, palette)` 튜플 반환으로 `extract_palette` 이중 호출 제거, I3 yaml에 `palette_strength`·`lineart_detector` 키 추가, I5 `assert` 6곳 → `_require_pipe_loaded()` + `raise RuntimeError`, I1/I2 pipeline.py 542줄 → 427줄 + 새 `pipeline_loader.py` 188줄.
- **잔여 (Important 미처리)**: I4(bg_removal device 명시적 전달), I6(MPS generator fallback), I7(Gradio queue 안전성). 모두 그룹 D 진입 직전 또는 Linux 실측 시 처리.
- **검증**: ruff All checks passed, pyright 0 errors, pytest 126/126 통과 (regression 0).
- **다음**: Linux RTX 4070에서 (1) `python scripts/download_models.py` ~20GB, (2) `python scripts/smoke_test.py --full`, (3) `python app/gradio_app.py`로 실제 reference→output 변환 시각 검사. 게이트 B 판정 후 그룹 C(평가 인프라) 또는 D(차별화) 병렬 진입.

### 2026-05-12 22:16 KST — 그룹 A 인프라 코드 작성 완료 (A3·A6-full 잔여)

- **A1**: `src/baselines/__init__.py` 신설 — 패키지 자리 확보 (구현체는 그룹 C4).
- **A5 구현 (implementer)**: `src/utils/{logging,device,repro,experiment}.py` 4개 모듈. `get_logger`/`get_device`/`get_dtype`/`set_seed`/`make_generator`/`set_deterministic`/`make_experiment_dir`/`record_environment`/`init_experiment`. `__init__.py`에서 공개 API re-export.
- **A5 테스트 (tester)**: `tests/test_utils_*.py` 4파일, 총 38 테스트. macOS 환경에서 pytest 1.03초 만에 38/38 통과.
- **A4 (implementer)**: `scripts/download_models.py` — 8개 모델 (SDXL, Animagine-XL-3.1, IP-Adapter, ControlNet canny/depth, RMBG-1.4, CLIP ViT-H, DINOv2) `huggingface_hub.snapshot_download` 기반. `--dry-run`/`--only` 지원.
- **A6 (implementer)**: `scripts/smoke_test.py` — 6단계 검증 (env/device/repro/diffusers import/SDXL 로딩(`--full`)/experiment dir). reference→output 단계는 B3 의존이라 보류.
- **reviewer**: Critical 0, Important 5(I1~I5) 발견. 모두 처리 — ID 전략 max+1 변경(+ 테스트 수정), `get_dtype` 중복 제거, `set_deterministic` export, diffusers private import `pyright: ignore`, CUBLAS 부작용 docstring 명시.
- **검증**: ruff format/check 0, pyright 0 errors (외부 스텁 부재 warning 2 무시), pytest 38/38 PASS.
- **잔여**: A3(Linux GPU 환경 검증)·A4 실제 다운로드(~20GB)·A6 full(B3 pipeline 의존)은 Linux 머신에서. 코드 작성은 macOS에서 종료.
- **다음**: 그룹 B (코어 파이프라인) 진입 — `src/preprocessing/{bg_removal,lineart,palette}.py` → `src/encoding/ip_adapter_wrapper.py` → `src/generation/pipeline.py`.
