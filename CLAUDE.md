# CLAUDE.md

> Claude Code가 매 세션 참조하는 컨텍스트. **간결 유지**. 상세는 `docs/` 참조.

---

## 🎯 프로젝트

벡터/일러스트 스타일의 정적 2D 게임 에셋(캐릭터, 사물, 아이템)을 reference 이미지 기준으로 **배치 일관성 + 형태 보존 + 스타일 변환**하는 로컬 vision pipeline.

**입력**: 원본 에셋(RGBA) N장 + reference 이미지 → **출력**: 변환된 에셋(RGBA) N장 (서로 일관된 스타일)

**차별화 narrative** (ADR-007): GPT Image 2.0 등 상용 모델과 raw 품질로 경쟁하지 않는다. 대신 상용 모델이 구조적으로 못 하는 4영역에서 정량 우위를 만든다: (1) 배치 일관성 σ, (2) 수학적 형태 보존, (3) 속성 단위 제어, (4) 재현성 + 로컬 실행.

**Mechanism level 시도** (ADR-010): 후처리 기반 일관성(§5.5.1)을 안전한 baseline으로 두고, 그 위에 StyleAligned 응용 cross-image attention(§5.5.2, 그룹 D5)을 시도한다. **검증 단계 architectural approach** — 실험 결과(A4 vs A5)에 따라 채택/폐기. 미충족 시 fallback narrative로 정직하게 약화. D1(후처리 baseline)이 fallback 보장.

현재 파이프라인은 카테고리별 분기 로직이 없어 캐릭터·사물·아이템 모두 동일한 흐름으로 처리됨. 픽셀 아트, UI/버튼, 배경/타일, 애니메이션/이펙트는 별개 처리가 필요해 도메인 외(ADR-006).

---

## 📍 현재 상태

진행은 시간 기반이 아닌 **의존 순서 기반 작업 큐**(그룹 A–F)로 추적. 상세는 `docs/roadmap.md`.

- **현재 그룹**: **B+C+D 코드 작성 완료** (B 코어 + C 평가 인프라 + D 차별화 기능 — 260개 mock 기반 테스트 통과)
- **다음 게이트**: B 게이트 (Linux 실측 시각 검사) → C 게이트 (run_eval로 30페어×9메트릭 실측) → F1 ablation (A4 vs A5 D5 판정, ADR-010)
- **Blocker**: 없음 (Linux 머신에서 A3·A4 환경 셋업 후 검증)

---

## 🏗️ 절대 원칙

1. **형태 보존이 최우선**. ControlNet을 약화시키는 변경은 두 번 생각하라.
2. **배치 일관성은 차별화의 핵심**. σ_palette·σ_linewidth·σ_shading을 약화시키는 결정은 narrative 자체를 흔든다. 후처리 baseline(§5.5.1)을 먼저 보장하고, shared attention(§5.5.2)은 그 위의 검증 시도 — ADR-010 의사결정 프로토콜 준수.
3. **도메인은 벡터/일러스트 스타일의 정적 2D 객체**. 픽셀 아트, UI/버튼, 배경/타일, 애니메이션/이펙트, 포토리얼은 범위 밖.
4. **RGBA 출력 유지**. 중간에 RGB가 되더라도 최종은 RGBA.
5. **결정론적 재현성**. 실험은 seed 명시 + `experiments/NNN/`에 config와 결과 보존. bit-exact 재생성 가능해야 함.
6. **GPT Image 2.0은 경쟁자가 아니라 평가 baseline**. ablation의 `A_GPT_Image` 컬럼으로 통합(ADR-008).
7. **Scope 엄수**. Unity 플러그인, 대규모 LoRA 학습, 애니메이션 프레임 일관성, SaaS 배포는 out of scope. (개인 LoRA만 그룹 E 야심 옵션으로 허용 — ADR-002 개정.)
8. **오케스트레이터(메인 Claude)만 `.claude/work-log.md`에 기록**. 한 턴 작업이 끝나면 호출한 서브 에이전트들의 보고를 1개 항목으로 총정리해 `## 기록` 헤더 직후에 삽입. 한국 시간(KST, `TZ='Asia/Seoul' date '+%Y-%m-%d %H:%M KST'`), 내림차순. 서브 에이전트는 직접 기록하지 않는다. 사용자가 명시적으로 "기록하지 마"라고 한 경우에만 생략.

---

## 🖥️ 개발 환경

**리눅스(RTX 4070, 12GB) = 유일한 실행 환경.**
- 모든 모델 추론, 실험, 벤치마크는 리눅스에서.
- `enable_model_cpu_offload()` 필수. 해상도 1024 초과 금지 (OOM 시 768로 축소).

**윈도우 / macOS = 코드 작성 전용.**
- 추론 코드를 로컬 실행하지 않는다. mock 또는 tiny 모델로만 단위 테스트.
- 에이전트가 "돌려보자"고 하면 → 리눅스에 커밋하고 리눅스에서 실행하라고 사용자에게 말할 것.

**크로스 플랫폼 코드 규칙**:
- `device` 하드코딩 금지. 경로는 `pathlib.Path` 사용.
- 줄바꿈은 LF (`.gitattributes`에 `* text=auto eol=lf` 설정).

---

## 🛠️ 기술 스택 (변경 금지)

| 영역 | 선택 |
|------|------|
| 언어 / 패키지 | Python 3.11 / **pip + venv** (poetry, uv 금지) |
| 프레임워크 | `diffusers`, `transformers`, `torch` |
| 베이스 모델 | SDXL + Animagine-XL-3.1 (ADR-001) |
| 제어 | IP-Adapter-Plus + ControlNet (lineart, depth) |
| UI | Gradio (Unity 플러그인 금지, ADR-003) |
| 평가 | LPIPS, CLIP, DINOv2, palette EMD + 자체 σ_consistency 메트릭 |
| 외부 baseline | GPT Image 2.0 (ADR-008) |

---

## 🚫 금지 사항

- `requirements.txt`를 임의 수정하지 말 것. 의존성 추가는 사용자 확인 후.
- 체크포인트(.safetensors/.ckpt/.bin)를 git에 커밋 금지.
- `data/`, `experiments/*/outputs/`를 git에 포함 금지.
- 추론 코드에 `print()` 금지. `src.utils.logging.get_logger` 사용.
- 윈도우/맥에서 모델 추론 실행 금지.

---

## 📁 디렉토리 핵심

```
style-unifier/
├── CLAUDE.md, STYLE.md, SETUP.md
├── .claude/agents/            # 서브 에이전트 정의
├── docs/technical_design.md   # 전체 설계 (섹션별 참조)
├── docs/roadmap.md            # 의존 순서 기반 작업 큐 (그룹 A–F)
├── configs/*.yaml             # 실험 설정
├── src/{preprocessing,encoding,generation,postprocessing,evaluation,utils,baselines}/
├── scripts/                   # 데이터 수집, 배치 실행, leaderboard
├── experiments/NNN_name/      # 실험별 폴더
├── data/{raw,processed,eval,cache}/
└── tests/
```

파일 500줄 초과 시 서브 모듈로 분리.

---

## 📝 코드 규칙 (상세는 `STYLE.md`)

- 타입 힌트 + Google-style docstring 필수.
- 임포트: 표준 → 서드파티 → 로컬. 그룹 사이 빈 줄.
- 예외: 구체적 것만 catch. `except Exception:` 금지.
- 설정: 하드코딩 금지. `configs/*.yaml`에서 로드.

---

## 🤖 멀티 에이전트

| 에이전트 | 역할 |
|---------|------|
| `implementer` | 모듈/함수 구현 |
| `reviewer` | 리뷰, 리팩토링 제안 |
| `tester` | pytest 테스트 작성 |
| `researcher` | 논문/라이브러리 조사 |
| `evaluator` | 실험 실행, 결과 분석 |

**호출 순서**:
- 새 기능: `implementer` → `tester` → `reviewer`
- 실험: `evaluator` 단독
- 설계 결정: `researcher` 조사 → 사용자와 논의
- **위험한 architectural 시도 (예: 그룹 D5 StyleAligned)**: `researcher` (선행 조사) → `implementer` (구현) → `tester` (attention shape 검증) → `reviewer` (ControlNet 충돌·VRAM·ADR-010 점검) → `evaluator` (A4 vs A5 실측으로 게이트 판정) → 사용자 (채택/폐기 결정 + narrative 조정)

상세는 `.claude/agents/*.md`.

---

## 🚦 작업 체크리스트

**시작 전**: 현재 그룹(A–F) 범위인가? 선행 게이트 통과했나? 관련 ADR 있는가? 새 의존성 필요한가 (있으면 사용자 확인)?
**종료 전**: 타입 힌트 + docstring? 테스트 추가? 실험이면 `experiments/NNN/`에 config + seed + git hash 로그?

---

## 🚫 자주 하는 실수

*(개발 중 반복 발견되면 여기 추가. 현재 없음.)*

---

## 📚 참조 우선순위

1. 이 파일 → 2. `docs/roadmap.md` → 3. `STYLE.md` → 4. `docs/technical_design.md` (필요 섹션만) → 5. `configs/default.yaml`

**참조 금지**: `data/raw/`, `experiments/*/outputs/`, `venv/`, `__pycache__/`.

---

*Last updated: 2026-05-12 (차별화 narrative 재구성 + 시간 기반 일정 폐기 + 의존 순서 기반 작업 큐 + StyleAligned 응용 시도 — ADR-007/008/009/010)*
