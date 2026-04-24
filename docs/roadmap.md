# docs/roadmap.md

> **살아있는 문서**. 매주 또는 진행 상황 변화 시 갱신.
> CLAUDE.md의 "현재 상태" 섹션이 이 문서를 요약 참조함.

**프로젝트 시작**: 2026-04-24 (Fri, Week 0)
**Week 1 시작**: 2026-04-27 (Mon)
**예상 종료**: 2026-08-09 (Sun, Week 15 말)

---

## 🎯 현재 상태 대시보드

> **매주 월요일 업데이트**. 이 섹션을 보고 바로 작업 시작 가능해야 함.

### 오늘 상태 (2026-04-24)

- **현재 주차**: Week 0 (Prep)
- **현재 Phase**: Foundation 시작 전
- **오늘 할 일**: 문서 정리 완료 → GitHub 저장소 초기화
- **Blocker**: 없음
- **이번 주 목표**: Week 1 시작 전 저장소 초기화 및 환경 구축 준비 완료

### 다음 3일 (일간 상세)

- [ ] **2026-04-24 (Fri)**: 저장소 초기화, `.gitignore`/`.gitattributes` 작성, 문서 모두 커밋
- [ ] **2026-04-25 (Sat)**: 리눅스 머신 SETUP.md 따라 NVIDIA 드라이버 설치, 재부팅 검증
- [ ] **2026-04-26 (Sun)**: Python 3.11 가상환경, PyTorch CUDA 설치, `torch.cuda.is_available()` 확인

### 이번 주 미해결 질문

- [ ] Kenney.nl CC0 에셋 라이선스 실제 확인 (배포용 재사용 가능?)
- [ ] Animagine-XL-3.1이 아닌 대체 일러스트 체크포인트 존재 여부

---

## 📊 전체 진행 현황

| Phase | 주차 | 목표 | 상태 | 진행률 |
|-------|------|------|------|--------|
| **Foundation** | Week 1-3 | 환경 구축, Baseline | ⏸️ 대기 | 0% |
| **Data & Evaluation** | Week 4-6 | 데이터 수집, 평가셋, Baseline 평가 | ⏸️ 대기 | 0% |
| **Refinement** | Week 7-10 | Postprocessing, HP 탐색, Multi-ref, 안정화 | ⏸️ 대기 | 0% |
| **Validation & Delivery** | Week 11-15 | 사용자 스터디, 게임 적용, 문서화, 발표 | ⏸️ 대기 | 0% |

**상태 범례**: ⏸️ 대기 / 🔵 진행 중 / ✅ 완료 / ⚠️ 지연 / ❌ 포기

### 주차별 요약 표

| Week | 날짜 | 목표 | 상태 | 완료일 |
|------|------|------|------|--------|
| 0 | 04-24 ~ 04-26 | 준비 | 🔵 진행 중 | - |
| 1 | 04-27 ~ 05-03 | 환경 구축 | ⏸️ | - |
| 2 | 05-04 ~ 05-10 | Baseline 구현 | ⏸️ | - |
| 3 | 05-11 ~ 05-17 | 전처리 파이프라인 | ⏸️ | - |
| 4 | 05-18 ~ 05-24 | 데이터 수집 | ⏸️ | - |
| 5 | 05-25 ~ 05-31 | 평가셋 & 메트릭 | ⏸️ | - |
| 6 | 06-01 ~ 06-07 | Baseline 평가 (Ablation A0-A3) | ⏸️ | - |
| 7 | 06-08 ~ 06-14 | Postprocessing | ⏸️ | - |
| 8 | 06-15 ~ 06-21 | 하이퍼파라미터 탐색 | ⏸️ | - |
| 9 | 06-22 ~ 06-28 | Multi-reference (Stretch) | ⏸️ | - |
| 10 | 06-29 ~ 07-05 | 안정화 | ⏸️ | - |
| 11 | 07-06 ~ 07-12 | 사용자 스터디 준비 | ⏸️ | - |
| 12 | 07-13 ~ 07-19 | 사용자 스터디 진행 | ⏸️ | - |
| 13 | 07-20 ~ 07-26 | 본인 게임 적용 | ⏸️ | - |
| 14 | 07-27 ~ 08-02 | 문서화 | ⏸️ | - |
| 15 | 08-03 ~ 08-09 | 발표 & 마무리 | ⏸️ | - |

---

## Phase 1: Foundation (Week 1-3)

### Week 0 — Prep (2026-04-24 ~ 04-26)

**목표**: 프로젝트 시작 준비 완료.

**일간 태스크** (현 시점 일간 상세):

- [x] **2026-04-24 (Fri)**:
    - [x] 기획서 작성 (technical_design.md)
    - [x] CLAUDE.md, STYLE.md, SETUP.md 작성
    - [x] 5개 에이전트 정의 파일 작성
    - [ ] GitHub 저장소 초기화
    - [ ] `.gitignore`, `.gitattributes` 작성
    - [ ] 문서 커밋
    - [ ] Notion 페이지 구조화 (계획 옮기기)

- [ ] **2026-04-25 (Sat)**:
    - [ ] 리눅스 Ubuntu 22.04 클린 설치 확인 (이미 되어 있으면 skip)
    - [ ] NVIDIA 드라이버 설치 (`sudo ubuntu-drivers autoinstall`)
    - [ ] 재부팅 후 `nvidia-smi` 확인
    - [ ] Mac/Windows 환경 Python 3.11 설치 확인

- [ ] **2026-04-26 (Sun)**:
    - [ ] 리눅스에서 Python 3.11 venv 생성
    - [ ] PyTorch 2.1.2 + CUDA 12.1 설치
    - [ ] `torch.cuda.is_available()` 확인
    - [ ] HuggingFace 토큰 발급 및 환경변수 설정

**완료 조건**: Week 1 월요일 아침에 바로 코드 작성 시작 가능한 상태.

---

### Week 1 — 환경 구축 (2026-04-27 ~ 05-03)

**목표**: 스모크 테스트 통과. 리눅스에서 모델 1장 생성 성공.

**주간 체크리스트**:

- [ ] 저장소 디렉토리 구조 생성 (src/, tests/, configs/, ...)
- [ ] `requirements.txt` 작성 및 설치
- [ ] `configs/default.yaml` 초안
- [ ] SDXL + IP-Adapter + ControlNet 모델 다운로드 (~20GB)
- [ ] `src/utils/device.py`, `src/utils/logging.py` 작성
- [ ] 첫 생성 스모크 테스트 성공 (reference 1장으로 1장 변환)
- [ ] Gradio UI 최소 화면 (input/reference/output 3개)

**일간 상세** (Week 0 말에 세분화):

- [ ] Mon: ...
- [ ] Tue: ...
- [ ] Wed: ...
- [ ] Thu: ...
- [ ] Fri: ...

> Week 0 말에 위 태스크를 5일로 분해할 것.

**에이전트 활용 계획**:
- `implementer`: src/utils 모듈, 스모크 테스트 스크립트
- `researcher`: 모델 다운로드 스크립트 모범 사례 조사 (필요시)

**완료 조건**:
- [ ] `python scripts/smoke_test.py` → All checks passed
- [ ] Gradio UI 접속 가능

**체크포인트 (⚠️ 중요)**:
> Week 1 말까지 베이스라인 1장 변환 성공하지 못하면 **프로젝트 생존 위기**. 
> 모델 로딩/인프라 문제 해결에 최우선 집중. 
> 최악의 경우 Colab Pro 임시 환경으로 우회하고 로컬 환경은 병행 수정.

---

### Week 2 — Baseline 구현 (2026-05-04 ~ 05-10)

**목표**: `StyleUnificationPipeline` 최소 동작. IP-Adapter 단독 변환 성공.

**주간 체크리스트**:

- [ ] `src/generation/pipeline.py`의 `StyleUnificationPipeline` 클래스 구현
- [ ] IP-Adapter 단독 변환 → 결과 샘플 10장 확보
- [ ] ControlNet(lineart) 추가 → 형태 보존 개선 확인
- [ ] Gradio UI에서 실제 변환 동작 확인
- [ ] 초기 관찰 사항 기록 (`experiments/000_week2_observations/`)

**에이전트 활용**:
- `implementer`: 파이프라인 구현
- `tester`: pipeline에 대한 단위 테스트
- `reviewer`: 초기 구현 품질 점검

**완료 조건**:
- [ ] `pipeline.transform(source, reference)` 호출로 이미지 반환
- [ ] 출력이 RGBA인지 자동 검증
- [ ] pytest -m "not slow" 통과

---

### Week 3 — 전처리 파이프라인 (2026-05-11 ~ 05-17)

**목표**: 전처리 모듈 완성. 입력 이미지에 대한 적절한 조건 신호 생성.

**주간 체크리스트**:

- [ ] `src/preprocessing/bg_removal.py` (RMBG-1.4 wrapper)
- [ ] `src/preprocessing/lineart.py` (controlnet_aux wrapper)
- [ ] `src/preprocessing/palette.py` (k-means)
- [ ] 각 모듈 단위 테스트
- [ ] Gradio UI에 전처리 결과 미리보기 추가 (선택)

**에이전트 활용**:
- `implementer`: 3개 전처리 모듈
- `tester`: 각 모듈 단위 테스트 (각 5-7개)
- `reviewer`: 완료 후 리뷰

**완료 조건**:
- [ ] 전처리 모듈 3개 모두 동작
- [ ] 단위 테스트 커버리지 > 80%
- [ ] Pipeline에 통합되어 end-to-end 동작

---

## Phase 2: Data & Evaluation (Week 4-6)

### Week 4 — 데이터 수집 (2026-05-18 ~ 05-24)

**목표**: CC0 게임 에셋 500-1000장 수집, 라벨링.

**주간 체크리스트**:

- [ ] `scripts/collect_data.py` 작성 (Kenney.nl 크롤)
- [ ] 라이선스 필터링 스크립트
- [ ] CLIP classifier로 자동 태깅
- [ ] 수동 검수 50장
- [ ] 데이터 카드 초안 (`data/DATA_CARD.md`)

**에이전트 활용**:
- `researcher`: 라이선스 이슈, 크롤링 모범 사례
- `implementer`: 수집/필터링 스크립트
- `tester`: 필터링 로직 단위 테스트

**완료 조건**:
- [ ] `data/raw/` 에 500+ 에셋
- [ ] 라이선스 검증 완료
- [ ] 태그 라벨 포함 메타데이터 JSON

> **리스크**: 데이터 수집 시간 과다 소요 가능. 엄격히 주차 내 끝낼 것. 부족하면 stretch로 이동.

---

### Week 5 — 평가셋 & 메트릭 (2026-05-25 ~ 05-31)

**목표**: 정량 평가 파이프라인 완성.

**주간 체크리스트**:

- [ ] 합성 paired 평가셋 30 세트 구축
- [ ] `src/evaluation/metrics.py` 5개 메트릭 구현
    - [ ] CLIP style similarity
    - [ ] LPIPS structure
    - [ ] DINOv2 identity
    - [ ] Palette distance
    - [ ] Gram matrix distance
- [ ] 메트릭 단위 테스트
- [ ] 평가 자동화 스크립트 (`scripts/run_eval.py`)

**에이전트 활용**:
- `implementer`: 메트릭 구현
- `tester`: 메트릭 단위 테스트
- `researcher`: 각 메트릭의 구현 세부사항

**완료 조건**:
- [ ] 5개 메트릭 모두 동작 및 테스트 통과
- [ ] 평가셋 30 세트 `data/eval/` 에 저장
- [ ] `python scripts/run_eval.py --config configs/default.yaml` 실행 가능

---

### Week 6 — Baseline 평가 (2026-06-01 ~ 06-07)

**목표**: Ablation A0~A3 실행, 결과 분석.

**주간 체크리스트**:

- [ ] A0 실험: SDXL img2img only (`experiments/001_*/`)
- [ ] A1 실험: IP-Adapter only (`experiments/002_*/`)
- [ ] A2 실험: ControlNet only (`experiments/003_*/`)
- [ ] A3 실험: IP-Adapter + ControlNet (`experiments/004_*/`)
- [ ] 결과 비교 리포트
- [ ] 실패 케이스 수집

**에이전트 활용**:
- `evaluator`: 4개 실험 실행 및 분석

**완료 조건**:
- [ ] 4개 실험 결과 JSON + summary.md
- [ ] A3가 A0-A2 대비 개선됨 확인 (기본 가설)
- [ ] 주요 실패 패턴 식별

**체크포인트 (⚠️ 중요)**:
> Week 6 말까지 A3가 A0 대비 정량 지표에서 명확한 개선이 없으면 접근 재검토 필요.
> ControlNet 세팅, IP-Adapter scale 등 근본 점검.
> 개선이 있으면 Phase 3 진행.

---

## Phase 3: Refinement (Week 7-10)

### Week 7 — Postprocessing (2026-06-08 ~ 06-14)

**목표**: 후처리 모듈로 A4 실험 실행.

**주간 체크리스트**:

- [ ] `src/postprocessing/alpha_restore.py`
- [ ] `src/postprocessing/palette_quantize.py`
- [ ] `src/postprocessing/quality_check.py`
- [ ] Pipeline에 통합
- [ ] A4 실험 실행 (`experiments/005_*/`)
- [ ] A3 vs A4 비교

**에이전트 활용**:
- `implementer`, `tester`, `reviewer`: 구현/테스트/리뷰
- `evaluator`: A4 실험

**완료 조건**:
- [ ] 3개 후처리 모듈 동작
- [ ] A4가 A3 대비 팔레트/식별성 개선 확인

---

### Week 8 — 하이퍼파라미터 탐색 (2026-06-15 ~ 06-21)

**목표**: 핵심 하이퍼파라미터 최적값 확정.

**주간 체크리스트**:

- [ ] `ip_adapter_scale` sweep (0.4-1.0, step 0.1)
- [ ] `controlnet_scale` sweep
- [ ] `denoising_strength` sweep
- [ ] 최적 조합을 `configs/best.yaml`에 저장

**에이전트 활용**:
- `evaluator`: sweep 실행 및 결과 분석

**완료 조건**:
- [ ] 3개 파라미터 최적값 확정
- [ ] `configs/best.yaml` 생성

---

### Week 9 — Multi-reference (Stretch Goal) (2026-06-22 ~ 06-28)

**목표**: 다중 reference 지원 (stretch). **시간 부족하면 skip 가능**.

**주간 체크리스트**:

- [ ] `src/encoding/multi_ref.py` (mean strategy)
- [ ] A5 실험 (`experiments/006_*/`)
- [ ] Single-ref vs Multi-ref 비교

> **결정 지점**: Week 8 말에 진행 속도 평가. 지연되었다면 Week 9를 안정화로 전환.

---

### Week 10 — 안정화 (2026-06-29 ~ 07-05)

**목표**: 프로덕션 품질로 정리. 버그 수정.

**주간 체크리스트**:

- [ ] 엣지 케이스 테스트 (작은 에셋, 단색 reference, 비정형 비율)
- [ ] 메모리 누수 검증 (100장 배치 돌려보기)
- [ ] 에러 핸들링 강화
- [ ] `reviewer` 에이전트로 전체 코드베이스 리뷰
- [ ] 리뷰 결과 반영

**에이전트 활용**:
- `reviewer`: 전체 코드베이스 리뷰
- `implementer`: 리뷰 피드백 반영

**완료 조건**:
- [ ] 50장 배치 처리가 메모리 누수 없이 완료
- [ ] `ruff check`, `pyright`, `pytest` 전부 통과

**체크포인트 (⚠️ 중요)**:
> Week 10 말에 본인 게임 에셋에 적용 가능한 품질 확보됐는가?
> 불가능하면 Phase 4에서 "완전 자동화" 대신 "반자동 + 수동 보정" 포지셔닝으로 변경.

---

## Phase 4: Validation & Delivery (Week 11-15)

### Week 11 — 사용자 스터디 준비 (2026-07-06 ~ 07-12)

- [ ] 설문 폼 설계 (Google Forms)
- [ ] 샘플 20세트 준비 (baseline vs full)
- [ ] 참가자 모집 (5-10명 목표)

### Week 12 — 사용자 스터디 진행 (2026-07-13 ~ 07-19)

- [ ] 데이터 수집
- [ ] 결과 분석

### Week 13 — 본인 게임 적용 (2026-07-20 ~ 07-26)

- [ ] 현재 게임의 스타일 불일치 에셋 10-20장 선정
- [ ] 일괄 변환
- [ ] Before/after 스크린샷
- [ ] 실패 케이스 기록

### Week 14 — 문서화 (2026-07-27 ~ 08-02)

- [ ] README 작성
- [ ] API 문서 (docstring → mkdocs)
- [ ] 실험 결과 정리 (tables, figures)
- [ ] 최종 보고서 초안

### Week 15 — 발표 & 마무리 (2026-08-03 ~ 08-09)

- [ ] 발표 자료
- [ ] 데모 영상
- [ ] GitHub 공개
- [ ] 최종 제출

---

## 🚨 주요 체크포인트 요약

| 시점 | 기준 | 실패 시 대응 |
|------|------|-------------|
| Week 1 말 | 스모크 테스트 통과 | Colab Pro로 우회, 로컬 환경 병행 수정 |
| Week 6 말 | A3 > A0 정량 개선 | 접근법 재검토, ControlNet 세팅 점검 |
| Week 10 말 | 프로덕션 품질 | 포지셔닝 변경 (반자동 + 수동 보정) |
| Week 13 말 | 실제 게임 적용 가능 | 포트폴리오 범위 조정 |

---

## 📝 변경 로그

> 계획 변경 시 여기에 기록. 나중에 "왜 Week 9를 skip 했지?" 같은 질문에 답할 수 있도록.

### 2026-04-24
- 프로젝트 roadmap 초안 작성.
- Week 1 시작은 2026-04-27(월)로 확정.
- Week 0(2026-04-24 ~ 04-26)은 준비 기간으로 설정.

---

## 🔄 주차 회고 (Retrospective)

> 주차 종료 시 간단히 기록. "무엇을 배웠는가", "무엇이 안 됐는가".

*(아직 기록 없음. Week 1 종료 시 시작.)*

### 템플릿

```markdown
### Week N 회고 (YYYY-MM-DD)

**계획 대비 결과**: 
- ✅ 완료한 것: ...
- ⚠️ 지연된 것: ...
- ❌ 포기한 것: ...

**배운 것**:
- ...

**다음 주 반영**:
- ...
```

---

## 🔗 관련 문서

- `CLAUDE.md` — 매 세션 컨텍스트 (이 문서 요약 참조)
- `docs/technical_design.md` — Section 8 (원본 15주 계획)
- `docs/technical_design.md` — Section 9 (ADR)
- `experiments/` — 각 실험 결과

---

*Last updated: 2026-04-24*
*Next review: 2026-05-03 (Week 1 종료 시)*
