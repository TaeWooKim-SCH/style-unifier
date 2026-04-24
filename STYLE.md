# STYLE.md

> 프로젝트 코드 스타일 가이드. 에이전트와 사람 모두 이 규칙을 따른다.
> CLAUDE.md가 "매 세션 참조"라면, 이 문서는 "코드 작성 시 참조".

**도구 체인**: Ruff (포맷 + 린트) · Pyright (타입) · pytest (테스트) · Google-style docstring

---

## 목차

1. [자동화 도구 설정](#1-자동화-도구-설정)
2. [파일 구조](#2-파일-구조)
3. [네이밍 컨벤션](#3-네이밍-컨벤션)
4. [타입 힌트](#4-타입-힌트)
5. [Docstring](#5-docstring)
6. [에러 처리](#6-에러-처리)
7. [로깅](#7-로깅)
8. [설정 · 경로 · 상수](#8-설정--경로--상수)
9. [테스트](#9-테스트)
10. [ML/PyTorch 특수 규칙](#10-mlpytorch-특수-규칙)
11. [Commit & PR](#11-commit--pr)

---

## 1. 자동화 도구 설정

**지킬 수 없는 규칙은 자동화로 강제한다.** 사람이 기억해야 할 규칙은 최소화.

### 1.1 Ruff (포맷 + 린트)

`pyproject.toml`에 다음을 포함:

```toml
[tool.ruff]
line-length = 100
target-version = "py310"

[tool.ruff.lint]
select = [
    "E", "W",    # pycodestyle
    "F",         # pyflakes
    "I",         # isort (임포트 순서)
    "N",         # pep8-naming
    "UP",        # pyupgrade
    "B",         # bugbear (흔한 버그 패턴)
    "SIM",       # simplify
    "RUF",       # Ruff 고유 규칙
    "PTH",       # pathlib 사용 강제
]
ignore = [
    "E501",      # line too long (formatter가 처리)
]

[tool.ruff.lint.pydocstyle]
convention = "google"

[tool.ruff.format]
quote-style = "double"
indent-style = "space"
```

**실행**:
```bash
ruff format .     # 포맷
ruff check .      # 린트
ruff check --fix  # 자동 수정 가능한 것 수정
```

### 1.2 Pyright

`pyrightconfig.json`:

```json
{
  "include": ["src", "scripts", "app", "tests"],
  "exclude": ["**/__pycache__", "venv", "data", "experiments"],
  "pythonVersion": "3.10",
  "typeCheckingMode": "standard",
  "reportMissingTypeStubs": "warning",
  "reportUnknownMemberType": "none",
  "reportUnknownArgumentType": "none"
}
```

**`strict` 모드가 아닌 `standard`를 쓰는 이유**: PyTorch, diffusers, transformers가 타입 스텁이 불완전해서 strict는 False positive가 너무 많다. standard로 실용성 확보.

### 1.3 pre-commit (권장)

`.pre-commit-config.yaml`:

```yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.5.0
    hooks:
      - id: ruff
        args: [--fix]
      - id: ruff-format

  - repo: https://github.com/RobertCraigie/pyright-python
    rev: v1.1.370
    hooks:
      - id: pyright
```

**설치**: `pip install pre-commit && pre-commit install`

### 1.4 실행 체크리스트

커밋 전 다음이 전부 통과해야 한다:

```bash
ruff format . --check
ruff check .
pyright
pytest
```

에이전트가 코드를 작성한 후 **반드시 위 4개를 실행**해서 에러 없음을 확인할 것.

---

## 2. 파일 구조

### 2.1 임포트 순서 (Ruff가 자동 처리)

```python
# 1) 표준 라이브러리
import logging
from pathlib import Path
from typing import Optional

# 2) 서드파티
import numpy as np
import torch
from PIL import Image
from diffusers import StableDiffusionXLPipeline

# 3) 로컬 (프로젝트 내부)
from src.utils.device import get_device
from src.utils.logging import get_logger
```

각 그룹 사이 빈 줄 1개. 같은 그룹 내에서는 알파벳 순.

### 2.2 파일 크기

- **한 파일 500줄 초과 금지.** 초과 시 서브 모듈로 분리.
- **한 함수 50줄 초과 시 재검토.** 80줄 넘으면 반드시 분리.
- **한 클래스 300줄 초과 시 재검토.**

### 2.3 `__init__.py` 규칙

Public API를 명시적으로 export:

```python
# src/preprocessing/__init__.py

from src.preprocessing.bg_removal import remove_background
from src.preprocessing.lineart import extract_lineart
from src.preprocessing.palette import extract_palette

__all__ = [
    "remove_background",
    "extract_lineart",
    "extract_palette",
]
```

빈 `__init__.py`는 허용. **와일드카드 import (`from X import *`) 금지.**

### 2.4 모듈 내부 순서

파일 상단부터:

1. 모듈 docstring
2. `from __future__` 임포트 (있으면)
3. 표준/서드파티/로컬 임포트
4. 상수 (UPPER_CASE)
5. 타입 별칭
6. 헬퍼 함수 (private, `_`로 시작)
7. 메인 함수/클래스
8. `if __name__ == "__main__":` 블록 (스크립트인 경우)

---

## 3. 네이밍 컨벤션

| 대상 | 규칙 | 예시 |
|------|------|------|
| 변수, 함수 | `snake_case` | `image_tensor`, `extract_lineart()` |
| 클래스 | `PascalCase` | `StyleUnificationPipeline` |
| 상수 | `UPPER_SNAKE_CASE` | `DEFAULT_RESOLUTION = 1024` |
| Private | `_leading_underscore` | `_internal_cache`, `_validate()` |
| 타입 별칭 | `PascalCase` | `ImageArray = np.ndarray` |
| 모듈/파일 | `snake_case.py` | `bg_removal.py` |

**피해야 할 것**:
- 한 글자 변수 (`x`, `y`). 예외: 루프 카운터(`i`), 좌표(`x, y`).
- 파이썬 내장 shadowing (`list`, `dict`, `id`, `type` 같은 변수명 금지).
- 부정 boolean (`is_not_valid` 대신 `is_invalid` 또는 `not is_valid`).

**약어**: 업계 표준만 허용. `img`, `cfg`, `num`, `idx`, `ref`는 OK. 본인 편의 약어는 금지.

---

## 4. 타입 힌트

### 4.1 필수 규칙

- **모든 public 함수/메서드**: 인자 + 반환 타입 힌트 필수.
- **public 클래스 속성**: 타입 힌트 필수.
- **지역 변수**: 추론 가능하면 생략. 모호하면 명시.

```python
# ✅ Good
def extract_lineart(
    image: Image.Image,
    detector: str = "lineart_anime",
    threshold: int = 100,
) -> Image.Image:
    ...

# ❌ Bad
def extract_lineart(image, detector="lineart_anime", threshold=100):
    ...
```

### 4.2 모던 문법 (Python 3.10+)

```python
# ✅ Good (3.10+)
def f(x: int | None) -> list[str]:
    ...

# ❌ Old (3.9 이하)
from typing import Optional, List
def f(x: Optional[int]) -> List[str]:
    ...
```

단, `Optional`과 `| None`은 어느 쪽이든 허용 (가독성 우선).

### 4.3 텐서/배열 타입

```python
import numpy as np
import numpy.typing as npt
import torch

# NumPy
def process(arr: npt.NDArray[np.float32]) -> npt.NDArray[np.uint8]:
    ...

# PyTorch: 텐서는 torch.Tensor로 표기. shape는 docstring에 명시.
def encode(image: torch.Tensor) -> torch.Tensor:
    """
    Args:
        image: (B, C, H, W) float32 tensor in [0, 1].
    
    Returns:
        (B, D) float32 embedding.
    """
    ...
```

PyTorch는 shape 정보를 타입에 담을 수 없으므로 **docstring에 shape를 반드시 명시**.

### 4.4 Protocol 활용

duck typing이 필요하면 Protocol:

```python
from typing import Protocol

class Segmenter(Protocol):
    def segment(self, image: Image.Image) -> np.ndarray: ...
```

---

## 5. Docstring

### 5.1 형식 (Google Style)

모든 public 함수/클래스에 docstring 필수.

```python
def extract_lineart(
    image: Image.Image,
    detector: str = "lineart_anime",
    threshold: int = 100,
) -> Image.Image:
    """이미지에서 lineart 맵을 추출한다.

    벡터/일러스트 스타일에 적합한 라인 맵을 생성한다. ControlNet의
    구조 조건 신호로 사용된다.

    Args:
        image: 입력 이미지. RGBA 가능.
        detector: 사용할 detector. 'lineart_anime' | 'canny' | 'hed'.
        threshold: Canny 사용 시 low threshold 값.

    Returns:
        Grayscale lineart 이미지. 흰색 배경에 검은 선.

    Raises:
        ValueError: detector가 지원되지 않는 값일 때.

    Example:
        >>> img = Image.open("char.png")
        >>> lineart = extract_lineart(img, detector="lineart_anime")
        >>> lineart.save("char_lineart.png")
    """
    ...
```

### 5.2 Docstring이 생략 가능한 경우

- Private 함수/메서드 (`_`로 시작)
- 자명한 getter/setter
- `__init__`은 클래스 docstring에 통합 가능

### 5.3 클래스 docstring

```python
class StyleUnificationPipeline:
    """스타일 통일 변환 파이프라인.

    SDXL + IP-Adapter + ControlNet을 조합하여 원본 에셋의 형태를
    유지하면서 reference 이미지의 스타일로 변환한다.

    Attributes:
        pipe: 내부 diffusers 파이프라인 인스턴스.
        config: 로드된 설정 딕셔너리.

    Example:
        >>> pipeline = StyleUnificationPipeline.from_config("configs/default.yaml")
        >>> result = pipeline.transform(source, reference, [lineart])
    """

    def __init__(self, config: dict) -> None:
        """파이프라인 초기화.

        Args:
            config: YAML에서 로드된 설정 딕셔너리.
        """
        ...
```

### 5.4 모듈 docstring

파일 최상단에 1-3줄:

```python
"""Lineart 추출 모듈.

ControlNet 입력으로 사용할 엣지 맵을 생성한다. Canny, HED, lineart_anime
detector를 지원한다.
"""
```

---

## 6. 에러 처리

### 6.1 기본 원칙

- **구체적 예외만 catch.** `except Exception:` 금지.
- **조용히 무시 금지.** 최소한 로깅.
- **모르면 re-raise.** 처리할 수 없는 예외는 올려보내라.

```python
# ✅ Good
try:
    model = load_model(path)
except FileNotFoundError:
    logger.error(f"Model not found: {path}")
    raise
except torch.cuda.OutOfMemoryError:
    logger.warning("OOM, retrying with CPU offload")
    model = load_model(path, offload=True)

# ❌ Bad
try:
    model = load_model(path)
except Exception:  # 너무 광범위
    pass  # 조용히 무시
```

### 6.2 커스텀 예외

프로젝트 고유 에러는 `src/utils/exceptions.py`에 정의:

```python
class StyleUnifierError(Exception):
    """이 프로젝트의 모든 예외의 베이스."""

class ModelLoadError(StyleUnifierError):
    """모델 로딩 실패."""

class QualityCheckFailedError(StyleUnifierError):
    """품질 검증 실패."""
```

### 6.3 Validation

인자 검증은 함수 시작 직후:

```python
def transform(strength: float) -> Image.Image:
    if not 0.0 <= strength <= 1.0:
        raise ValueError(f"strength must be in [0, 1], got {strength}")
    ...
```

### 6.4 Context Manager 활용

리소스 정리가 필요하면 항상 `with`:

```python
# ✅ Good
with torch.no_grad():
    output = model(input)

# ❌ Bad (try/finally로 수동 관리)
torch.set_grad_enabled(False)
try:
    output = model(input)
finally:
    torch.set_grad_enabled(True)
```

---

## 7. 로깅

### 7.1 Logger 설정

프로젝트 공통 로거 사용:

```python
# src/utils/logging.py
import logging

def get_logger(name: str) -> logging.Logger:
    """모듈별 로거 반환."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger
```

각 모듈에서:

```python
from src.utils.logging import get_logger

logger = get_logger(__name__)

def load_model(path: str) -> ...:
    logger.info(f"Loading model: {path}")
    ...
```

### 7.2 Level 가이드

| Level | 용도 | 예시 |
|-------|------|------|
| `DEBUG` | 개발 중 상세 추적 | `logger.debug(f"Tensor shape: {x.shape}")` |
| `INFO` | 정상 동작 주요 이벤트 | `logger.info("Pipeline initialized")` |
| `WARNING` | 비정상이지만 계속 진행 가능 | `logger.warning("CPU offload enabled due to low VRAM")` |
| `ERROR` | 기능 실패, 예외 발생 | `logger.error(f"Failed to load: {e}")` |
| `CRITICAL` | 시스템 중단 | 거의 사용 안 함 |

### 7.3 금지 사항

- **`print()` 금지.** 모든 출력은 logger로.
- 예외: CLI 진행률 표시(`tqdm`), 사용자 대면 메시지(Gradio UI).

### 7.4 f-string vs lazy

간단한 경우 f-string 허용, 성능 크리티컬하면 `%` 스타일:

```python
# 일반
logger.info(f"Loaded {n} samples")

# 성능 민감 (DEBUG이 꺼져 있어도 f-string은 평가됨)
logger.debug("Tensor: %s", expensive_repr(x))
```

---

## 8. 설정 · 경로 · 상수

### 8.1 경로 처리

**`pathlib.Path`만 사용. 문자열 경로 concat 금지.**

```python
# ✅ Good
from pathlib import Path

data_dir = Path("data/raw")
file_path = data_dir / "kenney" / "char_001.png"
if file_path.exists():
    ...

# ❌ Bad
import os
file_path = os.path.join("data/raw", "kenney", "char_001.png")  # OS 의존
file_path = "data/raw" + "/" + "kenney" + "/" + "char_001.png"  # 금지
```

### 8.2 설정 로딩

하드코딩 금지. `configs/*.yaml`에서 로드:

```python
import yaml
from pathlib import Path

def load_config(config_path: Path) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

config = load_config(Path("configs/default.yaml"))
```

### 8.3 상수 관리

모듈 수준 상수는 파일 상단:

```python
# src/generation/pipeline.py

DEFAULT_RESOLUTION = 1024
MIN_RESOLUTION = 512
MAX_RESOLUTION = 1536

SUPPORTED_SCHEDULERS = ("DPMSolverMultistepScheduler", "EulerAncestralDiscreteScheduler")
```

전역 상수는 `src/constants.py`에 집중.

### 8.4 환경 변수

민감 정보(API 키)는 환경 변수:

```python
import os

HF_TOKEN = os.environ.get("HF_TOKEN")
if HF_TOKEN is None:
    logger.warning("HF_TOKEN not set; some models may not download")
```

**`.env` 파일은 `.gitignore`에 포함.**

---

## 9. 테스트

### 9.1 파일 구조

```
tests/
├── conftest.py                 # 공통 fixture
├── test_preprocessing.py
├── test_encoding.py
├── test_generation.py          # 가벼운 통합 테스트만
├── test_postprocessing.py
└── fixtures/                   # 테스트용 샘플 이미지
    ├── char_small.png
    └── ref_small.png
```

### 9.2 네이밍

```python
def test_<함수명>_<시나리오>():
    ...

# 예시
def test_extract_lineart_on_rgba_input():
    ...

def test_remove_background_preserves_shape():
    ...

def test_pipeline_raises_on_invalid_strength():
    ...
```

### 9.3 Fixture

공통 리소스는 `conftest.py`:

```python
# tests/conftest.py
import pytest
from pathlib import Path
from PIL import Image

@pytest.fixture
def sample_character() -> Image.Image:
    return Image.open(Path("tests/fixtures/char_small.png"))

@pytest.fixture
def sample_reference() -> Image.Image:
    return Image.open(Path("tests/fixtures/ref_small.png"))
```

### 9.4 마커로 느린 테스트 격리

모델 로딩이 필요한 테스트는 별도 마커:

```python
# pyproject.toml
[tool.pytest.ini_options]
markers = [
    "slow: 모델 로딩/추론이 필요한 느린 테스트",
    "gpu: GPU가 필요한 테스트",
]
```

```python
@pytest.mark.slow
@pytest.mark.gpu
def test_full_pipeline_end_to_end():
    ...
```

**실행**:
```bash
pytest                       # 빠른 테스트만 (기본)
pytest -m "not slow"         # 명시적으로 느린 것 제외
pytest -m slow               # 느린 것만
pytest -m "gpu and slow"     # GPU 느린 것만
```

### 9.5 원칙

- **단위 테스트는 모델 없이.** mock 또는 tiny 입력으로 로직만 검증.
- **통합 테스트는 @mark.slow + @mark.gpu.** CI 기본 실행에서 제외.
- **한 테스트에 한 가지만 검증.** 긴 시나리오는 분리.
- **Flaky test 금지.** 랜덤성 있으면 seed 고정.

---

## 10. ML/PyTorch 특수 규칙

이 프로젝트의 고유 규칙. 일반 Python 스타일 가이드에 없는 항목.

### 10.1 Device 관리

**하드코딩 금지.** `src/utils/device.py`의 유틸 사용:

```python
# src/utils/device.py
import torch

def get_device() -> torch.device:
    """사용 가능한 최적 device 반환."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
```

```python
# ✅ Good
from src.utils.device import get_device
device = get_device()
model = model.to(device)

# ❌ Bad
model = model.to("cuda")  # 맥/윈도우에서 터짐
```

### 10.2 dtype 처리

CUDA/MPS만 fp16 지원. CPU는 fp32:

```python
def get_dtype(device: torch.device) -> torch.dtype:
    if device.type in ("cuda", "mps"):
        return torch.float16
    return torch.float32
```

### 10.3 Seed 관리

**모든 실험은 seed를 명시.** 재현성이 깨지면 실험이 무의미.

```python
import random
import numpy as np
import torch

def set_seed(seed: int) -> None:
    """모든 랜덤 소스에 seed 설정."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# 생성 시에도 generator 명시
generator = torch.Generator(device=device).manual_seed(seed)
pipe(..., generator=generator)
```

### 10.4 그라디언트 관리

추론 코드는 **반드시** `torch.no_grad()` 또는 `torch.inference_mode()`:

```python
# ✅ Good
with torch.inference_mode():
    output = model(input)

# ❌ Bad (메모리 낭비)
output = model(input)  # grad tracking이 활성화됨
```

### 10.5 메모리 관리

큰 텐서는 사용 후 즉시 해제:

```python
# 배치 처리 후
del output, intermediate
torch.cuda.empty_cache()
```

**모델 로딩 시 반드시 최적화**:

```python
pipe.enable_model_cpu_offload()
pipe.enable_vae_slicing()
try:
    pipe.enable_xformers_memory_efficient_attention()
except ModuleNotFoundError:
    logger.warning("xformers not installed, using default attention")
```

### 10.6 텐서 vs PIL Image

**경계 명확히**: UI/IO는 PIL, 모델 내부는 Tensor.

```python
def transform(
    source: Image.Image,       # 외부 인터페이스는 PIL
    reference: Image.Image,
) -> Image.Image:              # 반환도 PIL
    source_tensor = to_tensor(source).to(device)     # 변환
    # ... 내부 처리는 tensor ...
    result_tensor = pipeline(source_tensor)
    return to_pil(result_tensor)                     # 다시 PIL로
```

함수 시그니처에서 **PIL인지 Tensor인지 즉시 보여야** 한다.

### 10.7 실험 로그

모든 실험은 `experiments/NNN_name/` 폴더에:

```
experiments/
└── 002_ipadapter_scale_sweep/
    ├── config.yaml         # 사용한 설정
    ├── log.md              # 관찰 사항
    ├── results.json        # 정량 메트릭
    └── samples/            # 생성된 이미지 (gitignore)
```

실험 ID는 3자리 숫자 + 짧은 이름. 절대 덮어쓰지 말 것.

---

## 11. Commit & PR

### 11.1 커밋 메시지

Conventional Commits 준수:

```
<type>(<scope>): <subject>

<body (optional)>
```

**Type**:
- `feat`: 새 기능
- `fix`: 버그 수정
- `refactor`: 기능 변화 없는 리팩토링
- `docs`: 문서만
- `test`: 테스트 추가/수정
- `chore`: 빌드, 설정 변경
- `exp`: 실험 (결과/로그 추가)

**예시**:
```
feat(preprocessing): add lineart_anime detector support

feat(generation): implement multi-reference style aggregation
fix(device): handle MPS fallback when ControlNet fails
exp(002): ipadapter_scale sweep from 0.4 to 1.0
docs(readme): add quickstart section
```

### 11.2 브랜치

- `main`: 항상 동작 가능한 상태
- `feat/<name>`: 기능 개발
- `exp/<NNN>`: 실험 (병합 안 해도 됨, archive 목적)

### 11.3 커밋 크기

- **한 커밋 = 한 의도.** 리팩토링 + 새 기능 섞지 말 것.
- **큰 기능은 작은 커밋으로 쪼개라.** 500줄 이상 diff는 재검토.

---

## 부록 A: 빠른 참조 카드

```python
# 새 모듈 템플릿
"""<모듈 설명 한 줄>.

<선택: 추가 설명>
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from src.utils.device import get_device
from src.utils.logging import get_logger

logger = get_logger(__name__)

DEFAULT_VALUE = 42


def public_function(
    arg: int,
    optional: str | None = None,
) -> list[str]:
    """한 줄 요약.

    상세 설명.

    Args:
        arg: 설명.
        optional: 설명.

    Returns:
        설명.

    Raises:
        ValueError: 조건.
    """
    if arg < 0:
        raise ValueError(f"arg must be non-negative, got {arg}")
    
    logger.info(f"Processing with arg={arg}")
    return ["result"]


def _private_helper(x: int) -> int:
    return x * 2
```

## 부록 B: 커밋 전 최종 체크

```bash
# 1. 포맷
ruff format .

# 2. 린트 (자동 수정 포함)
ruff check --fix .

# 3. 타입 체크
pyright

# 4. 테스트
pytest -m "not slow"

# 5. 커밋
git add .
git commit -m "feat(scope): message"
```

---

*규칙이 작업을 방해하면 규칙을 바꾸자. 단, CLAUDE.md와 STYLE.md에 변경 사항을 명시적으로 기록.*
