# SETUP.md

> 프로젝트 개발 환경 구축 가이드. 3개 OS 지원.
> **실행 환경 = 리눅스 RTX 4070** / **작성 환경 = 윈도우, macOS**
>
> Last verified: 2026-04-24 (버전 정보는 6개월 단위 재검증 권장)

---

## 목차

1. [Quick Start](#1-quick-start)
2. [사전 준비 공통](#2-사전-준비-공통)
3. [리눅스 상세 가이드 (주 환경)](#3-리눅스-상세-가이드-주-환경)
4. [macOS 가이드 (M4)](#4-macos-가이드-m4)
5. [Windows 가이드](#5-windows-가이드)
6. [모델 다운로드](#6-모델-다운로드)
7. [스모크 테스트 (검증)](#7-스모크-테스트-검증)
8. [VS Code 설정](#8-vs-code-설정)
9. [트러블슈팅](#9-트러블슈팅)

---

## 1. Quick Start

**이미 환경이 준비된 경우** (CUDA, Python 설치 완료):

```bash
# 1. 저장소 클론
git clone <repo-url> style-unifier
cd style-unifier

# 2. 가상환경 + 의존성
python3.11 -m venv venv
source venv/bin/activate          # Linux/macOS
# venv\Scripts\activate            # Windows

pip install --upgrade pip
pip install -r requirements.txt

# 3. HuggingFace 인증
huggingface-cli login             # 토큰 입력

# 4. 모델 다운로드 (~20GB)
python scripts/download_models.py

# 5. 검증
python scripts/smoke_test.py
```

성공 시 `✅ All checks passed`가 출력됩니다. 실패 시 → [9. 트러블슈팅](#9-트러블슈팅).

처음 설치라면 → [2. 사전 준비](#2-사전-준비-공통)부터 순서대로 진행.

---

## 2. 사전 준비 공통

### 2.1 Git

모든 OS에서 Git이 필요합니다.

- 리눅스: `sudo apt install git` (Ubuntu/Debian)
- macOS: `brew install git` 또는 Xcode Command Line Tools
- Windows: [git-scm.com](https://git-scm.com/download/win) 설치

**LF 줄바꿈 설정** (프로젝트 루트에 `.gitattributes`로 관리되지만 글로벌 설정도 권장):

```bash
git config --global core.autocrlf input    # Linux/macOS
git config --global core.autocrlf true     # Windows
```

### 2.2 Python 3.11 설치

**왜 3.11인가**: PyTorch/xformers/bitsandbytes 생태계 호환성 최고. 자세한 사유는 STYLE.md 참조.

**리눅스 (Ubuntu 22.04+)**:

```bash
# deadsnakes PPA 추가 (Ubuntu에 3.11이 기본 아닌 경우)
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update
sudo apt install python3.11 python3.11-venv python3.11-dev

# 확인
python3.11 --version  # Python 3.11.x
```

**macOS**:

```bash
# Homebrew
brew install python@3.11

# 확인
python3.11 --version
```

**Windows**:

- [python.org/downloads](https://www.python.org/downloads/) 에서 Python 3.11.x 설치
- 설치 시 **"Add Python to PATH" 반드시 체크**
- PowerShell에서 확인: `python --version`

### 2.3 HuggingFace 계정 및 토큰

1. [huggingface.co](https://huggingface.co) 로그인
2. Settings → Access Tokens → **New token**
3. 이름: `style-unifier-dev`, 권한: **Read** (Write 불필요)
4. 토큰 복사 (한 번만 보임)

**환경 변수 설정**:

- Linux/macOS (`~/.bashrc` 또는 `~/.zshrc`):
  ```bash
  export HF_TOKEN="hf_..."
  ```

- Windows (PowerShell):
  ```powershell
  [System.Environment]::SetEnvironmentVariable("HF_TOKEN", "hf_...", "User")
  ```

**확인**: 새 터미널에서 `echo $HF_TOKEN` (Windows는 `$env:HF_TOKEN`).

---

## 3. 리눅스 상세 가이드 (주 환경)

**가장 중요한 섹션입니다.** 모든 모델 추론과 실험은 여기서 수행됩니다.

### 3.1 시스템 요구사항 확인

```bash
# OS 확인 (Ubuntu 22.04 이상 권장)
lsb_release -a

# RAM 확인 (16GB 이상 권장)
free -h

# GPU 확인 (RTX 4070, VRAM 12GB)
lspci | grep -i nvidia

# 디스크 여유 공간 (최소 50GB 권장, 모델 + 데이터)
df -h
```

### 3.2 NVIDIA 드라이버 설치

**중요**: 드라이버는 CUDA보다 먼저 설치. CUDA는 나중에 PyTorch에 번들된 것을 사용할 예정이라 시스템 CUDA 설치는 필수 아님.

**Ubuntu 22.04 기준**:

```bash
# 1. 기존 NVIDIA 드라이버 확인
nvidia-smi

# 만약 "command not found"가 나오면 설치 필요:

# 2. 권장 드라이버 확인
ubuntu-drivers devices

# 3. 자동 설치 (권장)
sudo ubuntu-drivers autoinstall

# 4. 재부팅
sudo reboot

# 5. 재부팅 후 확인
nvidia-smi
```

성공 시 `nvidia-smi` 출력 예시:
```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 550.xx   Driver Version: 550.xx   CUDA Version: 12.4     |
|-----------------------------------------------------------------------------|
| RTX 4070    ... 12288MiB ...                                        |
+-----------------------------------------------------------------------------+
```

**드라이버 버전 요구사항**:
- PyTorch 2.1+ with CUDA 12.1: **드라이버 530+ 필요**
- CUDA 12.4: 드라이버 550+ 권장

### 3.3 프로젝트 클론 및 가상환경

```bash
cd ~/dev    # 본인 선호 위치
git clone <repo-url> style-unifier
cd style-unifier

# Python 3.11 가상환경
python3.11 -m venv venv
source venv/bin/activate

# pip 최신화
pip install --upgrade pip setuptools wheel
```

### 3.4 PyTorch 설치 (CUDA 12.1 버전)

**중요**: `requirements.txt`의 torch보다 PyTorch를 먼저 정확한 CUDA 버전으로 설치.

```bash
# CUDA 12.1 버전 (RTX 4070 기준 가장 안정)
pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121
```

**최신 CUDA 12.4 또는 12.6 시도** (선택, 트러블슈팅 추가 가능성):
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

**검증**:
```bash
python -c "import torch; print(torch.__version__); print('CUDA:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0))"
```

예상 출력:
```
2.1.2+cu121
CUDA: True
Device: NVIDIA GeForce RTX 4070
```

`CUDA: False`가 나오면 → [9.1 CUDA 트러블슈팅](#91-cuda가-인식되지-않을-때).

### 3.5 나머지 의존성 설치

```bash
pip install -r requirements.txt
```

**주요 패키지 (reference용)**:
```
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
pytest
ruff
pyright
```

### 3.6 xformers 설치 (VRAM 절약, 권장)

xformers는 attention 메모리를 50%+ 절감합니다. 12GB VRAM에서 중요.

```bash
pip install xformers==0.0.23.post1 --index-url https://download.pytorch.org/whl/cu121
```

**주의**: xformers는 PyTorch, CUDA 버전과 정확히 맞아야 합니다. 버전 불일치 시 설치됐더라도 런타임 에러.

**검증**:
```bash
python -c "import xformers; print(xformers.__version__)"
```

설치 실패 시 → [9.3 xformers 트러블슈팅](#93-xformers-설치-실패).

### 3.7 개발 도구 설정

```bash
# pre-commit hook (선택이지만 권장)
pip install pre-commit
pre-commit install
```

---

## 4. macOS 가이드 (M4)

**역할**: 코드 작성 전용. 모델 추론 실행 금지 (불안정 + 느림).

### 4.1 Homebrew 및 Python

```bash
# Homebrew 설치 (없으면)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Python 3.11
brew install python@3.11
```

### 4.2 프로젝트 설치

```bash
cd ~/dev
git clone <repo-url> style-unifier
cd style-unifier

python3.11 -m venv venv
source venv/bin/activate
pip install --upgrade pip
```

### 4.3 PyTorch (MPS 지원, Apple Silicon)

```bash
# CUDA 없음, MPS 백엔드 사용
pip install torch torchvision
```

**검증**:
```bash
python -c "import torch; print('MPS:', torch.backends.mps.is_available())"
```

예상: `MPS: True`

### 4.4 나머지 의존성

```bash
pip install -r requirements.txt

# xformers는 Mac에서 동작 안 함 → 설치 생략
```

### 4.5 중요 제약

- **모델 추론 실행 금지**. CLAUDE.md에 명시된 규칙.
- 단위 테스트만 실행 (`pytest -m "not slow"`)
- 실제 동작 확인은 Linux에서만

---

## 5. Windows 가이드

**역할**: 코드 작성 전용 (CPU only). 모델 추론 실행 금지.

### 5.1 Python 및 Git

- [Python 3.11](https://www.python.org/downloads/) 설치 ("Add to PATH" 체크)
- [Git for Windows](https://git-scm.com/download/win) 설치

### 5.2 PowerShell 또는 Git Bash 사용

**권장**: PowerShell 7+ 또는 Git Bash. cmd.exe는 비권장.

### 5.3 프로젝트 설치

```powershell
cd $HOME\dev
git clone <repo-url> style-unifier
cd style-unifier

python -m venv venv
venv\Scripts\activate

pip install --upgrade pip
```

### 5.4 PyTorch (CPU only)

```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

**검증**:
```powershell
python -c "import torch; print(torch.__version__); print('CUDA:', torch.cuda.is_available())"
```

예상: `CUDA: False` (의도된 결과. Windows는 추론 안 함.)

### 5.5 나머지 의존성

```powershell
pip install -r requirements.txt
```

### 5.6 제약

- **모델 추론 실행 금지**. 
- `pytest -m "not slow"`로 빠른 테스트만 실행.
- 실제 동작 확인은 Linux에서 커밋 후 확인.

---

## 6. 모델 다운로드

**리눅스에서만 필요** (Mac/Windows는 추론 안 하므로 선택).

### 6.1 자동 다운로드 스크립트

```bash
python scripts/download_models.py
```

스크립트가 수행:
- SDXL base + Animagine-XL-3.1 (~7GB)
- IP-Adapter weights (~1GB)
- ControlNet lineart (~2.5GB)
- CLIP ViT-H/14 image encoder (~3.5GB)
- RMBG-1.4 (~200MB)
- LPIPS weights (~5MB)

총 다운로드: **~15-20GB**. 네트워크 속도에 따라 20분~2시간.

### 6.2 수동 다운로드 (필요시)

스크립트 실패 시 개별 다운로드:

```bash
# HuggingFace CLI 로그인 확인
huggingface-cli whoami

# SDXL Animagine
huggingface-cli download cagliostrolab/animagine-xl-3.1

# IP-Adapter
huggingface-cli download h94/IP-Adapter

# ControlNet
huggingface-cli download diffusers/controlnet-canny-sdxl-1.0

# RMBG
huggingface-cli download briaai/RMBG-1.4
```

### 6.3 캐시 위치 확인

기본 캐시 위치:
- Linux/macOS: `~/.cache/huggingface/hub/`
- Windows: `%USERPROFILE%\.cache\huggingface\hub\`

**변경하고 싶다면**:
```bash
export HF_HOME=/path/to/large/disk/hf_cache
```

### 6.4 디스크 여유 확인

```bash
df -h ~/.cache/huggingface
```

최소 30GB 여유 권장.

---

## 7. 스모크 테스트 (검증)

환경 구축이 제대로 됐는지 전체 검증.

### 7.1 테스트 스크립트 실행

```bash
python scripts/smoke_test.py
```

**스크립트가 확인하는 것**:
1. Python 버전 (3.11.x)
2. PyTorch 설치 및 device 인식
3. Diffusers/transformers import
4. 모델 캐시 존재
5. 작은 해상도(256x256)로 1장 생성 테스트
6. 출력이 RGBA인지 확인

성공 시:
```
✅ Python version: 3.11.8
✅ PyTorch: 2.1.2+cu121
✅ Device: cuda (NVIDIA GeForce RTX 4070, 12GB)
✅ Diffusers: 0.27.0
✅ Models cached: 5/5
✅ Generation test: 256x256 image in 12.3s
✅ Output mode: RGBA

All checks passed. Environment is ready.
```

### 7.2 개별 컴포넌트 테스트

스모크 테스트가 실패하면 개별 확인:

```bash
# 1. PyTorch + CUDA
python -c "import torch; assert torch.cuda.is_available()"

# 2. Diffusers
python -c "from diffusers import StableDiffusionXLPipeline"

# 3. 모델 로드 (메모리만 점유, 추론 X)
python -c "
from diffusers import StableDiffusionXLPipeline
import torch
pipe = StableDiffusionXLPipeline.from_pretrained(
    'cagliostrolab/animagine-xl-3.1',
    torch_dtype=torch.float16,
).to('cuda')
print('Model loaded, VRAM used:', torch.cuda.memory_allocated() / 1e9, 'GB')
"

# 4. pytest
pytest tests/ -m "not slow" -v
```

### 7.3 Gradio UI 실행

```bash
python app/gradio_app.py
```

브라우저가 `http://127.0.0.1:7860`을 열고 UI가 나타나면 완전 성공.

---

## 8. VS Code 설정

### 8.1 필수 확장

다음 확장을 설치:

- **Python** (`ms-python.python`)
- **Pylance** (`ms-python.vscode-pylance`)
- **Pyright** (`ms-pyright.pyright`)
- **Ruff** (`charliermarsh.ruff`)
- **GitLens** (`eamodio.gitlens`)
- **YAML** (`redhat.vscode-yaml`)

선택:
- **Error Lens** (`usernamehw.errorlens`) — 에러를 줄 옆에 표시
- **Jupyter** (`ms-toolsai.jupyter`) — 노트북 실험용
- **Better Comments** (`aaron-bond.better-comments`)

### 8.2 프로젝트 설정 (`.vscode/settings.json`)

이 프로젝트의 `.vscode/settings.json`에 다음이 포함되어야 합니다:

```json
{
  "python.defaultInterpreterPath": "${workspaceFolder}/venv/bin/python",
  
  "editor.formatOnSave": true,
  "editor.codeActionsOnSave": {
    "source.organizeImports": "explicit",
    "source.fixAll": "explicit"
  },
  
  "[python]": {
    "editor.defaultFormatter": "charliermarsh.ruff",
    "editor.tabSize": 4
  },
  
  "ruff.organizeImports": true,
  "ruff.fixAll": true,
  
  "python.analysis.typeCheckingMode": "standard",
  
  "python.testing.pytestEnabled": true,
  "python.testing.unittestEnabled": false,
  "python.testing.pytestArgs": [
    "tests",
    "-m", "not slow"
  ],
  
  "files.exclude": {
    "**/__pycache__": true,
    "**/.pytest_cache": true,
    "**/*.pyc": true,
    "venv": false,
    ".venv": false
  },
  
  "files.associations": {
    "CLAUDE.md": "markdown",
    "STYLE.md": "markdown"
  },
  
  "yaml.schemas": {
    "./schema/config.json": "configs/*.yaml"
  }
}
```

**Windows 사용자 주의**: `python.defaultInterpreterPath`를 `${workspaceFolder}/venv/Scripts/python.exe`로 변경.

### 8.3 Python 인터프리터 선택

VS Code 하단 상태바 또는 `Cmd/Ctrl + Shift + P` → "Python: Select Interpreter" → `./venv/bin/python`

### 8.4 launch.json (디버깅용)

`.vscode/launch.json`:

```json
{
  "version": "0.2.0",
  "configurations": [
    {
      "name": "Python: Current File",
      "type": "debugpy",
      "request": "launch",
      "program": "${file}",
      "console": "integratedTerminal",
      "justMyCode": false,
      "env": {
        "PYTHONPATH": "${workspaceFolder}"
      }
    },
    {
      "name": "Gradio App",
      "type": "debugpy",
      "request": "launch",
      "program": "${workspaceFolder}/app/gradio_app.py",
      "console": "integratedTerminal",
      "env": {
        "PYTHONPATH": "${workspaceFolder}"
      }
    },
    {
      "name": "Pytest: Current File",
      "type": "debugpy",
      "request": "launch",
      "module": "pytest",
      "args": ["${file}", "-v"],
      "console": "integratedTerminal"
    }
  ]
}
```

### 8.5 워크스페이스 권장 설정

**Claude Code와 함께 쓸 때**:

1. `.claude/agents/` 디렉토리 내용이 인식되는지 확인: VS Code 터미널에서 `claude /agents` 명령.
2. `CLAUDE.md`는 항상 편집 가능하도록 워크스페이스 루트에 두기.
3. `docs/` 폴더를 사이드바에 pin 해두면 편리.

---

## 9. 트러블슈팅

### 9.1 CUDA가 인식되지 않을 때

**증상**: `torch.cuda.is_available()` 이 `False`

**체크리스트**:

1. **NVIDIA 드라이버 확인**:
   ```bash
   nvidia-smi
   ```
   실패 시 → 드라이버 재설치 ([3.2](#32-nvidia-드라이버-설치))

2. **PyTorch CUDA 버전 확인**:
   ```bash
   python -c "import torch; print(torch.version.cuda)"
   ```
   `None`이면 CPU 버전 설치된 것. 재설치:
   ```bash
   pip uninstall torch torchvision
   pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121
   ```

3. **드라이버와 CUDA 호환성**:
   - CUDA 12.1 → 드라이버 530+
   - CUDA 12.4 → 드라이버 550+
   - `nvidia-smi` 출력의 "CUDA Version"이 PyTorch CUDA 이상이어야 함

4. **WSL2 사용자**: WSL2에서는 Windows GPU 드라이버가 자동 노출됨. 별도 Linux 드라이버 설치 **금지**. Windows 측 드라이버를 최신으로.

### 9.2 CUDA OOM (Out of Memory)

**증상**: `torch.cuda.OutOfMemoryError: CUDA out of memory`

**해결 순서**:

1. **Model CPU Offload**:
   ```python
   pipe.enable_model_cpu_offload()
   ```

2. **VAE Slicing**:
   ```python
   pipe.enable_vae_slicing()
   ```

3. **xformers**:
   ```python
   pipe.enable_xformers_memory_efficient_attention()
   ```

4. **해상도 축소**: 1024 → 768 (VRAM 44% 감소)

5. **다른 프로세스 종료**:
   ```bash
   nvidia-smi
   # PID 확인 후
   kill <PID>
   ```

6. **최후의 수단 - Sequential CPU Offload** (매우 느림):
   ```python
   pipe.enable_sequential_cpu_offload()
   ```

### 9.3 xformers 설치 실패

**증상**: `pip install xformers`가 빌드 에러 또는 설치 후 import 실패

**원인**: PyTorch + CUDA + Python 버전 불일치

**해결**:

1. **정확한 버전 매칭**:
   ```bash
   # PyTorch 2.1.2 + CUDA 12.1 + Python 3.11 → xformers 0.0.23.post1
   pip install xformers==0.0.23.post1 --index-url https://download.pytorch.org/whl/cu121
   ```

2. **버전 맞는 것 찾기**: [xformers Release](https://github.com/facebookresearch/xformers/releases)에서 본인 PyTorch 버전에 맞는 것 선택.

3. **빌드 실패 시 생략**: xformers 없이도 동작. 성능만 떨어짐.
   ```python
   try:
       pipe.enable_xformers_memory_efficient_attention()
   except Exception as e:
       logger.warning(f"xformers unavailable: {e}")
   ```

### 9.4 HuggingFace 다운로드 실패

**증상**: `OSError: ... is not a valid model identifier` 또는 401 Unauthorized

**해결**:

1. **토큰 재확인**:
   ```bash
   huggingface-cli whoami
   ```

2. **gated model 접근 권한**: 일부 모델은 HuggingFace에서 수동 승인 필요. 모델 페이지 방문해서 "Agree and access" 클릭.

3. **네트워크 문제**: 
   ```bash
   # 중간까지 받은 캐시 삭제 후 재시도
   rm -rf ~/.cache/huggingface/hub/.locks
   python scripts/download_models.py
   ```

4. **프록시 사용**:
   ```bash
   export HF_HUB_ENABLE_HF_TRANSFER=1  # 빠른 전송
   ```

### 9.5 Windows 관련 이슈

**"pathlib가 이상하게 동작"**: `WindowsPath` vs `PosixPath` 혼용. 코드에서 `Path()` 사용하면 자동 처리됨. 문자열 경로 하드코딩했다면 수정.

**"venv activate가 안 됨"**: PowerShell 실행 정책.
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

**긴 경로 문제**: Git이 260자 제한 걸림.
```powershell
git config --global core.longpaths true
```

### 9.6 macOS 관련 이슈

**MPS 에러 "MPS does not support ..."** : 일부 연산이 MPS에서 미지원. **Mac은 추론 안 하므로 무시하고 리눅스에서 실행**.

**Rosetta 관련 경고**: M4에서 x86 바이너리 실행 경고. `brew install python@3.11`로 native 설치.

### 9.7 일반 문제

**pip 설치가 매우 느림**:
```bash
# 한국 미러 사용
pip install -r requirements.txt -i https://mirror.kakao.com/pypi/simple/
```

**가상환경이 꼬임**:
```bash
deactivate   # 먼저 비활성화
rm -rf venv
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**Claude Code가 에이전트를 인식 못 함**:
```bash
# 세션 재시작
# 또는 Claude Code 내에서
/agents
```

---

## 부록 A: 환경 변수 요약

프로젝트에서 사용하는 환경 변수 (모두 선택적):

```bash
# 필수 (HuggingFace 인증)
export HF_TOKEN="hf_..."

# 선택 (캐시 위치 변경)
export HF_HOME="/custom/path/hf_cache"

# 선택 (빠른 다운로드)
export HF_HUB_ENABLE_HF_TRANSFER=1

# 선택 (PYTHONPATH)
export PYTHONPATH="/path/to/style-unifier:$PYTHONPATH"
```

## 부록 B: 디스크 공간 요구사항

| 항목 | 크기 |
|------|------|
| Python 3.11 + 가상환경 | ~500MB |
| PyTorch + CUDA 런타임 | ~3GB |
| 기타 의존성 | ~2GB |
| HuggingFace 모델 캐시 | ~20GB |
| 개발 중 생성 데이터 | ~10GB |
| 실험 결과 (experiments/) | ~5GB |
| **합계** | **~40GB 최소** |

50GB 여유 권장.

## 부록 C: 설치 검증 체크리스트

환경 구축 완료 시 다음 모두 통과해야:

- [ ] `python --version` → 3.11.x
- [ ] `pip list | grep torch` → 2.1.2+cu121 (Linux) / 2.1.2 (Mac/Win)
- [ ] Linux: `nvidia-smi` 정상 출력
- [ ] Linux: `python -c "import torch; assert torch.cuda.is_available()"`
- [ ] macOS: `python -c "import torch; assert torch.backends.mps.is_available()"`
- [ ] `python -c "from diffusers import StableDiffusionXLPipeline"`
- [ ] `huggingface-cli whoami` 정상 출력
- [ ] Linux: `python scripts/smoke_test.py` → All checks passed
- [ ] `pytest -m "not slow"` 통과
- [ ] `ruff check .` 통과
- [ ] `pyright` 통과
- [ ] VS Code에서 Python 인터프리터로 `./venv/bin/python` 선택됨

---

*이 문서는 환경 변경 시 업데이트 필요. 특히 PyTorch/CUDA 버전 변경 시 검증 후 Last verified 갱신.*
