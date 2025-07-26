# KoboldCpp

[![License](https://img.shields.io/badge/license-AGPL%20v3-blue.svg)](LICENSE.md)
[![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20Linux%20%7C%20macOS%20%7C%20Android-lightgrey.svg)](README.md)
[![Documentation](https://img.shields.io/badge/docs-Architecture%20%7C%20Developer%20Guide-green.svg)](ARCHITECTURE.md)

KoboldCpp is an easy-to-use AI text-generation software for GGML and GGUF models, inspired by the original **KoboldAI**. It's a single self-contained distributable that builds off **llama.cpp** and adds many additional powerful features.

## 📚 Documentation

- **[Technical Architecture](ARCHITECTURE.md)** - Comprehensive system architecture with mermaid diagrams
- **[Developer Guide](DEVELOPER_GUIDE.md)** - Contributing, extending, and development setup
- **[API Documentation](https://lite.koboldai.net/koboldcpp_api)** - Complete API reference
- **[Wiki](https://github.com/LostRuins/koboldcpp/wiki)** - FAQ, troubleshooting, and guides

## 🚀 Quick Start

### One-Click Installation

| Platform | Download | Instructions |
|----------|----------|--------------|
| 🪟 **Windows** | [koboldcpp.exe](https://github.com/LostRuins/koboldcpp/releases/latest) | Download and run directly |
| 🐧 **Linux** | [koboldcpp-linux-x64](https://github.com/LostRuins/koboldcpp/releases/latest) | `chmod +x` then execute |
| 🍎 **macOS** | [koboldcpp-mac-arm64](https://github.com/LostRuins/koboldcpp/releases/latest) | Download, allow in security settings |
| ☁️ **Cloud** | [Google Colab](https://colab.research.google.com/github/LostRuins/koboldcpp/blob/concedo/colab.ipynb) | No installation required |

| UI Theme | Screenshot |
|----------|------------|
| **Chat Interface** | ![Preview](media/preview.png) |
| **Adventure Mode** | ![Preview](media/preview2.png) |
| **Writer Interface** | ![Preview](media/preview3.png) |

<details>
<summary>🖼️ More Screenshots</summary>

| Feature | Screenshot |
|---------|------------|
| **Settings Panel** | ![Preview](media/preview4.png) |
| **Model Selection** | ![Preview](media/preview5.png) |
| **API Interface** | ![Preview](media/preview6.png) |

</details>

## ✨ Features

### 🎯 Core Capabilities
- **Single file executable** - No installation required, no external dependencies
- **Universal model support** - All GGML and GGUF models with backward compatibility
- **Multi-modal AI** - Text generation, image creation, speech processing
- **Cross-platform** - Windows, Linux, macOS, and Android support

### 🤖 AI Features
| Feature | Description | API Support |
|---------|-------------|-------------|
| **Text Generation** | LLM inference with multiple architectures | ✅ KoboldAI, OpenAI, Ollama |
| **Image Generation** | Stable Diffusion (1.5, SDXL, SD3, Flux) | ✅ A1111, ComfyUI |
| **Speech-to-Text** | Whisper-based voice recognition | ✅ Whisper API |
| **Text-to-Speech** | OuteTTS voice synthesis | ✅ XTTS, OpenAI Speech |
| **Cognitive Reasoning** | OpenCog neural-symbolic AI | ✅ Custom endpoints |

### 🎨 User Interface
- **KoboldAI Lite UI** with editing tools, save formats, memory management
- **Multiple modes**: Chat, Adventure, Instruct, Story Writer
- **UI Themes**: Aesthetic roleplay, Classic writer, Corporate assistant, Messenger
- **Character support**: Tavern Character Cards, JSON import/export

### ⚡ Performance Features
- **GPU Acceleration**: CUDA, Vulkan, CLBlast support
- **CPU optimization**: AVX2, multi-threading, BLAS operations  
- **Memory efficiency**: Quantization, layer offloading, context compression
- **Advanced sampling**: Multiple samplers, regex support, custom patterns

## 🖥️ Installation & Usage

<details>
<summary><strong>🪟 Windows Usage (Recommended)</strong></summary>

### Installation
- Download **[koboldcpp.exe](https://github.com/LostRuins/koboldcpp/releases/latest)** from releases
- No installation required - just run the executable

### Quick Start
1. **Launch**: Double-click `koboldcpp.exe`
2. **Configure**: Use the GUI to set `Presets` and `GPU Layers`  
3. **Load Model**: Select your GGUF model file
4. **Connect**: Open http://localhost:5001 in your browser

### Command Line
```cmd
koboldcpp.exe --help                    # Show all options
koboldcpp.exe --model model.gguf        # Basic usage
koboldcpp.exe --model model.gguf --gpulayers 20 --usecublas  # GPU acceleration
```

</details>

<details>
<summary><strong>🐧 Linux Usage</strong></summary>

### Quick Install
```bash
# Download and install
curl -fLo koboldcpp https://github.com/LostRuins/koboldcpp/releases/latest/download/koboldcpp-linux-x64-oldpc && chmod +x koboldcpp

# Run
./koboldcpp --model model.gguf
```

### Using the Build Script
```bash
git clone https://github.com/LostRuins/koboldcpp.git
cd koboldcpp
./koboldcpp.sh dist    # Build from source
./koboldcpp.sh --help  # Show options
```

### GPU Support
```bash
# CUDA support
./koboldcpp --model model.gguf --usecublas --gpulayers 30

# Vulkan support  
./koboldcpp --model model.gguf --usevulkan --gpulayers 30
```

</details>

<details>
<summary><strong>🍎 macOS Usage</strong></summary>

### Installation
1. Download [koboldcpp-mac-arm64](https://github.com/LostRuins/koboldcpp/releases/latest)
2. Make executable: `chmod +x koboldcpp-mac-arm64`
3. Allow in Security Settings if blocked ([video guide](https://youtube.com/watch?v=NOW5dyA_JgY))

### Usage
```bash
./koboldcpp-mac-arm64 --model model.gguf
./koboldcpp-mac-arm64 --model model.gguf --gpulayers 20  # Metal GPU support
```

</details>

<details>
<summary><strong>☁️ Cloud & Container Options</strong></summary>

### Google Colab
- **[Official Colab Notebook](https://colab.research.google.com/github/LostRuins/koboldcpp/blob/concedo/colab.ipynb)** - Free GPU access

### Cloud Providers
- **[RunPod](https://koboldai.org/runpodcpp)** - Scalable GPU cloud
- **[Novita AI](https://koboldai.org/novitacpp)** - Alternative GPU cloud

### Docker
```bash
# Official Docker image
docker run -p 5001:5001 koboldai/koboldcpp

# Custom build
docker build --build-arg LLAMA_PORTABLE=1 -t koboldcpp .
```

</details>

<details>
<summary><strong>📱 Android (Termux)</strong></summary>

### Quick Setup
```bash
# Auto-installation script
curl -sSL https://raw.githubusercontent.com/LostRuins/koboldcpp/concedo/android_install.sh | sh
```

### Manual Installation
```bash
# Install Termux from F-Droid
apt update && apt install openssl
pkg install wget git python
git clone https://github.com/LostRuins/koboldcpp.git
cd koboldcpp && make LLAMA_PORTABLE=1
python koboldcpp.py --model model.gguf
```

</details>

### 📥 Getting Models

**Need help finding a model?** [Read our model guide!](https://github.com/LostRuins/koboldcpp/wiki#getting-an-ai-model-file)

#### 📄 Text Models (GGUF)
| Model Size | Recommended | Use Case |
|------------|-------------|----------|
| **7B** | [Airoboros Mistral 7B](https://huggingface.co/TheBloke/airoboros-mistral2.2-7B-GGUF/resolve/main/airoboros-mistral2.2-7b.Q4_K_S.gguf) | General purpose, fast |
| **13B** | [Tiefighter 13B](https://huggingface.co/KoboldAI/LLaMA2-13B-Tiefighter-GGUF/resolve/main/LLaMA2-13B-Tiefighter.Q4_K_S.gguf) | Balanced performance |
| **22B** | [Beepo 22B](https://huggingface.co/concedo/Beepo-22B-GGUF/resolve/main/Beepo-22B-Q4_K_S.gguf) | High quality output |

#### 🎨 Image Models
- [Anything v3](https://huggingface.co/admruul/anything-v3.0/resolve/main/Anything-V3.0-pruned-fp16.safetensors)
- [Deliberate V2](https://huggingface.co/Yntec/Deliberate2/resolve/main/Deliberate_v2.safetensors)  
- [Dreamshaper SDXL](https://huggingface.co/Lykon/dreamshaper-xl-v2-turbo/resolve/main/DreamShaperXL_Turbo_v2_1.safetensors)

#### 🗣️ Speech Models
- **Speech Recognition**: [Whisper models](https://huggingface.co/koboldcpp/whisper/tree/main)
- **Text-to-Speech**: [TTS models](https://huggingface.co/koboldcpp/tts/tree/main)
- **Vision**: [MMproj models](https://huggingface.co/koboldcpp/mmproj/tree/main)

#### 🔧 Convert Your Own Models
Download conversion tools [here](https://kcpptools.concedo.workers.dev):
1. `convert-hf-to-gguf.py` - Convert HuggingFace models
2. `quantize_gguf.exe` - Quantize for better performance

## ⚡ Performance Optimization

### 🚀 GPU Acceleration
| Backend | Platforms | Performance | Setup |
|---------|-----------|-------------|-------|
| **CUDA** | NVIDIA GPUs | Excellent | `--usecublas` |
| **Vulkan** | All modern GPUs | Very Good | `--usevulkan` |
| **CLBlast** | All GPUs | Good | `--useclblast` |
| **Metal** | Apple Silicon | Excellent | `--usemetal` (macOS) |

### 🧠 Memory Optimization
```bash
# GPU layer offloading (adjust based on VRAM)
--gpulayers 20          # Offload 20 layers to GPU

# Context size optimization  
--contextsize 4096      # Increase context window

# Memory efficiency
--usemmap              # Use memory mapping
--usemlock             # Lock model in memory
```

### 🎛️ Advanced Settings
```bash
# CPU optimization
--threads 8            # Set CPU thread count
--blasbatchsize 512    # Batch processing size

# Model modifications
--ropeconfig 1.0 10000 # RoPE frequency scaling
--tensor_split 70,30   # Multi-GPU tensor splitting
```

For detailed optimization guide, see our [Performance Wiki](https://github.com/LostRuins/koboldcpp/wiki).

## 🔧 Building from Source

<details>
<summary><strong>🐧 Linux Build (Automated)</strong></summary>

### Quick Build Script
```bash
git clone https://github.com/LostRuins/koboldcpp.git
cd koboldcpp

# Build options
./koboldcpp.sh                    # Launch GUI
./koboldcpp.sh --help            # Show all commands  
./koboldcpp.sh rebuild           # Rebuild libraries
./koboldcpp.sh dist              # Create binary
```

### Manual Build
```bash
# Basic CPU build
make

# Full-featured build
make LLAMA_CLBLAST=1 LLAMA_CUBLAS=1 LLAMA_VULKAN=1 LLAMA_PORTABLE=1

# GPU-specific builds
make LLAMA_CUBLAS=1              # CUDA support
make LLAMA_VULKAN=1              # Vulkan support  
make LLAMA_CLBLAST=1             # CLBlast support
```

### Dependencies
```bash
# Arch Linux
sudo pacman -S cblas clblast

# Debian/Ubuntu  
sudo apt install libclblast-dev
```

</details>

<details>
<summary><strong>🪟 Windows Build</strong></summary>

### Prerequisites
1. Download [w64devkit](https://github.com/skeeto/w64devkit) (vanilla version)
2. Clone repository: `git clone https://github.com/LostRuins/koboldcpp.git`

### Build Process
```cmd
# Basic build (w64devkit terminal)
make LLAMA_PORTABLE=1

# Full build with all backends
make LLAMA_CLBLAST=1 LLAMA_VULKAN=1 LLAMA_PORTABLE=1

# Create executable
pip install PyInstaller
make_pyinstaller.bat
```

### CUDA Build (Advanced)
- Requires Visual Studio + CMake + CUDA Toolkit
- Open CMakeLists.txt in Visual Studio
- Copy generated `koboldcpp_cublas.dll` to project directory

</details>

<details>
<summary><strong>🍎 macOS Build</strong></summary>

```bash
git clone https://github.com/LostRuins/koboldcpp.git
cd koboldcpp

# Basic build
make LLAMA_PORTABLE=1

# Metal GPU support
make LLAMA_METAL=1 LLAMA_PORTABLE=1

# Run
python koboldcpp.py --model model.gguf --gpulayers 20
```

</details>

<details>
<summary><strong>📱 Android Build (Termux)</strong></summary>

### Auto-Installation
```bash
curl -sSL https://raw.githubusercontent.com/LostRuins/koboldcpp/concedo/android_install.sh | sh
```

### Manual Build
```bash
# Install Termux from F-Droid
apt update
pkg install wget git python openssl
pkg upgrade

# Build
git clone https://github.com/LostRuins/koboldcpp.git
cd koboldcpp
make LLAMA_PORTABLE=1

# Test with small model
wget https://huggingface.co/concedo/KobbleTinyV2-1.1B-GGUF/resolve/main/KobbleTiny-Q4_K.gguf
python koboldcpp.py --model KobbleTiny-Q4_K.gguf
```

</details>

## 🔧 Third Party & Community Resources

<details>
<summary><strong>📦 Package Managers</strong></summary>

### Arch Linux
```bash
# AUR packages available
yay -S koboldcpp-cuda     # CUDA support
yay -S koboldcpp-hipblas  # AMD ROCm support
```

### Nix/NixOS
```nix
# Add to configuration.nix or home.nix
environment.systemPackages = [ pkgs.koboldcpp ];
# or
home.packages = [ pkgs.koboldcpp ];
```
[Example Nix setup and information](examples/nix_example.md)

</details>

<details>
<summary><strong>🐳 Community Docker Images</strong></summary>

- [korewaChino's Docker](https://github.com/korewaChino/koboldCppDocker)
- [noneabove1182's Docker](https://github.com/noneabove1182/koboldcpp-docker)

</details>

<details>
<summary><strong>🔗 Integrations</strong></summary>

### GPTLocalhost
[GPTLocalhost](https://gptlocalhost.com/demo#KoboldCpp) - Use KoboldCpp in Microsoft Word as a local alternative to "Copilot in Word"

### API Compatibility
KoboldCpp provides multiple API endpoints:
- **KoboldAI API** - Native format
- **OpenAI API** - `/v1/` compatible
- **Ollama API** - `/ollama/` compatible  
- **A1111 API** - `/sdapi/` for image generation
- **ComfyUI API** - `/comfy/` for workflows
- **Whisper API** - `/whisper/` for speech recognition
- **XTTS API** - `/xtts/` for text-to-speech

</details>

## 💡 AMD GPU Users

For AMD GPU acceleration, you have several options:

### Vulkan (Recommended)
```bash
# Works on both NVIDIA and AMD
koboldcpp --usevulkan --gpulayers 30
```

### ROCm Fork
For advanced AMD support, try the [ROCm fork](https://github.com/YellowRoseCx/koboldcpp-rocm) (may be outdated).

## 📋 Supported Model Architectures

KoboldCpp supports **hundreds of GGUF models**. If it's GGUF format, it should work!

**Popular architectures include:**
- Llama / Llama2 / Llama3 / Alpaca
- Mistral / Mixtral / Miqu  
- GPT-2 / GPT-NeoX / GPT-J
- Vicuna / Koala / Pygmalion
- Qwen / Qwen2 / Yi / Gemma / Gemma2
- Phi-2 / Phi-3 / Cerebras
- Falcon / Starcoder / Deepseek
- RWKV4 / MPT / Dolly / RedPajama
- And many more!

## 🆘 Support & Community

### 📚 Documentation & Help
- **[FAQ & Knowledge Base](https://github.com/LostRuins/koboldcpp/wiki)** - Common questions and solutions
- **[Technical Architecture](ARCHITECTURE.md)** - System design and diagrams  
- **[Developer Guide](DEVELOPER_GUIDE.md)** - Contributing and development
- **[API Documentation](https://lite.koboldai.net/koboldcpp_api)** - Complete API reference

### 💬 Community
- **[KoboldAI Discord](https://koboldai.org/discord)** - Real-time support and discussion
- **[GitHub Issues](https://github.com/LostRuins/koboldcpp/issues)** - Bug reports and feature requests
- **[GitHub Discussions](https://github.com/LostRuins/koboldcpp/discussions)** - General questions and ideas

### 🎮 Try Online
- **[Public Demo](https://koboldai-koboldcpp-tiefighter.hf.space/)** - Test KoboldCpp without installation (please don't abuse)

## 🏛️ Version History & Compatibility

### Legacy Support
- **v1.15+**: CLBlast support added
- **v1.33+**: Extended context size beyond official model limits  
- **v1.42+**: GGUF format support for Llama and Falcon
- **v1.55+**: Hardcoded CUDA paths on Linux
- **v1.60+**: Native Stable Diffusion image generation
- **v1.75+**: OpenBLAS deprecated, native CPU implementation

### Backward Compatibility
KoboldCpp maintains backward compatibility with **ALL past llama.cpp models**. However, reconverting/updating models is recommended for best results.

## 📄 License & Attribution

### Core Components
- **GGML Library** - [MIT License](MIT_LICENSE_GGML_SDCPP_LLAMACPP_ONLY.md) by ggerganov
- **llama.cpp** - [MIT License](https://github.com/ggml-org/llama.cpp) by ggerganov  
- **stable-diffusion.cpp** - [MIT License](https://github.com/leejet/stable-diffusion.cpp) by leejet

### KoboldCpp
- **KoboldCpp** - [AGPL v3.0 License](LICENSE.md)
- **KoboldAI Lite** - [AGPL v3.0 License](https://github.com/LostRuins/lite.koboldai.net)

### Contact
For inquiries, contact **@concedo** on Discord or **LostRuins** on GitHub.

---

### 🎯 Quick Reference

| Need | Link |
|------|------|
| **Download** | [Latest Release](https://github.com/LostRuins/koboldcpp/releases/latest) |
| **Models** | [Model Guide](https://github.com/LostRuins/koboldcpp/wiki#getting-an-ai-model-file) |
| **Help** | [Wiki](https://github.com/LostRuins/koboldcpp/wiki) \| [Discord](https://koboldai.org/discord) |
| **API** | [Documentation](https://lite.koboldai.net/koboldcpp_api) |
| **Development** | [Architecture](ARCHITECTURE.md) \| [Developer Guide](DEVELOPER_GUIDE.md) |
