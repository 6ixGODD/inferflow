# Audex

[Documentation](https://6ixgodd.github.io/audex/) • [Installation Guide](https://6ixgodd.github.io/audex/installation/) • [API Reference](https://6ixgodd.github.io/audex/reference/)

---

## System Requirements

- Python 3.10-3.13
- Poetry
- PortAudio
- FFmpeg
- SQLite3
- PyQt6 (Linux: install from system packages)
- NetworkManager (Linux: for WiFi connectivity)

### System Dependencies

**Ubuntu/Debian:**
```bash
sudo apt-get install python3-pyqt6 python3-pyqt6.qtwebengine \
    portaudio19-dev ffmpeg sqlite3 network-manager \
    libfcitx5-qt6-1 alsa-utils gcc build-essential
```

**macOS:**
```bash
brew install portaudio ffmpeg sqlite3
pip install PyQt6 PyQt6-WebEngine
```

**Windows:**
- PortAudio is bundled with PyAudio wheel
- FFmpeg: Download from https://ffmpeg.org/download.html and add to `PATH`
- SQLite3: Included with Python installation

---

## Installation

### From PyPI

```bash
pip install audex
```

### From Source

```bash
git clone https://github.com/6ixGODD/audex.git
cd audex
poetry install
```

### DEB Package (Debian/Ubuntu/Raspberry Pi)

Download the appropriate DEB package for your architecture from [Releases](https://github.com/6ixGODD/audex/releases).

For detailed installation instructions, see [Installation Guide](https://6ixgodd.github.io/audex/installation/).

**Quick Install:**

```bash
# Download and install
sudo dpkg -i audex_{version}_arm64.deb
sudo apt-get install -f

# Run configuration wizard
sudo audex-setup

# Start application
sudo audex
```

---

## Usage

### Run Application

```bash
# Start with config file
audex -c config.yaml

# Using installed package
python -m audex -c config.yaml
```

### Initialize Configuration

```bash
# Generate default configuration
audex init gencfg --format yaml --output config.yaml

# Generate system configuration (Linux)
audex init gencfg --format system --output /etc/audex/config.yml --platform linux
```

### Initialize VPR Group

```bash
# Initialize voice print recognition group
audex init vprgroup --config config.yaml
```

### File Export Server

```bash
# Start file export server
audex serve --config config.yaml
```

---

## Configuration

Configuration file structure:

```yaml
core:
  app:
    app_name: Audex
    native: true
  logging:
    targets:
      - logname: stdout
        loglevel: info
  audio:
    sample_rate: 16000

provider:
  transcription:
    provider: dashscope
    dashscope:
      credential:
        api_key: <YOUR_API_KEY>

  vpr:
    provider: xfyun
    xfyun:
      credential:
        app_id: <YOUR_APP_ID>
        api_key: <YOUR_API_KEY>
        api_secret: <YOUR_API_SECRET>

infrastructure:
  sqlite:
    uri: "sqlite+aiosqlite:///path/to/audex.db"
  store:
    type: localfile
    base_url: /path/to/store
```

See `config.example.yml` for complete configuration options.

---

## Development

### Install Development Dependencies

```bash
# Development environment
poetry install --extras dev

# Testing environment
poetry install --extras test

# Documentation environment
poetry install --extras docs
```

### Build Package

```bash
# Build wheel and sdist
poetry build

# Output: dist/audex-{version}-py3-none-any.whl
```

### Run Tests

```bash
poetry install --extras test
poetry run pytest
```

### Documentation

```bash
poetry install --extras docs
poetry run mkdocs serve

# Visit: http://127.0.0.1:8000
```

---

## DEB Package Development

### Build DEB Package

**Prerequisites:**
- Docker

**Build:**

```bash
cd packaging/linux

# Build for ARM64 (Raspberry Pi)
./build.sh

# Build for AMD64
./build.sh amd64
```

**Output:** `dist/audex_{version}_{arch}.deb`

### Test DEB Package

```bash
cd packaging/linux
./test.sh arm64
```

**Inside test container:**

```bash
# Install package
dpkg -i /tmp/audex.deb
apt-get install -f

# Verify installation
which audex
audex --version

# View configurations
cat /etc/audex/config.system.yml
cat /etc/audex/config.example.yml

# Run configuration wizard
audex-setup

# Exit container
exit
```

---

## Project Structure

```
audex/
├── audex/                 # Main package
│   ├── cli/               # Command-line interface
│   ├── service/           # Business layer
│   ├── entity/            # Entities
│   ├── filters/           # Data filters
│   ├── valueobj/          # Value objects
│   ├── view/              # View layer
│   └── lib/               # Shared libraries
├── packaging/
│   └── linux/             # DEB packaging
│       ├── templates/     # Package templates
│       ├── build.sh       # Build script
│       └── test.sh        # Test script
├── scripts/               # Development scripts
├── tests/                 # Test suite
└── pyproject.toml         # Project configuration
```

---

## Links

- **Documentation**: [https://6ixgodd.github.io/audex/](https://6ixgodd.github.io/audex/)
- **PyPI**: [https://pypi.org/project/audex/](https://pypi.org/project/audex/)
- **GitHub**: [https://github.com/6ixGODD/audex](https://github.com/6ixGODD/audex)
- **Issues**: [https://github.com/6ixGODD/audex/issues](https://github.com/6ixGODD/audex/issues)
- **Releases**: [https://github.com/6ixGODD/audex/releases](https://github.com/6ixGODD/audex/releases)


---

# Audex

[文档](https://6ixgodd.github.io/audex/) • [安装指南](https://6ixgodd.github.io/audex/installation/) • [API 参考](https://6ixgodd.github.io/audex/reference/)

---

## 系统要求

- Python 3.10-3.13
- Poetry
- PortAudio
- FFmpeg
- SQLite3
- PyQt6（Linux：从系统包安装）
- NetworkManager（Linux：WiFi 连接支持）

### 系统依赖

**Ubuntu/Debian:**
```bash
sudo apt-get install python3-pyqt6 python3-pyqt6.qtwebengine \
    portaudio19-dev ffmpeg sqlite3 network-manager \
    libfcitx5-qt6-1 alsa-utils gcc build-essential
```

**macOS:**
```bash
brew install portaudio ffmpeg sqlite3
pip install PyQt6 PyQt6-WebEngine
```

**Windows:**
- PortAudio 已包含在 PyAudio wheel 中
- FFmpeg：从 https://ffmpeg.org/download.html 下载并添加到 `PATH`
- SQLite3：已包含在 Python 安装中

---

## 安装

### 从 PyPI 安装

```bash
pip install audex
```

### 从源码安装

```bash
git clone https://github.com/6ixGODD/audex.git
cd audex
poetry install
```

### DEB 包安装（Debian/Ubuntu/Raspberry Pi）

从 [Releases](https://github.com/6ixGODD/audex/releases) 下载对应架构的 DEB 包。

详细安装说明请参阅 [安装指南](https://6ixgodd.github.io/audex/installation/)。

**快速安装：**

```bash
# 下载并安装
sudo dpkg -i audex_{version}_arm64.deb
sudo apt-get install -f

# 运行配置向导
sudo audex-setup

# 启动应用
sudo audex
```

---

## 使用

### 运行应用

```bash
# 使用配置文件启动
audex -c config.yaml

# 使用已安装的包
python -m audex -c config.yaml
```

### 初始化配置

```bash
# 生成默认配置
audex init gencfg --format yaml --output config.yaml

# 生成系统配置（Linux）
audex init gencfg --format system --output /etc/audex/config.yml --platform linux
```

### 初始化 VPR 组

```bash
# 初始化声纹识别组
audex init vprgroup --config config.yaml
```

### 文件导出服务器

```bash
# 启动文件导出服务器
audex serve --config config.yaml
```

---

## 配置

配置文件结构：

```yaml
core:
  app:
    app_name: Audex
    native: true
  logging:
    targets:
      - logname: stdout
        loglevel: info
  audio:
    sample_rate: 16000

provider:
  transcription:
    provider: dashscope
    dashscope:
      credential:
        api_key: <YOUR_API_KEY>

  vpr:
    provider: xfyun
    xfyun:
      credential:
        app_id: <YOUR_APP_ID>
        api_key: <YOUR_API_KEY>
        api_secret: <YOUR_API_SECRET>

infrastructure:
  sqlite:
    uri: "sqlite+aiosqlite:///path/to/audex.db"
  store:
    type: localfile
    base_url: /path/to/store
```

完整配置选项请参阅 `config.example.yml`。

---

## 开发

### 安装开发依赖

```bash
# 开发环境
poetry install --extras dev

# 测试环境
poetry install --extras test

# 文档环境
poetry install --extras docs
```

### 构建包

```bash
# 构建 wheel 和 sdist
poetry build

# 输出：dist/audex-{version}-py3-none-any.whl
```

### 运行测试

```bash
poetry install --extras test
poetry run pytest
```

### 文档

```bash
poetry install --extras docs
poetry run mkdocs serve

# 访问：http://127.0.0.1:8000
```

---

## DEB 包开发

### 构建 DEB 包

**前置要求：**
- Docker

**构建：**

```bash
cd packaging/linux

# 构建 ARM64（Raspberry Pi）
./build.sh

# 构建 AMD64
./build.sh amd64
```

**输出：** `dist/audex_{version}_{arch}.deb`

### 测试 DEB 包

```bash
cd packaging/linux
./test.sh arm64
```

**在测试容器中：**

```bash
# 安装包
dpkg -i /tmp/audex.deb
apt-get install -f

# 验证安装
which audex
audex --version

# 查看配置
cat /etc/audex/config.system.yml
cat /etc/audex/config.example.yml

# 运行配置向导
audex-setup

# 退出容器
exit
```

---

## 项目结构

```
audex/
├── audex/                 # 主包
│   ├── cli/               # 命令行界面
│   ├── service/           # 业务层
│   ├── entity/            # 实体
│   ├── filters/           # 数据过滤器
│   ├── valueobj/          # 值对象
│   ├── view/              # 视图层
│   └── lib/               # 共享库
├── packaging/
│   └── linux/             # DEB 打包
│       ├── templates/     # 包模板
│       ├── build.sh       # 构建脚本
│       └── test.sh        # 测试脚本
├── scripts/               # 开发脚本
├── tests/                 # 测试套件
└── pyproject.toml         # 项目配置
```

---

## 链接

- **文档**: [https://6ixgodd.github.io/audex/](https://6ixgodd.github.io/audex/)
- **PyPI**: [https://pypi.org/project/audex/](https://pypi.org/project/audex/)
- **GitHub**: [https://github.com/6ixGODD/audex](https://github.com/6ixGODD/audex)
- **Issues**: [https://github.com/6ixGODD/audex/issues](https://github.com/6ixGODD/audex/issues)
- **Releases**: [https://github.com/6ixGODD/audex/releases](https://github.com/6ixGODD/audex/releases)
