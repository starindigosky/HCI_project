# 閩南語-國語 雙向語音聊天機器人 (Min Nan-Mandarin Voice Chatbot)

這是一個提供閩南語（台語）與國語（台灣普通話）之間雙向語音辨識、翻譯及合成的後端 API 服務。專案採用 FastAPI 框架，並整合了多種先進的語音處理 AI 模型，支援 RESTful API 和 WebSocket 即時通訊。

## 核心功能

- **多語言語音辨識 (ASR)**:
  - 支援 **國語** (使用 `openai/whisper-large-v3` 模型)。
  - 支援 **閩南語** (使用 `facebook/wav2vec2-large-xlsr-53` 模型)。
- **閩南語語音合成 (TTS)**:
  - 可將文字轉換為自然的閩南語語音 (使用 `facebook/mms-tts-nan` 模型)。
- **雙向語音翻譯 (Voice-to-Voice)**:
  - **國語 -> 閩南語**: 輸入國語語音，輸出閩南語語音。
  - **閩南語 -> 國語**: 輸入閩南語語音，輸出翻譯後的國語文字或語音。
- **即時串流 (Real-time Streaming)**:
  - 提供 WebSocket 端點，可實現即時語音辨識和語音聊天功能。

## 技術棧

- **後端框架**: FastAPI
- **AI / 機器學習**: PyTorch, Transformers
- **語音處理**: Librosa, SoundFile, PyDub
- **容器化**: Docker, Docker Compose
- **主要模型**: OpenAI Whisper, Facebook MMS, Facebook Wav2Vec2

---

## 環境建置

您可以選擇使用 Docker（推薦）或手動安裝來設定環境。

### 方法一：使用 Docker (推薦)

這是最簡單且最可靠的部署方式，能確保環境一致性。

1.  **複製專案庫**:
    ```bash
    git clone <your-repo-url>
    cd MIn_nan_ASR2_ZH_tw-chatbot/backend
    ```

2.  **建立環境變數檔案**:
    從範本檔案複製一份 `.env`。
    ```bash
    cp .env.example .env
    ```
    您可以根據需求修改 `.env` 檔案，例如，如果您有 NVIDIA GPU 並已安裝 CUDA，可以將 `DEVICE` 改為 `cuda`。

3.  **啟動服務**:
    使用 Docker Compose 啟動後端服務。
    ```bash
    docker-compose up --build
    ```
    第一次啟動時，Docker 會自動下載所需的 AI 模型並存放在 `backend/model_cache` 資料夾，過程可能需要一些時間。服務成功啟動後，API 將運行在 `http://localhost:8000`。

    > **注意**: Docker 環境已內建 `ffmpeg`，您無需手動安裝。

### 方法二：手動安裝

如果您不想使用 Docker，可以依照以下步驟手動設定。

1.  **複製專案庫**:
    ```bash
    git clone <your-repo-url>
    cd MIn_nan_ASR2_ZH_tw-chatbot/backend
    ```

2.  **安裝 FFmpeg**:
    `ffmpeg` 是處理音訊檔案的必要工具。請根據您的作業系統進行安裝。

    -   **macOS (使用 [Homebrew](https://brew.sh/))**:
        ```bash
        brew install ffmpeg
        ```

    -   **Debian/Ubuntu**:
        ```bash
        sudo apt-get update && sudo apt-get install ffmpeg
        ```

    -   **Windows (使用 [Chocolatey](https://chocolatey.org/))**:
        ```bash
        choco install ffmpeg
        ```
    安裝完成後，請確保 `ffmpeg` 指令可以在您的終端機中被存取。

3.  **安裝 Python**:
    建議使用 Python 3.10 或更高版本。

4.  **建立並啟用虛擬環境**:
    ```bash
    # Windows
    python -m venv venv
    .\venv\Scripts\activate

    # macOS / Linux
    python3 -m venv venv
    source venv/bin/activate
    ```

5.  **安裝依賴套件**:
    ```bash
    pip install -r requirements.txt
    ```

6.  **建立環境變數檔案**:
    從範本檔案複製一份 `.env`。
    ```bash
    cp .env.example .env
    ```

7.  **啟動服務**:
    使用 Uvicorn 啟動 FastAPI 應用。
    ```bash
    uvicorn app.main:app --host 0.0.0.0 --port 8000
    ```
    第一次運行時，程式會自動下載所需的 AI 模型，請耐心等候。

---

## 如何使用

服務啟動後，您可以透過以下方式與 API 互動。

### API 文件

完整的 API 文件可透過瀏覽器訪問：
- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

### 即時語音辨識測試 (網頁)

本專案內建一個簡單的網頁客戶端，讓您能快速測試即時語音辨識功能。

1.  **確認後端服務已啟動** (透過 Docker 或手動方式)。
2.  在您的電腦上，用瀏覽器打開位於 `backend/examples/realtime_asr_client.html` 的檔案。
3.  點擊 "Start Recording"，並對著麥克風說話（可選擇國語或閩南語）。
4.  網頁會即時將您說的話顯示在畫面上。

### 主要 API 端點

- **POST** `/api/v1/asr/transcribe`: 上傳音檔進行語音辨識。
- **POST** `/api/v1/tts/synthesize`: 輸入文字以合成為閩南語語音。
- **POST** `/api/v1/voice-conversion`: 上傳音檔，進行完整的語音對語音翻譯。
- **GET** `/api/v1/audio/{filename}`: 下載合成的語音檔案。

### WebSocket 端點

- **WS** `/api/v1/ws/asr`: 用於即時語音辨識。
- **WS** `/api/v1/ws/voice-chat`: 用於即時語音聊天與翻譯。

## 專案結構

```
backend/
├── app/                # FastAPI 應用程式核心程式碼
│   ├── api/            # API 路由 (REST & WebSocket)
│   ├── services/       # 核心服務 (ASR, TTS, 翻譯)
│   ├── models/         # 資料模型 (Pydantic schemas)
│   └── main.py         # FastAPI 應用主入口
├── model_cache/        # 下載的 AI 模型快取
├── outputs/            # TTS 合成的語音檔案存放處
├── uploads/            # 使用者上傳的暫存音檔
├── examples/           # API 使用範例 (包含網頁客戶端)
├── docker-compose.yml  # Docker Compose 設定
├── Dockerfile          # Docker 映像檔設定
└── requirements.txt    # Python 依賴套件
```