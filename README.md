# 🎬 BrieflyAI

**BrieflyAI** is an AI-powered meeting/video assistant that turns a YouTube video or a local audio/video file into a transcript, a clean summary, extracted action items, key decisions, open questions — and lets you **chat with the meeting** afterwards using a RAG (Retrieval-Augmented Generation) pipeline.
🔗 Live Demo: https://brieflyyai.streamlit.app/

Give it a YouTube link or a file path, pick a language, and it will:

1. Download / convert the audio and split it into chunks
2. Transcribe it (locally with Whisper, or via Sarvam AI for Hindi → English)
3. Generate a short title
4. Summarize the meeting in bullet points
5. Extract **action items**, **key decisions**, and **open questions**
6. Build a vector store so you can ask follow-up questions about the transcript

---

## ✨ Features

- **Flexible input** — paste a YouTube URL or point to a local audio/video file
- **Local transcription** with OpenAI Whisper (no API needed for English)
- **Hinglish support** — Hindi (or mixed Hindi/English) audio is transcribed and translated to English via the Sarvam AI Speech-to-Text-Translate API
- **Map-reduce summarization** — long transcripts are chunked, summarized in parts, then combined into one polished summary
- **Structured meeting insights** — action items (with owner/deadline), key decisions, and open questions extracted automatically
- **RAG-powered chat** — ask questions about the meeting and get answers grounded strictly in the transcript, backed by a local Chroma vector store
- **Three ways to use it**:
  - A polished **Streamlit web app** (`app.py`)
  - A **command-line pipeline** (`main.py`)
  - A **FastAPI backend** (`server.py`) for a custom frontend

---

## 🧱 Project Structure

```
BrieflyAI/
├── app.py                  # Streamlit web UI (main entry point)
├── main.py                 # CLI pipeline + interactive chat
├── server.py                # FastAPI bridge exposing /process, /process-upload, /chat
├── test.py                  # Manual smoke-test script for the pipeline
├── requirements.txt
├── core/
│   ├── llm.py                # Groq LLM client + error handling
│   ├── transcriber.py         # Whisper (English) / Sarvam AI (Hinglish) transcription
│   ├── summarizer.py           # Map-reduce transcript summarization + title generation
│   ├── extractor.py            # Action items / key decisions / open questions extraction
│   ├── rag_engine.py            # LangChain LCEL RAG chain for Q&A over the transcript
│   └── vector_store.py           # Chroma vector store + HuggingFace embeddings
└── utils/
    └── audio_processor.py        # YouTube download, format conversion, audio chunking
```

> `app.html.backup` and `main.py.backup` are legacy/unused files and can be ignored.

---

## ⚙️ How It Works

```
Source (YouTube URL / local file)
        │
        ▼
 utils/audio_processor.py   → download / convert to WAV, chunked into segments
        │
        ▼
 core/transcriber.py        → Whisper (English) or Sarvam AI (Hinglish)
        │
        ▼
 core/summarizer.py         → title + map-reduce summary
        │
        ▼
 core/extractor.py          → action items, key decisions, open questions
        │
        ▼
 core/vector_store.py + core/rag_engine.py → Chroma vector store + LangChain RAG chain
        │
        ▼
   Chat with your meeting
```

All LLM calls go through **Groq** (via `langchain-groq`), using the `openai/gpt-oss-20b` model.

---

## 🛠️ Tech Stack

| Layer            | Tools |
|-------------------|-------|
| UI                | Streamlit |
| API server        | FastAPI + Uvicorn |
| Audio             | yt-dlp, pydub, ffmpeg |
| Transcription     | OpenAI Whisper (local), Sarvam AI STT-Translate API |
| LLM orchestration | LangChain (LCEL), langchain-groq |
| Vector store      | ChromaDB + `sentence-transformers` (`all-MiniLM-L6-v2`) embeddings |
| Export            | ReportLab, fpdf2 |

---

## 🚀 Getting Started

### Prerequisites

- Python **3.10+**
- [FFmpeg](https://ffmpeg.org/) installed and available on your `PATH` (required by `pydub`/`yt-dlp`)
- A [Groq API key](https://console.groq.com/) (required)
- A [Sarvam AI API key](https://www.sarvam.ai/) (only required if you plan to use the `hinglish` language option)

### Installation

```bash
git clone https://github.com/avishi0229/BrieflyAI.git
cd BrieflyAI

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### Configuration

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key_here
SARVAM_API_KEY=your_sarvam_api_key_here   # optional, only needed for "hinglish"

# Optional overrides
WHISPER_MODEL=small          # tiny / base / small / medium / large
SARVAM_STT_MODEL=saaras:v2.5
```

---

## ▶️ Usage

### 1. Streamlit Web App (recommended)

```bash
streamlit run app.py
```

Open the local URL Streamlit prints, paste a YouTube URL or local file path into the sidebar, choose **english** or **hinglish**, and click **Analyse**. Once processing finishes you'll see the title, summary, transcript, action items, decisions, open questions — plus a chat box to ask follow-up questions.

### 2. Command Line

```bash
python main.py
```

You'll be prompted for a source (YouTube URL or file path) and a language. After the pipeline runs, it drops you into an interactive chat loop (`exit` to quit).

### 3. FastAPI Backend

```bash
uvicorn server:app --reload
```

Exposes:

| Endpoint          | Method | Description |
|-------------------|--------|--------------|
| `/health`         | GET    | Health check |
| `/process`        | POST   | `{ "source": "<url or path>", "language": "english" \| "hinglish" }` — runs the full pipeline |
| `/process-upload` | POST   | Multipart file upload + `language` form field |
| `/chat`           | POST   | `{ "question": "..." }` — ask a question about the most recently processed meeting |

This is meant to sit behind a custom frontend that talks to it over HTTP.

---

## 📝 Notes

- Local files are converted to mono 16 kHz WAV and chunked into 10‑minute segments before transcription.
- Long transcripts are summarized using a map‑reduce strategy so there's no strict input-length limit.
- The RAG chat answers strictly from the transcript — if something isn't in the meeting, the assistant will say so rather than guessing.
- `vector_db/` (Chroma persistence) and `downloades/` (downloaded audio) directories are created automatically at runtime.

---

## 📄 License

No license specified yet — add one if you plan to share or open-source this project.
