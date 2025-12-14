# PHQ-9 Mental Health Chatbot

An empathetic, AI-powered mental health chatbot that administers the PHQ-9 depression screening, assesses risk, and provides personalized advice using a Knowledge Graph and LLM.

## Features

- **PHQ-9 Screening**: Interactive administration of the standard PHQ-9 questionnaire.
- **Risk Assessment**: Real-time analysis of suicide risk and emotional state (using MURIL + XGBoost pipelines).
- **Conversational Advice**: Providing empathetic, context-aware advice with the ability to ask follow-up questions.
- **Context Management**: Automatic conversation summarization to handle long sessions.
- **RAG Integration**: Retrieval-Augmented Generation for medical guidelines and protocols.
- **Gradio Interface**: User-friendly web interface for chat.

## Setup

### Prerequisites

- Python 3.10+
- Azure OpenAI API Key OR Groq API Key
- (Optional) GPU for local ML models

### Installation

1.  **Clone the repository** (if you haven't already).
2.  **Create a virtual environment**:
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```
3.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

### Configuration

1.  Copy `.env.example` to `.env`:
    ```bash
    cp .env.example .env
    ```
2.  Edit `.env` and fill in your API keys:
    *   **LLM Provider**: Set `AZURE_OPENAI_API_KEY` and endpoint details OR `GROQ_API_KEY`.
    *   **Pipelines**: By default, the ML pipelines (Emotion/Suicide Risk) load on startup. This requires significant RAM/GPU.
    *   **Testing**: Set `DISABLE_PIPELINES=1` in `.env` (or env var) to skip model loading for faster dev/testing.

### Google Colab Setup

If running in Google Colab:
1.  **Public URL**: You must enable `share=True` in `src/gradio_app.py`. Find the `demo.launch()` line and change it to `demo.launch(...,share=True)`.
2.  **Model Paths**: Before running the full mode, open `src/config.py` and update `EMOTION_MURIL_PATH`, `EMOTION_XGBOOST_PATH`, `SUICIDE_MURIL_PATH`, and `SUICIDE_XGBOOST_PATH` to point to the correct locations of your uploaded model files.

## Running the Application

### Light Mode (Recommended for testing chat logic)

Run without loading the heavy ML models (only chat):

```bash
DISABLE_PIPELINES=1 python3 src/gradio_app.py
```

### Full Mode (Production)

Run with all ML pipelines enabled:

```bash
python3 src/gradio_app.py
```

Access the web interface at `http://localhost:7860` and dashboard at `http://localhost:7860/dashboard`.



## Architecture

- **State Management**: Uses `LangGraph` for stateful transitions (Rapport -> Permission -> Questionnaire -> Advice).
- **Nodes**: Logical units handling different phases of the conversation (`src/nodes/`).
- **Graph**: Defined in `src/graph.py`, managing the flow and routing.
