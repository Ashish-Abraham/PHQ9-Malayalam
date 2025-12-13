# LLM Provider Configuration
# Options: "azure", "vllm", "huggingface", "groq"
LLM_PROVIDER = "groq"

GROQ_MODEL = "llama-3.1-8b-instant"

# Emotion Pipeline Configuration
EMOTION_MURIL_PATH = "/content/muril_cssrs_finetuned" # Specific fine-tuned model path
EMOTION_XGBOOST_PATH = "/content/xgboost_emotion_models.pkl"

EMOTION_LABELS = [
    'admiration', 'amusement', 'anger', 'annoyance', 'approval',
    'caring', 'confusion', 'curiosity', 'desire', 'disappointment',
    'disapproval', 'disgust', 'embarrassment', 'excitement', 'fear',
    'gratitude', 'grief', 'joy', 'love', 'nervousness', 'optimism',
    'pride', 'realization', 'relief', 'remorse', 'sadness', 'surprise',
    'neutral'
]

# Suicide Risk Pipeline Configuration
SUICIDE_MURIL_PATH = "/content/muril_cssrs_finetuned" # Specific fine-tuned model path
SUICIDE_XGBOOST_PATH = "/content/xgboost_cssrs_model.json"

# Labels 0-4
CSSRS_LABELS = {
    0: 'Supportive',
    1: 'Indicator',
    2: 'Ideation',
    3: 'Behavior',
    4: 'Attempt'
}
