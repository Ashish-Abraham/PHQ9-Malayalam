import os
import re
import sys
import pickle
import logging
import asyncio
import numpy as np
import torch
import xgboost as xgb
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from transformers import AutoTokenizer, AutoModel

# Try to import tweet-preprocessor
try:
    from preprocessor import clean
except ImportError:
    logging.warning("tweet-preprocessor not found. Using simple fallback for text cleaning.")
    def clean(text):
        return re.sub(r'http\S+', '', text)

from config import (
    EMOTION_MURIL_PATH, EMOTION_XGBOOST_PATH, EMOTION_LABELS,
    SUICIDE_MURIL_PATH, SUICIDE_XGBOOST_PATH, CSSRS_LABELS
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Spelling correction dictionary
SPELLING_CORRECTIONS = {
    'dont': "don't", 'didnt': "didn't", 'doesnt': "doesn't", 'wont': "won't",
    'cant': "can't", 'shouldnt': "shouldn't", 'wouldnt': "wouldn't",
    'couldnt': "couldn't", 'isnt': "isn't", 'wasnt': "wasn't", 'werent': "weren't",
    'havent': "haven't", 'hasnt': "hasn't", 'hadnt': "hadn't", 'youre': "you're",
    'theyre': "they're", 'were': "we're", 'ive': "I've", 'youve': "you've",
    'theyve': "they've", 'weve': "we've", 'im': "I'm", 'hes': "he's", 'shes': "she's",
    'its': "it's", 'thats': "that's", 'whats': "what's", 'heres': "here's",
    'theres': "there's", 'wheres': "where's", 'yall': "y'all", 'gonna': "going to",
    'wanna': "want to", 'gotta': "got to", 'lemme': "let me", 'gimme': "give me",
    'dunno': "don't know", 'kinda': "kind of", 'sorta': "sort of", 'lol': '',
    'omg': '', 'wtf': '', 'tbh': '', 'imo': '', 'imho': ''
}

class BasePipeline(ABC):
    """Abstract base class for efficient, enterprise-grade ML pipelines."""
    
    def __init__(self, name: str):
        self.name = name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.initialized = False
        self.tokenizer = None
        self.feature_extractor = None
        self.executor = ThreadPoolExecutor(max_workers=1) # Dedicate one thread per pipeline for thread safety
        
    @abstractmethod
    def _load_models(self):
        """Implement specific model loading logic."""
        pass
        
    @abstractmethod
    def clean_text(self, text: str) -> str:
        """Implement specific text cleaning logic."""
        pass
    
    @abstractmethod
    def predict(self, text: str):
        """Implement specific synchronous prediction logic."""
        pass

    async def predict_async(self, text: str):
        """Asynchronous wrapper for prediction to prevent blocking event loop."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, self.predict, text)

    def _load_muril_base(self, model_path: str):
        """Helper to load slightly different MURIL models efficiently."""
        logger.info(f"[{self.name}] Loading tokenizer and model from {model_path}...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            # Use FP16 for GPU to save memory
            if self.device.type == 'cuda':
                self.feature_extractor = AutoModel.from_pretrained(model_path, torch_dtype=torch.float16)
            else:
                self.feature_extractor = AutoModel.from_pretrained(model_path)
                
            self.feature_extractor.to(self.device)
            self.feature_extractor.eval()
        except OSError:
             logger.warning(f"[{self.name}] Could not load local model at {model_path}. Falling back to 'google/muril-base-cased'.")
             self.tokenizer = AutoTokenizer.from_pretrained("google/muril-base-cased")
             self.feature_extractor = AutoModel.from_pretrained("google/muril-base-cased")
             self.feature_extractor.to(self.device)
             self.feature_extractor.eval()

    def extract_features_base(self, texts: list):
        """Shared logic for MURIL feature extraction."""
        if self.feature_extractor is None:
            return np.zeros((len(texts), 768))

        all_features = []
        batch_size = 32
        
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i+batch_size]
                encoded = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=128,
                    return_tensors='pt'
                )
                input_ids = encoded['input_ids'].to(self.device)
                attention_mask = encoded['attention_mask'].to(self.device)
                
                outputs = self.feature_extractor(input_ids=input_ids, attention_mask=attention_mask)
                cls_embeddings = outputs.last_hidden_state[:, 0, :].float().cpu().numpy() # Ensure float32 for XGBoost
                all_features.append(cls_embeddings)
                
        if not all_features:
            return np.zeros((0, 768))
        return np.vstack(all_features)


class EmotionPipeline(BasePipeline):
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(EmotionPipeline, cls).__new__(cls)
            cls._instance.initialized = False
        return cls._instance

    def __init__(self):
        if self.initialized:
            return
        super().__init__("EmotionPipeline")
        self.xgb_models = {}
        self.labels = EMOTION_LABELS
        self._load_models()
        self.initialized = True
        logger.info("[EmotionPipeline] Initialized successfully.")

    def _load_models(self):
        self._load_muril_base(EMOTION_MURIL_PATH)
        
        if os.path.exists(EMOTION_XGBOOST_PATH):
            logger.info(f"[EmotionPipeline] Loading XGBoost models from {EMOTION_XGBOOST_PATH}...")
            with open(EMOTION_XGBOOST_PATH, 'rb') as f:
                self.xgb_models = pickle.load(f)
        else:
            logger.warning(f"[EmotionPipeline] XGBoost model not found at {EMOTION_XGBOOST_PATH}. Dummy inference.")
            self.xgb_models = {}

    def clean_text(self, text: str) -> str:
        if not text: return ""
        text = clean(text)
        text = text.replace('\xe2\x80\x99', "'").replace('\x27', "'")
        text = text.replace('\xe2\x80\x93', "-").replace('\xe2\x80\x94', "-")
        text = re.sub(r'[^a-zA-Z0-9\s\.,!?\'\"-]', '', text)
        words = text.split()
        words = [SPELLING_CORRECTIONS.get(word.lower(), word) for word in words]
        text = ' '.join(words)
        return re.sub(r'\s+', ' ', text).strip()

    def predict(self, text: str, threshold: float = 0.5):
        cleaned_text = self.clean_text(text)
        features = self.extract_features_base([cleaned_text])
        probabilities = {}
        predictions = {}

        if not self.xgb_models:
             return {'emotions': [], 'probabilities': {}}
        
        for emotion, model in self.xgb_models.items():
            try:
                probs = model.predict_proba(features)[:, 1]
                prob = float(probs[0])
                probabilities[emotion] = prob
                if prob > threshold:
                    predictions[emotion] = prob
            except Exception as e:
                logger.error(f"[EmotionPipeline] Error predicting {emotion}: {e}")
                probabilities[emotion] = 0.0
                
        sorted_emotions = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)[:5]
        return {
            'top_emotions': [e[0] for e in sorted_emotions],
            'probabilities': {e[0]: e[1] for e in sorted_emotions},
            'all_scores': probabilities
        }


class SuicideRiskPipeline(BasePipeline):
    _instance = None
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(SuicideRiskPipeline, cls).__new__(cls)
            cls._instance.initialized = False
        return cls._instance
    def __init__(self):
        if self.initialized:
            return
        super().__init__("SuicideRiskPipeline")
        self.xgb_model = None
        self.labels = CSSRS_LABELS
        self._load_models()
        self.initialized = True
        logger.info("[SuicideRiskPipeline] Initialized successfully.")
    def _load_models(self):
        self._load_muril_base(SUICIDE_MURIL_PATH)
        
        if os.path.exists(SUICIDE_XGBOOST_PATH):
            logger.info(f"[SuicideRiskPipeline] Loading XGBoost model from {SUICIDE_XGBOOST_PATH}...")
            self.xgb_model = xgb.XGBClassifier()
            self.xgb_model.load_model(SUICIDE_XGBOOST_PATH)
        else:
            logger.warning(f"[SuicideRiskPipeline] XGBoost model not found at {SUICIDE_XGBOOST_PATH}. Dummy inference.")
            self.xgb_model = None
    def clean_text(self, text: str) -> str:
        if not text: return ""
        # Specific reddit cleaning
        text = re.sub(r'http\S+|www\S+|https\S+', '[URL]', text, flags=re.MULTILINE)
        text = re.sub(r'\[deleted\]|\[removed\]', '', text)
        text = re.sub(r'[^\w\s.,!?;:\'\"-]', ' ', text)
        return re.sub(r'\s+', ' ', text).strip()
    def predict_risk(self, text: str): # Alias for consistency or specific naming
        return self.predict(text)
    def predict(self, text: str):
        cleaned_text = self.clean_text(text)
        if len(cleaned_text) < 10:
             return {'label': 'Supportive', 'label_id': 0, 'alert': False, 'probabilities': {}}
             
        features = self.extract_features_base([cleaned_text])
        if self.xgb_model is None:
             return {'label': 'Supportive', 'label_id': 0, 'alert': False, 'probabilities': {}}
        try:
            probs = self.xgb_model.predict_proba(features)[0]
            pred_id = int(np.argmax(probs))
            label = self.labels.get(pred_id, "Unknown")
            alert = pred_id >= 3 
            return {
                'label': label,
                'label_id': pred_id,
                'alert': alert,
                'probabilities': {self.labels[i]: float(p) for i, p in enumerate(probs)}
            }
        except Exception as e:
            logger.error(f"[SuicideRiskPipeline] Error predicting: {e}")
            return {'label': 'Supportive', 'label_id': 0, 'alert': False}


# Global instances & Accessors
pipeline_instance = None
suicide_pipeline_instance = None

def get_pipeline():
    global pipeline_instance
    if pipeline_instance is None:
        pipeline_instance = EmotionPipeline()
    return pipeline_instance

def get_suicide_pipeline():
    global suicide_pipeline_instance
    if suicide_pipeline_instance is None:
        suicide_pipeline_instance = SuicideRiskPipeline()
    return suicide_pipeline_instance

def detect_emotion(text: str) -> str:
    """Wrapper."""
    if os.environ.get("DISABLE_PIPELINES"):
        return "neutral"
    pipe = get_pipeline()
    result = pipe.predict(text)
    if result['top_emotions']:
        return result['top_emotions'][0]
    return "neutral"

def detect_suicidal_language(text: str) -> bool:
    """Wrapper."""
    if os.environ.get("DISABLE_PIPELINES"):
        return False
    pipe = get_suicide_pipeline()
    result = pipe.predict(text) # Uses predict (alias predict_risk logic)
    return result.get('alert', False)

