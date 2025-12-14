import os
import sys

# Add parent directory to path to allow importing config
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


import time
import random

class SafeLLM:
    """
    Wrapper around LLM to handle transient errors (especially Groq 503) 
    with exponential backoff.
    """
    def __init__(self, llm):
        self.llm = llm

    def invoke(self, *args, **kwargs):
        max_retries = 10
        base_delay = 2
        
        for attempt in range(max_retries):
            try:
                return self.llm.invoke(*args, **kwargs)
            except Exception as e:
                error_str = str(e).lower()
                # Catch specific 503/Capacity errors or generic internal server errors
                if "503" in error_str or "capacity" in error_str or "internal_server_error" in error_str or "rate limit" in error_str:
                     if attempt < max_retries - 1:
                        # Full jitter: sleep = random_between(0, base * 2^attempt)
                        # But simple exponential is fine too: base * 2^attempt
                        sleep_time = base_delay * (2 ** attempt) + random.uniform(0, 1)
                        print(f"[SafeLLM] Error (Attempt {attempt+1}/{max_retries}): {e}. Retrying in {sleep_time:.2f}s...")
                        time.sleep(sleep_time)
                        continue
                raise e

    def bind_tools(self, *args, **kwargs):
        # Allow binding tools, returning a new SafeLLM wrapping the bound runnable
        bound = self.llm.bind_tools(*args, **kwargs)
        return SafeLLM(bound)

    def __getattr__(self, name):
         # Delegate other attributes/methods
         return getattr(self.llm, name)

def get_llm():
    """Returns an instance of the configured LLM provider, wrapped for safety."""
    from config import LLM_PROVIDER
    from langchain_openai import AzureChatOpenAI, ChatOpenAI
    from dotenv import load_dotenv
    load_dotenv()
    
    llm_instance = None
    
    if LLM_PROVIDER == "azure":
        llm_instance = AzureChatOpenAI(
            azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o"),
            openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        )
    
    elif LLM_PROVIDER == "vllm":
        llm_instance = ChatOpenAI(
            model="google/gemma-3-4b-it",
            openai_api_key="EMPTY",
            openai_api_base="",
            max_tokens=3500,
            temperature=0.6,
            model_kwargs={
                "top_p": 0.95,
                "extra_body": {
                    "top_k": 20,
                    "thinking": False
                }
            }
        )
        
    elif LLM_PROVIDER == "huggingface":
        # Import here to avoid dependency issues if not using this provider
        from langchain_huggingface import HuggingFacePipeline
        from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
        import torch

        model_id = "google/gemma-3-4b-it"
        
        # We assume the user has logged in or the model is public/accessible
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )
        
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=3500,
            temperature=0.6,
            top_p=0.95,
            top_k=20,
            # thinking=False # 'thinking' param might not be standard in pipeline, check if supported or ignored
        )
        
        llm_instance = HuggingFacePipeline(pipeline=pipe)

    elif LLM_PROVIDER == "groq":
        from langchain_groq import ChatGroq
        
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not found in environment variables.")
            
        from config import GROQ_MODEL
        llm_instance = ChatGroq(
            temperature=0.6,
            model_name=GROQ_MODEL,
            groq_api_key=api_key,
            max_retries=5, # Increase default retries
            request_timeout=60 # Add timeout
        )
    
    else:
        raise ValueError(f"Unknown LLM_PROVIDER: {LLM_PROVIDER}")
        
    # Wrap in SafeLLM
    return SafeLLM(llm_instance)


def get_llm_for_small_tasks():
    """Returns an instance of the configured LLM provider, wrapped for safety."""
    from config import LLM_PROVIDER
    from langchain_openai import AzureChatOpenAI, ChatOpenAI
    from dotenv import load_dotenv
    load_dotenv()
    
    llm_instance = None
    
    if LLM_PROVIDER == "azure":
        llm_instance = AzureChatOpenAI(
            azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o-mini"),
            openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        )

    elif LLM_PROVIDER == "groq":
        from langchain_groq import ChatGroq
        
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not found in environment variables.")
            
        from config import GROQ_MODEL2
        llm_instance = ChatGroq(
            temperature=0.6,
            model_name=GROQ_MODEL2,
            groq_api_key=api_key,
            max_retries=5, # Increase default retries
            request_timeout=60 # Add timeout
        )
    
    else:
        raise ValueError(f"Unknown LLM_PROVIDER: {LLM_PROVIDER}")
        
    # Wrap in SafeLLM
    return SafeLLM(llm_instance)    


if __name__ == "__main__":
    # try:
        llm = get_llm()
        print("LLM initialized successfully.")
        response = llm.invoke("Hello, are you working?")
        print(f"Response: {response.content}")
    # except Exception as e:
        # print(f"Error: {e}")
