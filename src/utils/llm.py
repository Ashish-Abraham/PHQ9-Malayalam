import os
import sys

# Add parent directory to path to allow importing config
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

def get_llm():
    """Returns an instance of the configured LLM provider."""
    from config import LLM_PROVIDER
    from langchain_openai import AzureChatOpenAI, ChatOpenAI
    from dotenv import load_dotenv
    load_dotenv()
    
    if LLM_PROVIDER == "azure":
        return AzureChatOpenAI(
            azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o"),
            openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        )
    
    elif LLM_PROVIDER == "vllm":
        return ChatOpenAI(
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
        
        return HuggingFacePipeline(pipeline=pipe)

    elif LLM_PROVIDER == "groq":
        from langchain_groq import ChatGroq
        
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not found in environment variables.")
            
        from config import GROQ_MODEL
        return ChatGroq(
            temperature=0.6,
            model_name=GROQ_MODEL,
            groq_api_key=api_key
        )
    
    else:
        raise ValueError(f"Unknown LLM_PROVIDER: {LLM_PROVIDER}")

if __name__ == "__main__":
    # try:
        llm = get_llm()
        print("LLM initialized successfully.")
        response = llm.invoke("Hello, are you working?")
        print(f"Response: {response.content}")
    # except Exception as e:
        # print(f"Error: {e}")
