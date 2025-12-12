from langchain_core.tools import tool
from src.utils.vector_db import get_qdrant_client

@tool
def search_guidelines(query: str) -> str:
    """
    Search for medical guidelines, protocols, and PHQ-9 interpretation rules.
    Use this tool when the user asks about scoring, rules, severe/mild depression definitions, or protocols.
    """
    try:
        client = get_qdrant_client()
        # Qdrant with FastEmbed automatically handles embedding the query
        results = client.query(
            collection_name="phq9_docs",
            query_text=query,
            limit=2
        )
        
        if not results:
            return "No relevant guidelines found."
            
        context = "Here are the relevant guidelines found:\n\n"
        for res in results:
            # Depending on how we ingested, content is usually in metadata or 'document'
            # FastEmbed/QdrantClient.add puts text in `document` field of metadata by default or just as document
            # QdrantClient.query (new API) returns QueryResponse which has points. 
            # But wait, we used `client.add`. 
            # The result from `client.query` (fastembed wrapper) returns a list of ScoredPoint.
            # But with FastEmbed it might return objects with payload.
            # Let's check payload.
            
            # Note: client.query supports simplified API.
            # results is List[QueryResponse] which mimics ScoredPoint
            
            # Safer way to extract text:
            doc_content = res.metadata.get("document", "") or res.document
            context += f"- {doc_content}\n"
            
        return context
    except Exception as e:
        return f"Error searching guidelines: {str(e)}"
