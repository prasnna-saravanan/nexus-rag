"""
HyDE (Hypothetical Document Embeddings) Service.

Implementation:
1. User asks a question
2. Generate a hypothetical answer using LLM
3. Embed the hypothetical answer
4. Search with the hypothetical answer embedding
5. This bridges the gap between user questions and formal document text

Use Case: SOPs - User asks "What do I do if supplier fails audit?"
- HyDE generates: "The procedure for supplier audit failure is..."
- This matches better with formal SOP text than the original question
"""
from typing import List, Dict, Any
import openai
from app.core.config import Settings
from app.models.schemas import DocumentType


class HyDEService:
    """
    Hypothetical Document Embeddings service.
    
    Improves retrieval for formal documents by generating
    hypothetical answers that match document style.
    """
    
    def __init__(self, settings: Settings):
        """
        Initialize HyDE service.
        
        Args:
            settings: Application settings
        """
        self.settings = settings
        self.client = openai.AsyncOpenAI(api_key=settings.openai_api_key)
    
    async def generate_hypothetical_document(
        self,
        query: str,
        document_type: DocumentType
    ) -> str:
        """
        Generate a hypothetical document that answers the query.
        
        Args:
            query: User's question
            document_type: Type of document to match style
        
        Returns:
            Hypothetical document text
        """
        # Build prompt based on document type
        system_prompt = self._get_system_prompt(document_type)
        
        user_prompt = f"""Generate a hypothetical document excerpt that would answer this question:

Question: {query}

Generate a formal, detailed answer in the style of the document type specified. Be specific and use appropriate terminology."""
        
        # Call LLM
        response = await self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.7,
            max_tokens=300
        )
        
        return response.choices[0].message.content
    
    def _get_system_prompt(self, document_type: DocumentType) -> str:
        """Get system prompt based on document type."""
        prompts = {
            DocumentType.SOP: """You are generating excerpts from Standard Operating Procedures (SOPs).
Write in a formal, procedural style with step-by-step instructions.
Use phrases like "The procedure is...", "Follow these steps:", "In accordance with policy...""",
            
            DocumentType.MASTER_DATA: """You are generating product or vendor data records.
Write in a structured, factual style with specifications and details.
Use phrases like "Product specifications:", "Vendor details:", "Category:".""",
            
            DocumentType.INVOICE: """You are generating invoice or purchase order descriptions.
Write in a transactional, formal style with line items and amounts.
Use phrases like "Invoice total:", "Line items include:", "Payment terms:".""",
            
            DocumentType.EMAIL: """You are generating business email content.
Write in a professional, concise style.
Use phrases like "Regarding:", "Please note:", "As discussed:"."""
        }
        
        return prompts.get(
            document_type,
            "You are generating formal business document excerpts."
        )

