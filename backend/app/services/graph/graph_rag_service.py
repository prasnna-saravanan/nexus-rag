"""
Graph RAG Service for supply chain risk analysis.

Combines graph traversal with vector retrieval.
"""
from typing import List, Dict, Any, Optional
import openai
from app.core.config import Settings
from app.services.graph.neo4j_client import Neo4jClient


class GraphRAGService:
    """Graph RAG service for multi-hop reasoning in supply chain operations."""
    
    def __init__(self, settings: Settings, neo4j_client: Neo4jClient):
        """
        Initialize Graph RAG service.
        
        Args:
            settings: Application settings
            neo4j_client: Neo4j client for graph operations
        """
        self.settings = settings
        self.neo4j = neo4j_client
        self.client = openai.AsyncOpenAI(api_key=settings.openai_api_key)
    
    async def query_graph_rag(
        self,
        query: str,
        max_hops: int = 3,
        vector_context: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Execute Graph RAG query.
        
        Steps:
        1. Extract entities from query
        2. Find graph paths
        3. Combine with vector context
        4. Generate answer
        
        Args:
            query: User's question
            max_hops: Maximum hops in graph traversal
            vector_context: Optional vector search results
        
        Returns:
            Dict with answer, paths, entities, relationships
        """
        # Step 1: Extract entities from query using LLM
        entities = await self._extract_entities(query)
        
        if not entities:
            return {
                "answer": "No relevant entities found in the knowledge graph.",
                "graph_paths": [],
                "entities_involved": [],
                "relationships": []
            }
        
        # Step 2: Find paths in graph
        all_paths = []
        for entity_id in entities[:3]:  # Limit to top 3 entities
            paths = self.neo4j.find_paths(
                start_entity_id=entity_id,
                max_hops=max_hops
            )
            all_paths.extend(paths)
        
        # Step 3: Extract unique entities and relationships from paths
        unique_entities = self._extract_entities_from_paths(all_paths)
        relationships = self._extract_relationships_from_paths(all_paths)
        
        # Step 4: Build context for LLM
        graph_context = self._build_graph_context(all_paths)
        
        # Step 5: Generate answer
        answer = await self._generate_answer(query, graph_context, vector_context)
        
        return {
            "answer": answer,
            "graph_paths": [[node["name"] for node in path] for path in all_paths],
            "entities_involved": unique_entities,
            "relationships": relationships
        }
    
    async def _extract_entities(self, query: str) -> List[str]:
        """
        Extract entity IDs from query text.
        
        Uses Neo4j pattern matching to find mentioned entities.
        """
        # Simple keyword extraction
        # In production, use NER (spaCy, GPT-4, etc.)
        words = query.lower().split()
        
        # Search for entities by name
        entity_ids = []
        for word in words:
            if len(word) > 3:  # Skip short words
                matches = self.neo4j.query_by_pattern(word)
                entity_ids.extend([m["id"] for m in matches])
        
        return list(set(entity_ids))[:5]  # Dedupe and limit
    
    def _extract_entities_from_paths(self, paths: List[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """Extract unique entities from graph paths."""
        entities_dict = {}
        
        for path in paths:
            for node in path:
                entity_id = node.get("id")
                if entity_id and entity_id not in entities_dict:
                    entities_dict[entity_id] = {
                        "id": entity_id,
                        "type": node.get("type", "Unknown"),
                        "name": node.get("name", ""),
                        "properties": node.get("properties", {})
                    }
        
        return list(entities_dict.values())
    
    def _extract_relationships_from_paths(self, paths: List[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """Extract relationships from graph paths."""
        relationships = []
        
        for path in paths:
            for i in range(len(path) - 1):
                from_node = path[i]
                to_node = path[i + 1]
                
                # Infer relationship (in production, get from graph)
                relationships.append({
                    "from_entity": from_node.get("id"),
                    "to_entity": to_node.get("id"),
                    "relationship_type": "CONNECTED_TO",
                    "properties": {}
                })
        
        return relationships
    
    def _build_graph_context(self, paths: List[List[Dict[str, Any]]]) -> str:
        """Build human-readable context from graph paths."""
        if not paths:
            return "No graph paths found."
        
        context_parts = ["=== Knowledge Graph Context ===\n"]
        
        for idx, path in enumerate(paths[:5], 1):  # Limit to 5 paths
            path_desc = " → ".join([node.get("name", "Unknown") for node in path])
            context_parts.append(f"Path {idx}: {path_desc}")
        
        return "\n".join(context_parts)
    
    async def _generate_answer(
        self,
        query: str,
        graph_context: str,
        vector_context: Optional[List[Dict[str, Any]]] = None
    ) -> str:
        """
        Generate answer using LLM with graph + vector context.
        """
        # Build system prompt
        system_prompt = """You are a supply chain risk analysis assistant.
You have access to a knowledge graph showing relationships between suppliers, products, ports, and events.
Use the graph paths to explain how events propagate through the supply chain.
Be specific about the relationships and entities involved."""
        
        # Build user prompt
        user_prompt_parts = [
            graph_context,
            ""
        ]
        
        if vector_context:
            user_prompt_parts.append("\n=== Additional Context from Documents ===")
            for item in vector_context[:3]:
                user_prompt_parts.append(f"- {item.get('text', '')[:200]}...")
        
        user_prompt_parts.append(f"\nQuestion: {query}\n")
        user_prompt_parts.append("Please answer the question using the graph paths and context provided.")
        
        user_prompt = "\n".join(user_prompt_parts)
        
        # Call LLM
        response = await self.client.chat.completions.create(
            model="gpt-4",  # Use GPT-4 for complex reasoning
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.7,
            max_tokens=600
        )
        
        return response.choices[0].message.content

