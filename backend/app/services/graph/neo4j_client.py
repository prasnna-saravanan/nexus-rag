"""
Neo4j Graph Database Client for Graph RAG.

Handles supply chain knowledge graph operations:
- Entity creation (Suppliers, Products, Ports, Events)
- Relationship mapping (SUPPLIES, SHIPS_VIA, DEPENDS_ON, AFFECTS)
- Graph traversal for risk analysis
"""
from typing import List, Dict, Any, Optional
from neo4j import GraphDatabase
from app.core.config import Settings


class Neo4jClient:
    """
    Neo4j client for Graph RAG operations.
    
    Use Cases:
    - Supply chain risk analysis
    - Relationship discovery
    - Multi-hop reasoning
    """
    
    def __init__(self, settings: Settings):
        """
        Initialize Neo4j client.
        
        Args:
            settings: Application settings with Neo4j config
        """
        self.settings = settings
        self.driver = GraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_user, settings.neo4j_password)
        )
    
    def close(self):
        """Close the Neo4j driver."""
        if self.driver:
            self.driver.close()
    
    def health_check(self) -> bool:
        """Check if Neo4j is reachable."""
        try:
            with self.driver.session() as session:
                result = session.run("RETURN 1")
                return result.single()[0] == 1
        except Exception:
            return False
    
    def create_entity(
        self,
        entity_id: str,
        entity_type: str,
        name: str,
        properties: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Create or update an entity node.
        
        Args:
            entity_id: Unique identifier
            entity_type: Type (Supplier, Product, Port, Event, etc.)
            name: Display name
            properties: Additional properties
        """
        props = properties or {}
        props['id'] = entity_id
        props['name'] = name
        
        with self.driver.session() as session:
            query = f"""
            MERGE (n:{entity_type} {{id: $id}})
            SET n += $properties
            RETURN n
            """
            session.run(query, id=entity_id, properties=props)
            return True
    
    def create_relationship(
        self,
        from_id: str,
        to_id: str,
        relationship_type: str,
        properties: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Create a relationship between entities.
        
        Args:
            from_id: Source entity ID
            to_id: Target entity ID
            relationship_type: Type (SUPPLIES, SHIPS_VIA, DEPENDS_ON, AFFECTS)
            properties: Relationship metadata
        """
        props = properties or {}
        
        with self.driver.session() as session:
            query = f"""
            MATCH (a {{id: $from_id}})
            MATCH (b {{id: $to_id}})
            MERGE (a)-[r:{relationship_type}]->(b)
            SET r += $properties
            RETURN r
            """
            session.run(query, from_id=from_id, to_id=to_id, properties=props)
            return True
    
    def find_paths(
        self,
        start_entity_id: str,
        end_entity_id: Optional[str] = None,
        max_hops: int = 3,
        relationship_types: Optional[List[str]] = None
    ) -> List[List[Dict[str, Any]]]:
        """
        Find paths between entities (for supply chain risk analysis).
        
        Args:
            start_entity_id: Starting entity (e.g., Strike Event)
            end_entity_id: Optional target entity (e.g., Purchase Order)
            max_hops: Maximum relationship hops
            relationship_types: Filter by relationship types
        
        Returns:
            List of paths, where each path is a list of nodes
        """
        with self.driver.session() as session:
            if end_entity_id:
                # Find shortest paths between two entities
                query = """
                MATCH path = shortestPath(
                    (start {id: $start_id})-[*..%d]-(end {id: $end_id})
                )
                RETURN [node IN nodes(path) | {
                    id: node.id,
                    type: labels(node)[0],
                    name: node.name,
                    properties: properties(node)
                }] as path
                LIMIT 10
                """ % max_hops
                
                result = session.run(query, start_id=start_entity_id, end_id=end_entity_id)
            else:
                # Explore outward from starting entity
                query = """
                MATCH path = (start {id: $start_id})-[*..%d]-(connected)
                RETURN [node IN nodes(path) | {
                    id: node.id,
                    type: labels(node)[0],
                    name: node.name,
                    properties: properties(node)
                }] as path
                LIMIT 20
                """ % max_hops
                
                result = session.run(query, start_id=start_entity_id)
            
            paths = [record["path"] for record in result]
            return paths
    
    def query_by_pattern(
        self,
        query_text: str,
        max_results: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Find entities by name pattern (for entity extraction from queries).
        
        Args:
            query_text: Search text
            max_results: Maximum results
        
        Returns:
            List of matching entities
        """
        with self.driver.session() as session:
            query = """
            MATCH (n)
            WHERE toLower(n.name) CONTAINS toLower($search_text)
            RETURN {
                id: n.id,
                type: labels(n)[0],
                name: n.name,
                properties: properties(n)
            } as entity
            LIMIT $limit
            """
            
            result = session.run(query, search_text=query_text, limit=max_results)
            return [record["entity"] for record in result]
    
    def get_entity_context(
        self,
        entity_id: str,
        max_relationships: int = 20
    ) -> Dict[str, Any]:
        """
        Get full context for an entity (all relationships).
        
        Args:
            entity_id: Entity to get context for
            max_relationships: Max relationships to return
        
        Returns:
            Dict with entity and relationships
        """
        with self.driver.session() as session:
            query = """
            MATCH (n {id: $entity_id})
            OPTIONAL MATCH (n)-[r]-(connected)
            RETURN {
                entity: {
                    id: n.id,
                    type: labels(n)[0],
                    name: n.name,
                    properties: properties(n)
                },
                relationships: collect({
                    type: type(r),
                    direction: CASE 
                        WHEN startNode(r) = n THEN 'outgoing'
                        ELSE 'incoming'
                    END,
                    connected_entity: {
                        id: connected.id,
                        type: labels(connected)[0],
                        name: connected.name
                    }
                })[0..%d]
            } as context
            """ % max_relationships
            
            result = session.run(query, entity_id=entity_id)
            record = result.single()
            return record["context"] if record else {}
    
    def clear_graph(self) -> bool:
        """Clear all nodes and relationships (for testing)."""
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
            return True

