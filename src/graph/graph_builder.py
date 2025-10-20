import json
import re
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Dict, Any

class GraphBuilder:
    def __init__(self, uri: str, username: str, password: str):
        self.driver = GraphDatabase.driver(uri, auth=(username, password))
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
    def close(self):
        self.driver.close()
    
    def create_constraints_and_indexes(self):
        with self.driver.session() as session:
            constraints = [
                "CREATE CONSTRAINT IF NOT EXISTS FOR (kb:KnowledgeBase) REQUIRE kb.name IS UNIQUE",
                "CREATE CONSTRAINT IF NOT EXISTS FOR (e:Entry) REQUIRE e.id IS UNIQUE",
                "CREATE CONSTRAINT IF NOT EXISTS FOR (f:Fact) REQUIRE f.id IS UNIQUE",
                "CREATE CONSTRAINT IF NOT EXISTS FOR (c:Concept) REQUIRE c.name IS UNIQUE"
            ]
            
            indexes = [
                "CREATE INDEX IF NOT EXISTS FOR (e:Entry) ON (e.title)",
                "CREATE FULLTEXT INDEX entry_content IF NOT EXISTS FOR (e:Entry) ON EACH [e.content]",
                "CREATE FULLTEXT INDEX fact_content IF NOT EXISTS FOR (f:Fact) ON EACH [f.text]"
            ]
            
            for constraint in constraints:
                try:
                    session.run(constraint)
                    print(f"✓ Created constraint: {constraint}")
                except Exception as e:
                    print(f"Constraint already exists or error: {e}")
            
            for index in indexes:
                try:
                    session.run(index)
                    print(f"✓ Created index: {index}")
                except Exception as e:
                    print(f"Index already exists or error: {e}")
    
    def extract_concepts(self, text: str) -> List[str]:
        # Generalize concept extraction for animal data
        keywords = [
            'mammal', 'bird', 'fish', 'reptile', 'amphibian', 'carnivore', 'herbivore', 'omnivore',
            'savanna', 'ocean', 'forest', 'desert', 'arctic', 'habitat', 'diet', 'species',
            'pride', 'flock', 'school', 'pack', 'herd', 'colony', 'group', 'social',
            'echolocation', 'migration', 'camouflage', 'predator', 'prey', 'extinct', 'endangered'
        ]
        
        text_lower = text.lower()
        found_concepts = [kw for kw in keywords if kw in text_lower]
        return list(set(found_concepts))
    
    def generate_embedding(self, text: str) -> List[float]:
        embedding = self.embedding_model.encode(text)
        return embedding.tolist()
    
    def ingest_knowledge_base(self, json_file_path: str):
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        kb_data = data['knowledge_base']
        
        with self.driver.session() as session:
            # Create KnowledgeBase node
            kb_id = self._create_knowledge_base_node(session, kb_data)
            kb_name = kb_data['name']
            print(f"✓ Created KnowledgeBase: {kb_name}")
            
            # Process each entry
            for entry in kb_data.get('entries', []):
                self._process_entry(session, kb_id, entry, kb_name)

            for entry in kb_data.get('entries', []):
                for related_animal in entry.get('related_animals', []):
                    entry_id = f"{kb_name}:Entry:{entry['title']}"
                    related_animal_id = f"{kb_name}:Entry:{related_animal}"
                    self._create_related_animal_relationship(session, entry_id, related_animal_id)
            
            print(f"✓ Completed ingestion of {kb_data['name']}")
    
    def _create_knowledge_base_node(self, session, kb_data: Dict) -> str:
        query = """
        MERGE (kb:KnowledgeBase {name: $name})
        SET kb.description = $description,
            kb.created_at = datetime()
        RETURN kb.name as id
        """
        
        result = session.run(query,
            name=kb_data['name'],
            description=kb_data.get('description')
        )
        return result.single()['id']
    
    def _process_entry(self, session, kb_id: str, entry: Dict, kb_name: str):
        content = " ".join(entry.get('facts', []))
        embedding = self.generate_embedding(content)
        
        concepts = self.extract_concepts(content)
        
        entry_query = """
        MATCH (kb:KnowledgeBase {name: $kb_name})
        MERGE (e:Entry {id: $entry_id})
        SET e.title = $title,
            e.category = $category,
            e.habitat = $habitat,
            e.diet = $diet,
            e.content = $content,
            e.embedding = $embedding
        MERGE (kb)-[:CONTAINS]->(e)
        RETURN e.id as entry_id
        """
        
        entry_id = f"{kb_name}:Entry:{entry['title']}"
        
        session.run(entry_query,
            kb_name=kb_id,
            entry_id=entry_id,
            title=entry.get('title', ''),
            category=entry.get('category', ''),
            habitat=entry.get('habitat', ''),
            diet=entry.get('diet', ''),
            content=content,
            embedding=embedding
        )
        
        for concept in concepts:
            concept_query = """
            MATCH (e:Entry {id: $entry_id})
            MERGE (c:Concept {name: $concept})
            MERGE (e)-[:RELATES_TO]->(c)
            """
            session.run(concept_query, entry_id=entry_id, concept=concept)
        
        for fact_text in entry.get('facts', []):
            self._process_fact(session, entry_id, fact_text)
    
    def _process_fact(self, session, entry_id: str, fact_text: str):
        fact_id = f"{entry_id}:Fact:{hash(fact_text)}" # Simple hash for unique ID
        
        fact_query = """
        MATCH (e:Entry {id: $entry_id})
        MERGE (f:Fact {id: $fact_id})
        SET f.text = $text
        MERGE (e)-[:HAS_FACT]->(f)
        """
        session.run(fact_query,
            entry_id=entry_id,
            fact_id=fact_id,
            text=fact_text
        )
            
    def _create_related_animal_relationship(self, session, entry_id: str, related_animal_id: str):
        query = """
        MATCH (e1:Entry {id: $entry_id})
        MATCH (e2:Entry {id: $related_animal_id})
        MERGE (e1)-[:RELATED_TO]->(e2)
        """
        session.run(query, entry_id=entry_id, related_animal_id=related_animal_id)
