import requests
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Dict, Any

class GraphRAG:
    def __init__(self,
                neo4j_uri: str,
                neo4j_user: str,
                neo4j_password: str,
                ollama_url: str,
                ollama_model : str = "Llama3.1:8b"):
        self.driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.ollama_url = ollama_url
        self.model_name = ollama_model
        
    def close(self):
        self.driver.close()
    
    def semantic_search(self, query: str, top_k: int = 5) -> List[Dict]:
        query_embedding = self.embedding_model.encode(query)
        
        with self.driver.session() as session:
            cypher_query = """
            CALL db.index.vector.queryNodes(
              'entry_embeddings',
              $top_k,
              $query_embedding
            )
            YIELD node, score
            MATCH (kb:KnowledgeBase)-[:CONTAINS]->(node)
            RETURN node.id as id,
                   node.title as title,
                   node.category as category,
                   node.content as content,
                   kb.name as knowledge_base_name,
                   score
            """
            results = session.run(
            cypher_query,
            parameters={'query_embedding': query_embedding, 'top_k': top_k})
        
            entries = []
            for record in results:
                entries.append({
                    'id': record['id'],
                    'title': record['title'],
                    'category': record['category'],
                    'content': record['content'],
                    'knowledge_base_name': record['knowledge_base_name'],
                    'similarity': float(record['score']) # Score is the similarity
                })
            
            return entries
    
    def keyword_search(self, query: str, top_k: int = 5) -> List[Dict]:
        with self.driver.session() as session:
            cypher_query = """
            CALL db.index.fulltext.queryNodes('entry_content', $query)
            YIELD node, score
            MATCH (kb:KnowledgeBase)-[:CONTAINS]->(node)
            RETURN node.id as id,
                   node.title as title,
                   node.category as category,
                   node.content as content,
                   kb.name as knowledge_base_name,
                   score
            LIMIT $top_k
            """
            
            results = session.run(cypher_query, parameters={'query': query, 'top_k': top_k})
            
            return [dict(record) for record in results]
    
    def get_related_entries(self, entry_id: str, depth: int = 2) -> List[Dict]:
        with self.driver.session() as session:
            cypher_query = """
            MATCH path = (e:Entry {id: $entry_id})-[:SIMILAR_TO|RELATES_TO|RELATED_TO*1..""" + str(depth) + """]->(related)
            WHERE related:Entry
            MATCH (kb:KnowledgeBase)-[:CONTAINS]->(related)
            RETURN DISTINCT related.id as id,
                   related.title as title,
                   related.category as category,
                   related.content as content,
                   kb.name as knowledge_base_name,
                   length(path) as distance
            ORDER BY distance
            LIMIT 5
            """
            
            results = session.run(cypher_query, entry_id=entry_id)
            return [dict(record) for record in results]
    
    def hybrid_search(self, query: str, top_k: int = 10) -> List[Dict]:
        semantic_results = self.semantic_search(query, top_k=20)

        print("Semantic Results:")
        for res in semantic_results:
            print(f"{res['knowledge_base_name']} - Entry {res['title']} (Score: {res['similarity']})")
        
        keyword_results = self.keyword_search(query, top_k=5)
        
        related_entries_results = []
        if semantic_results:
            top_semantic_entry_id = semantic_results[0]['id']
            related_entries_results = self.get_related_entries(top_semantic_entry_id, depth=1)
            print("Related Entries Results:")
            for res in related_entries_results:
                print(f"{res['knowledge_base_name']} - Entry {res['title']} (Distance: {res['distance']})")
        
        all_results = {}
        
        for result in semantic_results:
            all_results[result['id']] = {
                **result,
                'semantic_score': result['similarity'],
                'keyword_score': 0,
                'related_score': 0,
            }
        
        for result in keyword_results:
            if result['id'] in all_results:
                all_results[result['id']]['keyword_score'] = result['score']
            else:
                all_results[result['id']] = {
                    **result,
                    'semantic_score': 0,
                    'keyword_score': result['score'],
                    'related_score': 0,
                }
        
        for result in related_entries_results:
            # Assign a score based on distance, inverse relationship (closer is higher score)
            related_score = 1 / result['distance'] if result['distance'] > 0 else 0
            if result['id'] in all_results:
                all_results[result['id']]['related_score'] = related_score
            else:
                all_results[result['id']] = {
                    **result,
                    'semantic_score': 0,
                    'keyword_score': 0,
                    'related_score': related_score,
                }
        
        for result in all_results.values():
            result['combined_score'] = (
                result['semantic_score'] * 0.5 +
                result['keyword_score'] * 0.4 +
                result['related_score'] * 0.1)
        
        sorted_results = sorted(all_results.values(),
                                key=lambda x: x['combined_score'],
                                reverse=True)
        
        return sorted_results[:top_k]
    
    def build_context(self, entries: List[Dict], max_length: int = 4000) -> str:
        context_parts = []
        current_length = 0
        
        for i, entry in enumerate(entries, 1):
            part = (
                f"[Source {i}] Knowledge Base: {entry['knowledge_base_name']}\n"
                f"Entry: {entry['title']} (Category: {entry['category']})\n"
                f"Content: {entry['content']}\n\n"
            )
            
            if current_length + len(part) > max_length:
                break
                
            context_parts.append(part)
            current_length += len(part)
        
        return "".join(context_parts)
    
    def generate_answer_ollama(self, query: str, context: str) -> str:
        system_prompt = """You are an AI assistant providing information about animals.
            Your role is to provide accurate, helpful information based on the provided context.

            Guidelines:
            1. Answer questions based ONLY on the provided context.
            2. Cite specific entries and categories when providing information.
            3. If the context doesn't contain enough information, clearly state that.
            4. Use clear, simple language.
            5. Be precise and avoid speculation.

            Format your responses clearly with:
            - Direct answer to the question
            - Relevant entry/category references
            - Plain language explanation
            - Any important caveats or limitations"""
        
        user_prompt = f"""Question: {query}

            Knowledge Base Context:
            {context}

            Please provide a comprehensive answer based on the above knowledge base context."""
        
        payload = {
            "model": self.model_name,
            "prompt": f"{system_prompt}\n\n{user_prompt}",
            "stream": False,
            "options": {
                "temperature": 0.3,
                "num_predict": 1000,
                "top_k": 40,
                "top_p": 0.9
            }
        }
        
        try:
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json=payload,
                timeout=120
            )
            response.raise_for_status()
            
            result = response.json()
            return result.get('response', 'No response generated')
            
        except requests.exceptions.RequestException as e:
            return f"Error communicating with Ollama: {str(e)}"
    
    def query(self, user_question: str) -> Dict[str, Any]:
        print(f"Processing query: {user_question}")
        
        print("Searching knowledge graph...")
        relevant_entries = self.hybrid_search(user_question, top_k=5)
        
        if not relevant_entries:
            return {
                'answer': "I couldn't find relevant information in the knowledge base for your question. Please rephrase or ask about specific animal topics.",
                'sources': []
            }
        
        context = self.build_context(relevant_entries)
        
        print("Generating answer with Llama3.1...")
        answer = self.generate_answer_ollama(user_question, context)
        
        sources = [
            {
                'knowledge_base': entry['knowledge_base_name'],
                'entry_title': entry['title'],
                'category': entry['category'],
                'relevance_score': entry['combined_score']
            }
            for entry in relevant_entries
        ]
        
        return {
            'answer': answer,
            'sources': sources,
            'context': context
        }