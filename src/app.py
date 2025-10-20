from dotenv import load_dotenv
from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from datetime import datetime
import uuid

from graph.graph_rag import GraphRAG

load_dotenv()


app = Flask(__name__)
CORS(app)

rag_system = GraphRAG(
    neo4j_uri=os.getenv('NEO4J_URI'),
    neo4j_user=os.getenv('NEO4J_USER'),
    neo4j_password=os.getenv('NEO4J_PASSWORD'),
    ollama_url=os.getenv('OLLAMA_URL'),
    ollama_model=os.getenv('OLLAMA_MODEL')
)

conversation_history = {}

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'model': 'llama3.1:8b'
    })

@app.route('/api/query', methods=['POST'])
def query():
    """
    Main query endpoint for legal questions
    
    Request body:
    {
        "question": "What is bail?",
        "session_id": "optional-session-id"
    }
    """
    try:
        data = request.json
        question = data.get('question', '').strip()
        session_id = data.get('session_id', str(uuid.uuid4()))
        
        if not question:
            return jsonify({
                'error': 'Question is required'
            }), 400
        
        result = rag_system.query(question)
        
        if session_id not in conversation_history:
            conversation_history[session_id] = []
        
        conversation_history[session_id].append({
            'question': question,
            'answer': result['answer'],
            'sources': result['sources'],
            'timestamp': datetime.now().isoformat()
        })
        
        return jsonify({
            'success': True,
            'answer': result['answer'],
            'sources': result['sources'],
            'session_id': session_id
        })
        
    except Exception as e:
        print(f"Error processing query: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/history/<session_id>', methods=['GET'])
def get_history(session_id):
    """Get conversation history for a session"""
    history = conversation_history.get(session_id, [])
    return jsonify({
        'session_id': session_id,
        'history': history,
        'count': len(history)
    })

@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Get statistics about the knowledge graph"""
    try:
        with rag_system.driver.session() as session:
            kb_count = session.run("MATCH (kb:KnowledgeBase) RETURN count(kb) as count").single()['count']
            
            entry_count = session.run("MATCH (e:Entry) RETURN count(e) as count").single()['count']
            
            concept_count = session.run("MATCH (c:Concept) RETURN count(c) as count").single()['count']
            
            rel_count = session.run("MATCH ()-[r]->() RETURN count(r) as count").single()['count']
            
            return jsonify({
                'success': True,
                'stats': {
                    'knowledge_bases': kb_count,
                    'entries': entry_count,
                    'concepts': concept_count,
                    'relationships': rel_count,
                    'total_sessions': len(conversation_history)
                }
            })
            
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({
        'success': False,
        'error': 'Endpoint not found'
    }), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        'success': False,
        'error': 'Internal server error'
    }), 500

if __name__ == '__main__':
    print("Starting Legal AI Advisor API...")
    print(f"Using Ollama {os.getenv('OLLAMA_URL')} model")
    print(f"Neo4j connection: {os.getenv('NEO4J_URI')}")
    print(f"Ollama URL: {os.getenv('OLLAMA_MODEL')}")
    app.run(debug=True, host='0.0.0.0', port=5000)