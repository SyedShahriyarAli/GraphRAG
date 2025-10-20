from flask import json
from graph.graph_builder import GraphBuilder
import os
from dotenv import load_dotenv

load_dotenv()

builder = GraphBuilder(
    uri=os.getenv('NEO4J_URI'),
    username=os.getenv('NEO4J_USER'),
    password=os.getenv('NEO4J_PASSWORD')
)

json_files = []

try:
    builder.create_constraints_and_indexes()
    
    print("\nIngesting knowledge base...")
    with open("./file_paths.json", 'r', encoding='utf-8') as file:
        json_files = json.load(file)
    
    for json_file in json_files:
        print(f"\n Processing: {json_file}")
        builder.ingest_knowledge_base(json_file)
    
    # print("\nCreating similarity relationships...")
    # builder.create_similarity_relationships(threshold=0.7)
    
    print("\n✅ Data ingestion complete!")
    
finally:
    builder.close()
