# GraphRAG: Graph-based Retrieval Augmented Generation

This project demonstrates a Graph-based Retrieval Augmented Generation (RAG) system. It leverages Neo4j as a graph database to store and retrieve information, enhancing the capabilities of a language model for more accurate and context-rich responses.

## Project Setup

Follow these steps to set up and run the GraphRAG project.

### 1. Set up Neo4j using Docker Compose

The project uses Neo4j as its graph database. You can easily set it up using Docker Compose.

First, ensure you have Docker and Docker Compose installed on your system.

Navigate to the root directory of the project and run the following command to start the Neo4j container:

```bash
docker-compose up -d
```

This command will start a Neo4j instance in the background. You can access the Neo4j Browser at `http://localhost:7474` and the Bolt port at `7687`.

### 2. Environment Variables

The project uses environment variables for sensitive information and configuration. These are stored in a `.env` file at the root of the project. Here's an example of the `.env` file:

```
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=password

OLLAMA_URL=<ollama-base-url>
OLLAMA_MODEL=llama3.1:8b

FLASK_ENV=development
FLASK_DEBUG=True
```

Make sure to create a `.env` file in the project root with your specific configurations.

### 3. Python Environment Setup

This project uses `uv` for dependency management.

First, create a virtual environment:

```bash
uv venv .venv
```

Then, activate the virtual environment (if not automatically activated by your IDE) and install the required packages:

```bash
uv sync
```

### 4. Ingest Data

Once Neo4j is running, you need to ingest the initial dataset into the graph database.

This project uses the `animal_data.json` file as its dataset. This file contains information about various animals, which will be transformed into a graph structure within Neo4j.

Run the following Python script to ingest the data:

```bash
cd src
python ingest_data.py
```

This script will connect to your running Neo4j instance and populate it with the data from `animal_data.json`, creating nodes and relationships based on the defined schema, and establishing relationships between these entities to form a rich knowledge graph.

### 5. Run the Application

After data ingestion, you can start the main application.

```bash
python app.py
```

This will start a Flask web server that exposes the query endpoint.

### 6. Query Endpoint and Health Check

The application provides a query endpoint for interacting with the RAG system and a health check endpoint.

#### Health Check

You can check the health of the application by navigating to:

```
http://localhost:5000/health
```

#### Query Endpoint

To query the RAG system, you can send requests to the query endpoint. This endpoint will perform a hybrid search on the Neo4j graph database to retrieve relevant information.

Example query (using `curl` or a similar tool):

```bash
curl -X POST "http://localhost:5000/api/query" -H "Content-Type: application/json" -d '{"question": "Tell me about lions"}'
```

### 7. Hybrid Search Weightage

The query endpoint utilizes a hybrid search mechanism, combining different search strategies (e.g., keyword search, vector similarity search, related entries) to provide comprehensive results. Each search component contributes to the final relevance score with a specific weightage.

The current implementation defines the following weightages for the hybrid search:

- **Semantic Search Weight:** 0.5
- **Keyword Search Weight:** 0.4
- **Related Entries Search Weight:** 0.1

These weights can be adjusted in the application's source code (`app.py` or `graph_rag.py`) to fine-tune the balance between different search methodologies based on desired retrieval performance.

### 8. Dataset Used

The project utilizes the `animal_data.json` file as its primary dataset. This JSON file contains structured information about various animals, including their characteristics, habitats, and other relevant attributes. This data is transformed into a graph structure within Neo4j, where animals, their properties, and relationships between them are represented as nodes and edges.

### 9. Graph Database Relations

Here's a conceptual diagram illustrating the relationships within the Neo4j graph database:

![Graph Database Relations Placeholder](data/graph_rag.png.png)