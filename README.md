# Real Estate AI Assistant

An AI-powered real estate assistant built with Python, FastAPI, Streamlit, SQLite, Ollama, and Retrieval-Augmented Generation (RAG).

The application helps users discover properties, ask questions about real estate projects, retrieve relevant information from documents, and receive property recommendations through a conversational interface.

## 1. Project Overview

The Real Estate AI Assistant combines structured property search with document-based question answering.

Users can:
- Search properties using location, budget, property type, and bedroom preferences.
- Ask questions about real estate project documents.
- Retrieve relevant information using hybrid search.
- Improve retrieval relevance through document reranking.
- Receive property recommendations.
- Maintain preferences across conversational interactions.
- Analyze property information using an AI-powered conversational interface.

## 2. Technology Stack

| Component | Technology |
|---|---|
| Programming language | Python 3.12 |
| AI model | Qwen 2.5 VL |
| Embedding model | Nomic Embed Text |
| LLM server | Ollama |
| Backend API | FastAPI |
| Frontend | Streamlit |
| Database | SQLite |
| Data processing | Pandas |
| Retrieval | Vector Search + Keyword Search |
| Reranking | Qwen |
| Containerization | Docker |
| Orchestration | Docker Compose |
| Testing | Pytest |

## 3. Architecture

The application follows a modular architecture.

### Frontend
Streamlit provides the user interface for conversational interactions and property searches.

### Backend
FastAPI handles requests, authentication, request validation, and communication with the AI application.

### AI Agent
The Python-based agent coordinates the interaction between the language model and available application tools.

### Retrieval-Augmented Generation
The RAG pipeline retrieves relevant information from uploaded property documents and provides context to the language model.

### Database
SQLite stores property records and document indexing information.

### Ollama
Ollama hosts the language and embedding models on a remote server.

## 4. Key Features

### Property Search
- Search properties by location.
- Filter by bedrooms.
- Filter by minimum and maximum budget.
- Retrieve property details from SQLite.

### Retrieval-Augmented Generation (RAG)
- Load and process property documents.
- Split documents into searchable chunks.
- Generate embeddings using Ollama.
- Retrieve relevant document passages.
- Provide retrieved context to the language model.

### Hybrid Search
Combines keyword-based retrieval and vector similarity retrieval to improve document discovery.

### Document Reranking
Uses the language model to rerank retrieved document chunks based on their relevance to the user's question.

### AI Agent
Uses a Python-controlled action-selection workflow to determine when to search properties, retrieve documents, or answer directly.

### Conversational Memory
Maintains relevant user preferences, including:
- Locations
- Property type
- Bedrooms
- Minimum budget
- Maximum budget

### Property Recommendations
Uses property attributes and recommendation scoring to identify suitable listings.

### Answer Validation
Checks whether document-based answers are supported by retrieved information and avoids presenting unsupported claims as confirmed facts.

### API Security
- API key authentication.
- Request validation.
- Rate limiting.
- Configurable environment variables.

## 5. Project Structure

```text
real_estate_ai/
│
├── app.py
├── api.py
├── rag.py
├── vision.py
├── recommendations.py
├── cache.py
│
├── streamlit_app.py
├── dashboard.py
├── dashboard_app.py
│
├── evaluation_dataset.py
├── evaluate_retrieval.py
├── evaluate_answers.py
│
├── documents/
│   ├── project_a.txt
│   └── project_b.txt
│
├── images/
│
├── test_ocr.py
├── test_recommendations.py
├── test_reranking.py
│
├── requirements.txt
├── Dockerfile
├── Dockerfile.backend
├── docker-compose.yml
├── .dockerignore
├── .gitignore
├── .env
└── README.md
```

## 6. Prerequisites

Install the following:

- Python 3.12
- Docker Desktop
- Git
- VS Code
- Ollama running on a reachable server

Required Ollama models:

```bash
ollama pull qwen2.5vl:latest
ollama pull nomic-embed-text
```

The Ollama server must be accessible from the application containers.

## 7. Environment Configuration

Create a `.env` file in the project root.

Example:

```env
OLLAMA_BASE_URL=http://YOUR_OLLAMA_SERVER:11434
OLLAMA_MODEL=qwen2.5vl:latest
OLLAMA_EMBED_MODEL=nomic-embed-text
API_SECRET_KEY=YOUR_SECRET_KEY
```

Replace the placeholder values with your actual configuration.

**Security:** Never commit `.env` or expose your API secret in GitHub.

## 8. Installation and Setup

Clone the repository:

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

Navigate to the project:

```bash
cd real_estate_ai
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## 9. Run with Docker

Build and start the application:

```bash
docker compose up -d --build
```

Check running containers:

```bash
docker compose ps
```

Access the application:

| Service | URL |
|---|---|
| Streamlit frontend | http://localhost:8501 |
| FastAPI backend | http://localhost:8000 |
| API documentation | http://localhost:8000/docs |
| Health endpoint | http://localhost:8000/health |

Stop the application:

```bash
docker compose down
```

The database is stored in a Docker named volume to preserve data across container restarts.

Do not use `docker compose down -v` unless you intentionally want to delete the database volume.

## 10. API Authentication

The chat API requires an API key.

Example request:

```python
import requests

url = "http://localhost:8000/chat"

headers = {
    "X-API-Key": "YOUR_SECRET_KEY"
}

payload = {
    "session_id": "session-001",
    "question": "Find 2-bedroom properties in Powai"
}

response = requests.post(
    url,
    headers=headers,
    json=payload,
    timeout=300
)

print(response.status_code)
print(response.json())
```

Keep API keys confidential and use environment variables rather than hardcoding credentials.

## 11. Testing

Run the test suite inside the backend container:

```bash
docker compose exec backend python -m pytest -q
```

Run an individual test:

```bash
docker compose exec backend python -m pytest test_recommendations.py -q
```

The current verified test suite passed:

```text
2 passed
```

Additional evaluation scripts are included for retrieval and answer evaluation.

## 12. Database

SQLite is used for property records and document indexing.

The database is stored at:

```text
/app/data/real_estate_ai.db
```

The application uses a shared database path between its modules.

## 13. Security Considerations

- API key authentication is enabled.
- API request validation is implemented.
- Rate limiting is configured.
- Environment variables are used for secrets.
- `.env` is excluded through `.gitignore`.
- Database persistence is managed through Docker volumes.

For public deployment, additionally configure HTTPS, production-grade secret management, restricted CORS origins, monitoring, and appropriate access controls.

## 14. Current Limitations

- Property records are sample data and should not be treated as live real estate listings.
- Answer quality depends on the available property documents and retrieved context.
- Ollama must be reachable for AI features.
- The current test suite is small and should be expanded before commercial deployment.
- Production deployment requires additional operational and security hardening.

## 15. Future Improvements

Potential enhancements include:

- Integration with live property listing APIs.
- More comprehensive automated tests.
- User accounts and persistent chat sessions.
- Advanced image-based property analysis.
- Improved retrieval evaluation.
- Production monitoring and logging.
- Cloud deployment and scalability improvements.

## 16. Project Objective

This project demonstrates practical AI engineering concepts, including:

- Large Language Model integration.
- Retrieval-Augmented Generation.
- Embedding generation.
- Vector and keyword retrieval.
- Hybrid search.
- Document reranking.
- AI agent workflows.
- Conversational memory.
- Structured database querying.
- REST API development.
- Containerization.
- API authentication.
- Automated testing.

---

**Built as an AI Engineering portfolio project using Python, Ollama, FastAPI, Streamlit, SQLite, and Docker.**