# RAG Backend - Quick Start Guide

## 🚀 Setup (5 minutes)

### 1. Start Qdrant Vector Database
```bash
docker-compose up -d
```

Verify it's running: http://localhost:6333/dashboard

### 2. Install Backend Dependencies
```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Configure Environment
```bash
cp .env.example .env
```

Edit `.env` and add your OpenAI API key:
```
OPENAI_API_KEY=sk-your-key-here
```

### 4. Start Backend Server
```bash
uvicorn app.main:app --reload --port 8000
```

## 🧪 Test the API

### Interactive Docs
Open http://localhost:8000/docs for full Swagger UI

### Quick Test with cURL

**1. Upload a document:**
```bash
echo "The quick brown fox jumps over the lazy dog. 
This is a sample document about animals and nature. 
Foxes are known for their cunning and agility." > test.txt

curl -X POST "http://localhost:8000/api/upload" \
  -F "file=@test.txt" \
  | jq
```

Save the `document_id` from the response.

**2. Index the document:**
```bash
curl -X POST "http://localhost:8000/api/index" \
  -H "Content-Type: application/json" \
  -d '{
    "document_id": "YOUR_DOCUMENT_ID_HERE",
    "chunking_strategy": "recursive",
    "chunk_size": 1000,
    "chunk_overlap": 200,
    "embedding_provider": "openai"
  }' | jq
```

**3. Search for similar content:**
```bash
curl -X POST "http://localhost:8000/api/search" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Tell me about foxes",
    "top_k": 3
  }' | jq
```

**4. RAG Query (Full pipeline - retrieve + generate):**
```bash
curl -X POST "http://localhost:8000/api/rag" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What animals are mentioned in the document?",
    "top_k": 3,
    "model": "gpt-3.5-turbo"
  }' | jq
```

## 🔧 Experiment with Chunking Strategies

### List Available Strategies
```bash
curl http://localhost:8000/api/strategies | jq
```

### Try Different Strategies

**Fixed-size chunking:**
```json
{
  "document_id": "YOUR_DOC_ID",
  "chunking_strategy": "fixed",
  "chunk_size": 500,
  "chunk_overlap": 100
}
```

**Recursive (recommended):**
```json
{
  "document_id": "YOUR_DOC_ID",
  "chunking_strategy": "recursive",
  "chunk_size": 1000,
  "chunk_overlap": 200
}
```

## 📚 Learning Path

### 1. **Start with Basic RAG**
- Upload a document
- Index with default settings (recursive chunking)
- Try search queries
- Test RAG endpoint

### 2. **Experiment with Chunking**
- Upload the same document multiple times
- Index with different `chunk_size` values (500, 1000, 2000)
- Index with different `chunk_overlap` values (0, 100, 200)
- Compare search results

### 3. **Compare Strategies**
- Index with `recursive` strategy
- Index with `fixed` strategy
- Notice how results differ based on chunk boundaries

### 4. **Understand the Pipeline**
Look at the code in order:
1. `backend/app/services/chunking/` - How text is split
2. `backend/app/services/embedding/` - How chunks become vectors
3. `backend/app/services/vector/` - How Qdrant stores/searches vectors
4. `backend/app/services/rag_service.py` - How LLM generates answers

### 5. **Next Steps (Extend the System)**
- Implement semantic chunking (group by meaning, not just size)
- Add reranking (use Cohere or cross-encoders)
- Implement hybrid search (dense + sparse/BM25)
- Add metadata filtering
- Implement query transformations

## 🐛 Troubleshooting

**Qdrant not connecting?**
```bash
docker ps  # Check if Qdrant is running
docker-compose logs qdrant
```

**OpenAI errors?**
- Check your API key in `.env`
- Verify you have credits: https://platform.openai.com/usage

**Import errors?**
```bash
pip install -r requirements.txt --upgrade
```

## 📖 API Endpoints Reference

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/` | GET | Root info |
| `/api/health` | GET | Check system health |
| `/api/upload` | POST | Upload document |
| `/api/index` | POST | Chunk + embed + index |
| `/api/search` | POST | Semantic search only |
| `/api/rag` | POST | Search + LLM generation |
| `/api/strategies` | GET | List chunking strategies |
| `/docs` | GET | Interactive API docs |

## 💡 Tips

1. **Start small**: Test with a small text file first
2. **Check the logs**: The terminal shows helpful debug info
3. **Use /docs**: Swagger UI is perfect for exploring the API
4. **Experiment**: Try different chunk sizes and see what works best
5. **Read the code**: Each service is well-documented with docstrings

Happy learning! 🎉

