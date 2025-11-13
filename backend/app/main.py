"""
FastAPI application entry point.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.advanced_routes import router as advanced_router
from app.api.reranking_routes import router as reranking_router
from app.api.routes import router
from app.core.config import get_settings

settings = get_settings()

app = FastAPI(
    title="Enterprise RAG Platform API",
    description="Advanced RAG system with Graph RAG, Hybrid Search, and specialized chunking strategies",
    version="2.0.0",
    debug=settings.debug,
)

# CORS middleware for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:3001",
        "http://localhost:8080",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(router, prefix="/api", tags=["Basic RAG"])
app.include_router(advanced_router, prefix="/api", tags=["Advanced RAG"])
app.include_router(reranking_router, prefix="/api", tags=["Reranking"])


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "RAG Learning Platform API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/api/health",
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
