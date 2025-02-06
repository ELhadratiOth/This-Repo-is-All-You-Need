from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from main import handle_query
from pydantic import BaseModel

app = FastAPI(
    title="ENSAH RAG API",
    version="1.0",
    description="API for querying ENSAH information using handle_query"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    query: str
    session_id: str = "default"

@app.post("/query")
async def query_endpoint(request: QueryRequest):
    """Endpoint to handle queries using handle_query function"""
    response = handle_query(request.query, request.session_id)
    return response

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
