from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List

from models import AskRequest, AskResponse, Source
from rag_service import RAGService

app = FastAPI(
    title="Appliance Troubleshooter RAG API",
    description="Upload manuals and ask questions about your appliances.",
    version="1.0.0",
)

# Enable CORS for frontend testing
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # for dev; restrict later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

rag_service = RAGService()


@app.post("/upload-manual")
async def upload_manual(device_id: str, file: UploadFile = File(...)):
    """
    Upload a device manual (PDF). It will be indexed for later Q&A.
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    file_bytes = await file.read()

    try:
        chunks_count = rag_service.index_manual(device_id, file_bytes, file.filename)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Indexing error: {str(e)}")

    return {"message": "Manual indexed successfully", "device_id": device_id, "chunks": chunks_count}


@app.post("/ask", response_model=AskResponse)
async def ask_question(payload: AskRequest):
    """
    Ask a question about a specific device's manual.
    """
    try:
        answer, srcs = rag_service.ask_question(payload.device_id, payload.question)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during question answering: {str(e)}")

    sources: List[Source] = [
        Source(page=s.get("page"), snippet=s.get("snippet", "")) for s in srcs
    ]

    return AskResponse(answer=answer, sources=sources)
