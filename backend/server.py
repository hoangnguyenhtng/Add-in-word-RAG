from multiprocessing import context
import os
import hashlib
from typing import List, Dict, Optional
from datetime import datetime
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

# ========================
# Pinecone (gRPC classic) + cấu hình
# ========================
from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec
import google.generativeai as genai
import gc, hashlib, time
# ========================
# Config từ ENV
# ========================
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
KB_MIN_SCORE = 0.70  

if not GOOGLE_API_KEY:
    raise RuntimeError("Missing GOOGLE_API_KEY")
if not PINECONE_API_KEY:
    raise RuntimeError("Missing PINECONE_API_KEY")

genai.configure(api_key=GOOGLE_API_KEY)

GEN_MODEL = "models/gemini-2.0-flash"

# ========================
# Pinecone
# ========================
pc = Pinecone(api_key=PINECONE_API_KEY)

INDEX_NAME = "word-rag-integrated"
REGION = "us-east-1"

# Tạo index kiểu integrated model
if not pc.has_index(INDEX_NAME):
    pc.create_index_for_model(
        name=INDEX_NAME,
        cloud="aws",
        region=REGION,
        embed={
            "model": "llama-text-embed-v2",
            # "text" là key trong inputs khi upsert
            # KHÔNG cần field_map vì model tự lấy từ inputs["text"]
        }
    )

index = pc.Index(INDEX_NAME)

# Config
CHUNK_SIZE = 1500
CHUNK_OVERLAP = 120
UPLOAD_BATCH = 20
TOP_K_LOCAL = 5
MIN_COSINE = 0.50
MAX_CONTEXT_CHARS = 8_000
MAX_LOCAL_CHARS = 20_000

# ========================
# FastAPI
# ========================
app = FastAPI(title="Word RAG Add-in API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000", "https://localhost:3000",
        "http://localhost:8000", "https://localhost:8000"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

WEB_DIR = Path(__file__).resolve().parents[1] / "web"
app.mount("/assets", StaticFiles(directory=str(WEB_DIR / "assets")), name="assets")

# ========================
# Models
# ========================
class AskPayload(BaseModel):
    question: str
    context: str = ""
    options: Dict = {}

class IndexDocumentPayload(BaseModel):
    content: str
    metadata: Dict = {}

class SearchPayload(BaseModel):
    query: str
    top_k: int = 5
    filter: Optional[Dict] = None

# ========================
# Static Endpoints
# ========================
def _file_or_404(path: Path, name_for_404: str):
    if not path.exists():
        raise HTTPException(404, detail=f"{name_for_404} not found at {path}")
    return FileResponse(path)

@app.get("/taskpane.html")
def taskpane_html():
    return _file_or_404(WEB_DIR / "taskpane.html", "taskpane.html")

@app.get("/taskpane.css")
def taskpane_css():
    return _file_or_404(WEB_DIR / "taskpane.css", "taskpane.css")

@app.get("/taskpane.js")
def taskpane_js():
    return _file_or_404(WEB_DIR / "taskpane.js", "taskpane.js")

@app.get("/ping")
def ping():
    try:
        stats = index.describe_index_stats()
        return {"ok": True, "index_name": INDEX_NAME, "stats": stats}
    except Exception as e:
        raise HTTPException(500, detail=f"Pinecone error: {e}")

# ========================
# Helpers
# ========================
def chunk_text(text: str, chunk_size=None, overlap=None):
    """Generator để chunk text, tiết kiệm RAM"""
    chunk_size = chunk_size or CHUNK_SIZE
    overlap = overlap or CHUNK_OVERLAP
    n = len(text)
    i = 0
    while i < n:
        end = min(i + chunk_size, n)
        t = text[i:end].strip()
        if t:
            yield {"text": t, "start": i, "end": end}
        i = end - overlap
        if i <= 0:
            i = end

def generate_doc_id(content: str, metadata: Dict) -> str:
    unique_str = content[:1000] + str(metadata)  # Chỉ hash 1000 ký tự đầu
    return hashlib.md5(unique_str.encode()).hexdigest()[:16]

# ========================
# RAG Endpoints
# ========================
import gc

@app.post("/index_document")
def index_document(payload: IndexDocumentPayload):
    """Index tài liệu vào knowledge base (Pinecone)"""
    try:
        content = (payload.content or "").strip()
        metadata = payload.metadata or {}
        
        if not content:
            raise HTTPException(400, detail="Content không được để trống")

        doc_id = generate_doc_id(content, metadata)
        metadata["doc_id"] = doc_id
        metadata["indexed_at"] = datetime.utcnow().isoformat()
        metadata["kind"] = "kb_document"

        batch = []
        cnt = 0
        
        for i, c in enumerate(chunk_text(content)):
            # ✅ ĐÚNG: Dùng inputs cho integrated model
            batch.append({
                "id": f"{doc_id}_chunk_{i}",
                "inputs": {
                    "text": c["text"]  # Text để model embed
                },
                "metadata": {
                    **metadata,
                    "chunk_id": i,
                    "chunk_text": c["text"],  # Lưu text gốc để hiển thị
                    "start": c["start"],
                    "end": c["end"],
                }
            })
            
            if len(batch) >= UPLOAD_BATCH:
                index.upsert(vectors=batch, namespace="documents")
                cnt += len(batch)
                batch.clear()
                gc.collect()
        
        if batch:
            index.upsert(vectors=batch, namespace="documents")
            cnt += len(batch)
            batch.clear()
            gc.collect()

        return {
            "success": True, 
            "doc_id": doc_id, 
            "chunks_indexed": cnt,
            "message": f"Đã index {cnt} chunks"
        }
        
    except Exception as e:
        raise HTTPException(500, detail=f"Lỗi khi index document: {str(e)}")


@app.post("/search_knowledge_base")
def search_knowledge_base(payload: SearchPayload):
    """Tìm kiếm trong knowledge base"""
    try:
        query = (payload.query or "").strip()
        if not query:
            raise HTTPException(400, detail="Query không được để trống")

        # ✅ ĐÚNG: Dùng inputs cho integrated model
        res = index.query(
            inputs={"text": query},  # Integrated model cần inputs
            top_k=payload.top_k,
            include_metadata=True,
            namespace="documents",
            filter=payload.filter
        )
        
        matches = []
        for m in res.matches:
            md = m.metadata or {}
            matches.append({
                "id": m.id,
                "score": float(m.score),
                "text": md.get("chunk_text", ""),
                "metadata": {k: v for k, v in md.items() if k != "chunk_text"}
            })
            
        return {
            "success": True, 
            "matches": matches, 
            "count": len(matches)
        }
        
    except Exception as e:
        raise HTTPException(500, detail=f"Lỗi khi search: {str(e)}")


@app.post("/ask")
def ask(payload: AskPayload):
    """Trả lời câu hỏi dựa trên context và knowledge base"""
    try:
        question = (payload.question or "").strip()
        context = payload.context or ""
        opts = payload.options or {}
        use_kb = bool(opts.get("useKnowledgeBase", True))
        with_citations = bool(opts.get("withCitations", True))

        if not question:
            return {"answer": "Vui lòng nhập câu hỏi."}

        # 1) Local context từ tài liệu hiện tại
        if len(context) > MAX_LOCAL_CHARS:
            context = context[:MAX_LOCAL_CHARS]
            
        local_context = ""
        local_cites = []
        if context.strip():
            local_context = "\n\n### Từ tài liệu hiện tại:\n" + context.strip()[:6000]

        # 2) Knowledge base context
        kb_context = ""
        kb_cites = []
        if use_kb:
            try:
                # ✅ ĐÚNG: Query với inputs
                kb_resp = index.query(
                    inputs={"text": question},
                    top_k=3,
                    include_metadata=True,
                    namespace="documents",
                    filter={"kind": "kb_document"}  # Chỉ lấy KB docs
                )
                
                kb_chunks = []
                for m in (kb_resp.matches or []):
                    if m.score is None or float(m.score) < KB_MIN_SCORE:
                        continue
                    md = m.metadata or {}
                    txt = md.get("chunk_text", "")
                    if not txt:
                        continue
                    kb_chunks.append(txt)
                    if with_citations:
                        kb_cites.append({
                            "source": "knowledge_base",
                            "doc_id": md.get("doc_id", "unknown"),
                            "score": float(m.score),
                            "snippet": txt[:150]
                        })
                        
                if kb_chunks:
                    kb_context = "\n\n### Từ kho dữ liệu:\n" + "\n\n".join(kb_chunks[:3])
                    
            except Exception as e:
                print(f"KB query error: {e}")  # Log lỗi nhưng không crash

        # 3) Gộp context
        combined = (local_context + kb_context).strip()
        if len(combined) > MAX_CONTEXT_CHARS:
            combined = combined[:MAX_CONTEXT_CHARS]

        # 4) Gọi Gemini
        model = genai.GenerativeModel(GEN_MODEL)
        
        if not combined:
            resp = model.generate_content(f"Bạn là trợ lý soạn thảo. Câu hỏi: {question}")
            return {"answer": resp.text, "citations": []}

        sys_msg = (
            "Bạn là trợ lý RAG trong Word. Trả lời dựa trên đoạn trích được cung cấp. "
            "Nếu có thông tin từ tài liệu cũ, hãy so sánh và phân tích sự khác biệt."
        )
        prompt = (
            f"{sys_msg}\n\n### Câu hỏi\n{question}\n\n"
            f"### Đoạn trích liên quan\n{combined}\n\n"
            f"### Yêu cầu\n"
            f"- Trả lời ngắn gọn, rõ ràng\n"
            f"- Nếu có thông tin từ cả tài liệu mới và cũ, hãy so sánh\n"
            f"- Dùng gạch đầu dòng khi phù hợp"
        )
        
        resp = model.generate_content(prompt)
        ans = resp.text or "Không nhận được phản hồi từ mô hình."
        
        return {
            "answer": ans, 
            "citations": (local_cites + kb_cites) if with_citations else []
        }

    except Exception as e:
        return {"answer": f"❌ Lỗi server: {type(e).__name__}: {e}"}


@app.get("/list_documents")
def list_documents():
    """Liệt kê thống kê documents"""
    try:
        stats = index.describe_index_stats()
        return {
            "success": True, 
            "total_vectors": stats.total_vector_count, 
            "namespaces": stats.namespaces
        }
    except Exception as e:
        raise HTTPException(500, detail=f"Lỗi: {e}")


@app.delete("/delete_document/{doc_id}")
def delete_document(doc_id: str):
    """Xóa document khỏi KB"""
    try:
        index.delete(filter={"doc_id": doc_id}, namespace="documents")
        return {"success": True, "message": f"Đã xóa document {doc_id}"}
    except Exception as e:
        raise HTTPException(500, detail=f"Lỗi khi xóa: {e}")


@app.delete("/cleanup_current_doc/{doc_id}")
def cleanup_current_doc(doc_id: str):
    """Xóa temporary document"""
    try:
        index.delete(filter={"doc_id": doc_id}, namespace="documents")
        return {"success": True}
    except Exception as e:
        raise HTTPException(500, detail=f"Cleanup error: {e}")