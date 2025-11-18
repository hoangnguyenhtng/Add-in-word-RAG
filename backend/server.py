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

import chromadb
from chromadb.config import Settings
from chromadb.utils.embedding_functions import EmbeddingFunction

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

STAGING_PATH = str((Path(__file__).resolve().parent / "data" / "chroma_staging").mkdir(parents=True, exist_ok=True) or (Path(__file__).resolve().parent / "data" / "chroma_staging"))
# Chroma 0.5+ PersistentClient
_chroma_client = chromadb.PersistentClient(path=STAGING_PATH, settings=Settings(anonymized_telemetry=False))

class ZeroEmbedding(EmbeddingFunction):
    def __call__(self, inputs):
        if isinstance(inputs, str):  # Chroma có thể gọi đơn lẻ
            inputs = [inputs]
        return [[0.0] * DIMENSION for _ in inputs]  # chỉ để “giữ chỗ”

def get_staging_collection():
    # Lưu chunks tạm; không dùng Chroma để search, nên dùng zero-embedding để khỏi tính vector lúc stage
    return _chroma_client.get_or_create_collection(
        name="kb_staging",
        embedding_function=ZeroEmbedding()
    ) 

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
# STAGE DOCUMENT (to Chroma)
# ========================
class StageDocumentPayload(BaseModel):
    # "url": server tự tải; "office": client gửi base64
    source: str
    url: Optional[str] = None
    filename: Optional[str] = None
    content_base64: Optional[str] = None
    metadata: Dict = {}
    options: Dict = {}

MAX_DOWNLOAD_MB = 50
MAX_PAGES_PDF = 2000

def _ext_from_filename(name: Optional[str]) -> str:
    if not name: return ""
    name = name.lower().strip()
    for ext in (".pdf", ".docx", ".txt"):
        if name.endswith(ext): return ext
    return ""

def _download_to_temp(url: str, temp_dir: Path, filename_hint: Optional[str]) -> Path:
    import requests
    r = requests.get(url, stream=True, timeout=30)
    r.raise_for_status()
    total = 0
    ext = _ext_from_filename(filename_hint) or ".bin"
    dst = temp_dir / f"download{ext}"
    with open(dst, "wb") as f:
        for chunk in r.iter_content(1024 * 1024):
            if chunk:
                f.write(chunk)
                total += len(chunk)
                if total > MAX_DOWNLOAD_MB * 1024 * 1024:
                    raise HTTPException(413, detail=f"File > {MAX_DOWNLOAD_MB}MB")
    return dst

def _write_base64_to_temp(b64: str, temp_dir: Path, filename_hint: Optional[str]) -> Path:
    import base64
    data = base64.b64decode(b64)
    if len(data) > MAX_DOWNLOAD_MB * 1024 * 1024:
        raise HTTPException(413, detail=f"File > {MAX_DOWNLOAD_MB}MB")
    ext = _ext_from_filename(filename_hint) or ".bin"
    dst = temp_dir / f"upload{ext}"
    with open(dst, "wb") as f:
        f.write(data)
    return dst

# PDF/DOCX/TXT extract
try:
    from pypdf import PdfReader
except Exception:
    PdfReader = None
try:
    from docx import Document as DocxDocument
except Exception:
    DocxDocument = None

def _extract_text_any(path: Path) -> str:
    name = path.name.lower()
    if name.endswith(".txt"):
        return path.read_text(encoding="utf-8", errors="ignore")
    if name.endswith(".docx"):
        if not DocxDocument: raise HTTPException(500, detail="Thiếu python-docx")
        d = DocxDocument(str(path))
        return "\n".join(p.text for p in d.paragraphs)
    if name.endswith(".pdf"):
        if not PdfReader: raise HTTPException(500, detail="Thiếu pypdf")
        reader = PdfReader(str(path))
        pages = min(len(reader.pages), MAX_PAGES_PDF)
        return "\n".join((reader.pages[i].extract_text() or "") for i in range(pages))
    raise HTTPException(415, detail="Chỉ hỗ trợ .pdf/.docx/.txt")

def _chunk_text_to_list(text: str, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    out = []
    n = len(text); i = 0
    while i < n:
        end = min(i + chunk_size, n)
        t = text[i:end].strip()
        if t:
            out.append({"text": t, "start": i, "end": end})
        i = end - overlap
        if i <= 0: i = end
    return out

# ========================
# FLUSH STAGED -> Pinecone
# ========================
class FlushPayload(BaseModel):
    doc_id: Optional[str] = None
    limit: int = 500       # tổng số chunks tối đa upload trong 1 lần (an toàn)
    embed_batch: int = 16  # batch embed
    upsert_batch: int = 32 # batch upsert

@app.post("/flush_staged")
def flush_staged(payload: FlushPayload):
    # Pinecone phải sẵn sàng
    if index is None:
        raise HTTPException(503, detail="Pinecone chưa sẵn sàng")

    col = get_staging_collection()

    # Lấy ids theo từng trang nhỏ để tránh load tất cả vào RAM
    # Chroma 0.5 có get(limit, offset, where)
    remaining = payload.limit
    offset = 0
    done = 0
    deleted_ids_total = []

    where = {"doc_id": payload.doc_id} if payload.doc_id else None

    while remaining > 0:
        page_size = min(200, remaining)
        batch = col.get(where=where, limit=page_size, offset=offset, include=["documents","metadatas","embeddings"])
        ids = batch.get("ids", [])
        docs = batch.get("documents", [])
        metas = batch.get("metadatas", [])

        if not ids:
            break  # hết dữ liệu

        # Tính embedding theo lô nhỏ
        embed_batch = max(1, payload.embed_batch)
        upsert_batch = max(1, payload.upsert_batch)

        to_upsert = []
        for i in range(0, len(docs), embed_batch):
            sub_docs = docs[i:i+embed_batch]
            embs = embed_texts(sub_docs)  # Gemini text-embedding-004 (768-D)

            # gom record
            for j, emb in enumerate(embs):
                k = i + j
                rec_id = ids[k]
                md = metas[k] or {}
                # đảm bảo doc_id tồn tại
                md.setdefault("doc_id", md.get("doc_id", "unknown"))
                # chunk_text giữ lại 1 phần đầu để search tóm tắt (tuỳ)
                to_upsert.append({
                    "id": rec_id,
                    "values": emb,
                    "metadata": {
                        **md,
                        "chunk_text": docs[k][:CHUNK_SIZE],  # giữ đoạn text
                    }
                })

            # đẩy lên Pinecone theo lô upsert_batch
            while len(to_upsert) >= upsert_batch:
                payload_up = to_upsert[:upsert_batch]
                index.upsert(vectors=payload_up, namespace=NAMESPACE)
                del to_upsert[:upsert_batch]
                time.sleep(0.05)

        # phần dư
        if to_upsert:
            index.upsert(vectors=to_upsert, namespace=NAMESPACE)
            to_upsert.clear()

        # Xoá các id đã upsert khỏi staging
        col.delete(ids=ids)
        deleted_ids_total.extend(ids)

        uploaded = len(ids)
        done += uploaded
        remaining -= uploaded
        # Không tăng offset khi đã xoá các id vừa lấy (tránh nhảy cóc)
        # Nếu bạn không xoá ở đây, cần offset += len(ids)

        # an toàn: tạm dừng ngắn tránh nóng máy
        time.sleep(0.05)

    return {
        "success": True,
        "uploaded_chunks": done,
        "deleted_from_staging": len(deleted_ids_total),
        "message": f"✅ Đã flush {done} chunks từ Chroma lên Pinecone và xóa staging tương ứng."
    }

@app.get("/staging_stats")
def staging_stats(doc_id: Optional[str] = None):
    col = get_staging_collection()
    # Chroma chưa expose count by where trực tiếp; ta sẽ lấy id theo lô nhỏ để đếm
    where = {"doc_id": doc_id} if doc_id else None
    offset = 0; total = 0
    while True:
        batch = col.get(where=where, limit=500, offset=offset, include=[])
        ids = batch.get("ids", [])
        if not ids: break
        total += len(ids)
        offset += len(ids)
    return {"success": True, "doc_id": doc_id, "staging_count": total}

@app.post("/stage_document")
def stage_document(payload: StageDocumentPayload):
    # Không cần Pinecone ở bước này
    temp_dir = Path(tempfile.mkdtemp(prefix="stage-doc-"))
    tmp = None
    try:
        # 1) lấy file
        if payload.source == "url":
            if not payload.url: raise HTTPException(400, detail="Thiếu url")
            tmp = _download_to_temp(payload.url, temp_dir, payload.filename)
        elif payload.source == "office":
            if not payload.content_base64: raise HTTPException(400, detail="Thiếu content_base64")
            tmp = _write_base64_to_temp(payload.content_base64, temp_dir, payload.filename)
        else:
            raise HTTPException(400, detail="source phải là 'url' hoặc 'office'")

        # 2) extract + chunk
        text = _extract_text_any(tmp)
        if not text.strip(): raise HTTPException(400, detail="Không trích xuất được nội dung")
        chunks = _chunk_text_to_list(text)
        chunk_count = len(chunks)

        # 3) metadata & doc_id
        meta = payload.metadata or {}
        base_meta = {
            **meta,
            "title": meta.get("title") or meta.get("name") or (payload.filename or tmp.name),
            "doc_type": meta.get("doc_type") or "other",
            "date": meta.get("date") or datetime.utcnow().date().isoformat(),
            "kind": "kb_document",
            "indexed_at": datetime.utcnow().isoformat(),
        }
        doc_id = generate_doc_id(text[:1000], base_meta)
        base_meta["doc_id"] = doc_id

        # 4) Lưu vào Chroma (zero-embedding) theo batch nhỏ
        col = get_staging_collection()
        BATCH = 100
        ids, docs, metas = [], [], []
        added = 0
        for i, ch in enumerate(chunks):
            ids.append(f"{doc_id}_chunk_{i}")
            docs.append(ch["text"])
            metas.append({**base_meta, "chunk_id": i, "start": ch["start"], "end": ch["end"]})
            if len(ids) >= BATCH:
                col.add(ids=ids, documents=docs, metadatas=metas)  # zero-embedding tự sinh
                added += len(ids)
                ids, docs, metas = [], [], []
        if ids:
            col.add(ids=ids, documents=docs, metadatas=metas)
            added += len(ids)

        return {
            "success": True,
            "doc_id": doc_id,
            "title": base_meta["title"],
            "staged_chunks": added,
            "message": f"✅ Đã STAGE {added}/{chunk_count} chunks vào Chroma (local)."
        }
    finally:
        try: shutil.rmtree(temp_dir, ignore_errors=True)
        except Exception: pass


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