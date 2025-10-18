# test_pinecone_grpc_ready.py
import os, time
from typing import List
from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec
import google.generativeai as genai

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY") or "PASTE_PINECONE_KEY"
GOOGLE_API_KEY   = os.getenv("GOOGLE_API_KEY")   or "PASTE_GEMINI_KEY"

INDEX_NAME = "docs-example-classic"
REGION     = "us-east-1"
NAMESPACE  = "example-namespace"

EMBED_MODEL = "models/text-embedding-004"  # 768-D
DIMENSION   = 768

def wait_index_ready(pc, name, timeout=120):
    start = time.time()
    while True:
        info = pc.describe_index(name).to_dict()
        state = info.get("status", {}).get("state")
        if state == "Ready":
            print(f"✅ Index ready: {state}")
            return
        if time.time() - start > timeout:
            raise TimeoutError(f"Index not ready after {timeout}s. State={state}")
        print(f"⏳ Waiting index ready... state={state}")
        time.sleep(2)

def embed_texts(texts: List[str]) -> List[List[float]]:
    out = []
    for t in texts:
        r = genai.embed_content(model=EMBED_MODEL, content=t, task_type="retrieval_document")
        emb = getattr(r, "embedding", None) or r.get("embedding")
        if not emb:
            raise RuntimeError("Gemini embedding failed")
        out.append(emb)
    return out

def main():
    genai.configure(api_key=GOOGLE_API_KEY)
    pc = Pinecone(api_key=PINECONE_API_KEY)

    # 1) Tạo index nếu thiếu
    if not pc.has_index(INDEX_NAME):
        print(f"🆕 Creating index '{INDEX_NAME}' (dim={DIMENSION}, cosine)…")
        pc.create_index(
            name=INDEX_NAME,
            dimension=DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region=REGION),
            deletion_protection="disabled",
        )
    else:
        print(f"✅ Index '{INDEX_NAME}' already exists.")

    # 2) Chờ READY (rất quan trọng)
    wait_index_ready(pc, INDEX_NAME)

    index = pc.Index(INDEX_NAME)

    # 3) Dọn namespace test (để đếm vector dễ)
    try:
        index.delete(delete_all=True, namespace=NAMESPACE)
        time.sleep(1)
    except Exception:
        pass

    # 4) Embed + upsert
    doc_id = "grpc_demo_doc"
    passages = [
        "Luật Giao thông 2024 quy định tốc độ tối đa trong khu dân cư là 50 km/h.",
        "Theo quy định năm 2016, tốc độ tối đa trong khu dân cư là 60 km/h."
    ]
    print("🧠 Embedding passages with Gemini…")
    vecs = embed_texts(passages)

    print("📤 Upserting vectors via gRPC…")
    upsert_payload = [
        {
            "id": f"{doc_id}_p{i}",
            "values": vecs[i],  # 768 floats
            "metadata": {"doc_id": doc_id, "chunk_id": i, "text": passages[i], "kind": "kb_document"},
        }
        for i in range(len(passages))
    ]
    resp = index.upsert(vectors=upsert_payload, namespace=NAMESPACE)
    print("🔎 UpsertResponse:", resp)

    # 5) Chờ đồng bộ vector_count tăng
    for _ in range(10):
        stats = index.describe_index_stats()
        count = stats.namespaces.get(NAMESPACE, {}).get("vector_count", 0)
        print(f"📈 vector_count in '{NAMESPACE}':", count)
        if count >= len(passages):
            break
        time.sleep(1.5)

    # 6) Embed câu hỏi + query
    question = "Tốc độ tối đa trong khu dân cư hiện nay là bao nhiêu?"
    print("🔍 Embedding question & querying…")
    q_vec = embed_texts([question])[0]
    res = index.query(
        vector=q_vec,
        top_k=3,
        include_metadata=True,
        namespace=NAMESPACE,
    )

    print("📊 Matches:")
    if not res.matches:
        print("⚠️ No matches. Check dimension, namespace, or vector_count above.")
    else:
        for m in res.matches:
            md = m.metadata or {}
            print(f"- score={m.score:.3f} id={m.id}")
            print("  text:", md.get("text", ""))
            print("  ---")

if __name__ == "__main__":
    main()
