import os, time, json, requests
from pinecone import Pinecone

API_KEY   = os.getenv("PINECONE_API_KEY") or "PASTE_YOUR_KEY"
INDEX     = "word-rag-integrated"   # đúng tên index của bạn
NAMESPACE = "documents"

# 1) Lấy host & spec (bằng SDK)
pc   = Pinecone(api_key=API_KEY)
desc = pc.describe_index(INDEX).to_dict()
host = desc["host"]
print("✅ Connected. Host:", host)
print("Integrated spec:", desc.get("embed"))

base = f"https://{host}"
hdrs = {"Api-Key": API_KEY, "Content-Type": "application/json"}

# (tuỳ chọn) dọn rác cũ qua SDK
try:
    idx = pc.Index(INDEX)
    idx.delete(filter={"doc_id": "rest_demo_doc"}, namespace=NAMESPACE)
except Exception:
    pass

# 2) UPSERT qua REST: /vectors/upsert  (KHÔNG gửi values; dùng metadata.text)
print("📤 Upserting 2 records via REST /vectors/upsert ...")
payload_upsert = {
    "namespace": NAMESPACE,
    "vectors": [
        {
            "id": "rest_demo_doc_1",
            "metadata": {
                "doc_id": "rest_demo_doc",
                "chunk_id": 1,
                # phải trùng field_map {"text":"text"}
                "text": "Luật Giao thông 2024 quy định tốc độ tối đa trong khu dân cư là 50 km/h.",
                "kind": "kb_document"
            }
        },
        {
            "id": "rest_demo_doc_2",
            "metadata": {
                "doc_id": "rest_demo_doc",
                "chunk_id": 2,
                "text": "Theo quy định cũ 2016, giới hạn tốc độ trong khu dân cư là 60 km/h.",
                "kind": "kb_document"
            }
        }
    ]
}
ru = requests.post(f"{base}/vectors/upsert", headers=hdrs, data=json.dumps(payload_upsert), timeout=30)
print("Upsert status:", ru.status_code, ru.text)
ru.raise_for_status()  # ném lỗi nếu không 2xx

# 3) Đợi đồng bộ một chút
time.sleep(2)

# 4) QUERY qua REST: /query  (gửi text, integrated sẽ tự embed)
question = "Tốc độ tối đa trong khu dân cư hiện nay là bao nhiêu?"
print("🔍 Query:", question)
payload_query = {
    "namespace": NAMESPACE,
    "topK": 3,
    "includeMetadata": True,
    "text": question
}
rq = requests.post(f"{base}/query", headers=hdrs, data=json.dumps(payload_query), timeout=30)
print("Query status:", rq.status_code, rq.text)
rq.raise_for_status()
data = rq.json()

print("📊 Matches:")
for m in (data.get("matches") or []):
    md = m.get("metadata") or {}
    print(f"- score={m.get('score'):.3f} id={m.get('id')}")
    print("  text:", md.get("text", ""))
    print("  ---")

# 5) Stats qua SDK (in cho chắc)
stats = idx.describe_index_stats()
print("📈 Namespaces:", stats.namespaces)
print("✅ DONE")
