# 🧠 Word RAG AI Add-in – Trợ lý văn bản pháp lý trong Word bằng Gemini & Pinecone

> **Một add-in cho MS Word hỗ trợ hỏi – đáp, tra cứu và phân tích văn bản pháp lý ngay trong tài liệu.**  
> ⚡ Dùng kỹ thuật RAG (Retrieval-Augmented Generation) kết hợp Google Gemini + Pinecone VectorDB.

---

## ✨ Tính năng chính

- 🤖 Chat AI trực tiếp trên Word dựa trên nội dung tài liệu
- 🔍 Tìm kiếm thông minh trong kho dữ liệu (KB) qua Pinecone VectorDB
- 📚 Lưu văn bản pháp luật vào “Knowledge Base” để tra cứu sau
- 📎 Trả lời có trích dẫn nguồn đoạn văn liên quan
- 💬 Hỗ trợ người dùng không cần kiến thức lập trình
- 🇻🇳 Hỗ trợ ngôn ngữ tiếng Việt (tích hợp Google Gemini)

---

## 📸 Demo (Screenshots)

![taskpane-demo](./screenshots/taskpane_demo.png)
*Ảnh minh họa giao diện Word Add-in với giao diện Chat + Quản lý KB*

---

## 🧱 Kiến trúc tổng quan

+-------------+ HTTP +-----------------+ gRPC +-----------------+
| Word Add-in | <--------------> | FastAPI Backend | <-------------------> | Pinecone Vector |
| (Office.js) | | (Python) | | Database |
+-------------+ +-----------------+ +-----------------+
|
Gemini API

yaml
Sao chép mã

---

## 🛠️ Tech Stack

| Layer         | Technology                               |
|---------------|-------------------------------------------|
| Frontend      | JavaScript, HTML, CSS, Office.js          |
| Backend       | Python, FastAPI, Uvicorn                  |
| AI / Model    | Google Gemini API (`gemini-2.0-flash`, `text-embedding-004`) |
| Vector DB     | Pinecone (Serverless, gRPC)               |
| DevOps/Tools  | Docker, VSCode, HTTPS dev certs           |

---

## 🚀 Cách cài đặt & chạy dự án

### 1. Clone dự án
```bash
git clone https://github.com/<username>/word-rag-addin.git
cd word-rag-addin
2. Cài đặt backend
bash
Sao chép mã
cd backend
python -m venv .venv
source .venv/bin/activate  # hoặc .venv\Scripts\activate trên Windows
pip install -r requirements.txt
3. Config biến môi trường
Tạo file .env:

env
Sao chép mã
GOOGLE_API_KEY=<your-google-api-key>
PINECONE_API_KEY=<your-pinecone-api-key>
4. Khởi động server
bash
Sao chép mã
uvicorn server:app --host 0.0.0.0 --port 8000 --reload --ssl-keyfile ./certs/key.pem --ssl-certfile ./certs/cert.pem
5. Chạy Word Add-in
bash
Sao chép mã
cd web
npm install
npm start
Sau đó mở Word, chọn:

Insert → My Add-ins → Upload Add-in → Chọn manifest.xml

📘 Các API chính
Method	Endpoint	Mô tả
POST	/ask	Hỏi – đáp văn bản
POST	/index_document	Lưu tài liệu vào KB
POST	/search_knowledge_base	Tìm trong KB
GET	/list_documents	Lấy tổng quan kho dữ liệu
GET	/ping	Kiểm tra kết nối Pinecone

⚙️ Cấu trúc dự án
Sao chép mã
├── backend/
│   ├── server.py
│   ├── requirements.txt
│   ├── certs/
│   └── ...
├── web/
│   ├── taskpane.html
│   ├── taskpane.js
│   ├── taskpane.css
│   └── manifest.xml
├── screenshots/
│   └── taskpane_demo.png
└── README.md
🧪 Test Pinecone kết nối
Bạn có thể chạy test đơn giản:

python
Sao chép mã
from pinecone.grpc import PineconeGRPC as Pinecone
pc = Pinecone(api_key="...")
index = pc.Index("docs-example")
index.upsert([...])
print(index.query(...))
🧩 Ghi chú
Tối ưu RAM bằng cách xử lý theo từng chunk nhỏ

Dùng integrated model của Pinecone để tránh tự xử lý embedding

Sử dụng HTTPS dev cert cho Word

🗺️ Hướng phát triển
Highlight trực tiếp đoạn văn trong Word theo kết quả AI

Tự động phát hiện nội dung trùng lặp

Xuất báo cáo pháp lý tự động theo câu hỏi

Hỗ trợ OCR từ văn bản scan

