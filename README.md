# 📄 RAG PDF Chat

A Retrieval-Augmented Generation (RAG) application that lets you upload PDF documents and ask questions using LangChain, ChromaDB, Groq LLM, and Streamlit.

---

## 🚀 Features

- 📁 Upload one or multiple PDF documents
- 💬 Chat with your documents in natural language
- 🔍 Retrieves most relevant chunks using vector similarity search
- 🤖 Powered by Groq LLM (free & fast)
- 🧠 HuggingFace embeddings (runs locally, no cost)
- 🗄️ ChromaDB for persistent vector storage
- 🖥️ Clean Streamlit UI

---

## 🛠️ Tech Stack

| Layer | Tool |
|---|---|
| Framework | LangChain |
| LLM | Groq (llama-3.3-70b-versatile) |
| Embeddings | HuggingFace (all-MiniLM-L6-v2) |
| Vector DB | ChromaDB |
| UI | Streamlit |
| PDF Loader | PyPDF |

---

## 📁 Project Structure

```
rag-pdf-chat/
│
├── app.py               # Streamlit UI
├── ingest.py            # Load, chunk & embed documents
├── rag_pipeline.py      # Retrieval + generation chain
├── .env                 # API keys (not committed)
├── requirements.txt     # Dependencies
│
├── data/                # Place your PDF files here
└── chroma_db/           # Vector DB (auto-created)
```

---

## ⚙️ Setup & Installation

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/rag-pdf-chat.git
cd rag-pdf-chat
```

### 2. Create Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Mac/Linux
python -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Set Up Environment Variables

Create a `.env` file in the root directory:
```env
GROQ_API_KEY=your_groq_api_key_here
```

Get your free Groq API key at → https://console.groq.com

---

## ▶️ Running the App

### Step 1 — Add your PDF
Place your PDF file inside the `/data` folder.

### Step 2 — Embed Documents
```bash
python ingest.py
```

### Step 3 — Launch the App
```bash
streamlit run app.py
```

Open your browser at → **http://localhost:8501**

---

## 💡 How It Works

```
PDF → Load → Chunk → Embed → Store in ChromaDB
                                      ↓
User Query → Embed Query → Search ChromaDB → Top K Chunks
                                                    ↓
                                         Groq LLM + Context → Answer
```

1. **Ingestion** — PDFs are loaded, split into chunks, embedded using HuggingFace, and stored in ChromaDB
2. **Retrieval** — User query is embedded and matched against stored chunks using cosine similarity
3. **Generation** — Top matching chunks are passed to Groq LLM as context to generate an answer

---

## 📦 Requirements

```txt
langchain>=0.2.0
langchain-groq
langchain-chroma
langchain-community
langchain-huggingface
chromadb
sentence-transformers
streamlit
streamlit-chat
pypdf
python-dotenv
```

---

## 🔑 Environment Variables

| Variable | Description |
|---|---|
| `GROQ_API_KEY` | Your free Groq API key from console.groq.com |

---

## ⚠️ Important Notes

- Always run `python ingest.py` before launching the app
- Stop Streamlit (`Ctrl + C`) before deleting the `chroma_db` folder
- Never commit your `.env` file to GitHub
- Add `.env` and `chroma_db/` to `.gitignore`

---

## 📝 .gitignore

```
venv/
.env
chroma_db/
__pycache__/
*.pyc
.DS_Store
```

---

## 🌐 Deploy on Streamlit Cloud

1. Push your code to GitHub
2. Go to → https://streamlit.io/cloud
3. Connect your GitHub repository
4. Add `GROQ_API_KEY` in the Secrets section
5. Click Deploy!

---

## 🙌 Acknowledgements

- [LangChain](https://langchain.com)
- [Groq](https://groq.com)
- [ChromaDB](https://trychroma.com)
- [Streamlit](https://streamlit.io)
- [HuggingFace](https://huggingface.co)

---


