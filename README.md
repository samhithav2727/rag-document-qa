# 📄 RAG Document Q&A

An AI-powered document question answering app built using Retrieval Augmented Generation (RAG). Upload any PDF and ask questions — get instant, accurate answers!

## 🌐 Live Demo
[👉 Click here to try the app](https://rag-document-app-lzy2ijw7qjwd5mhzhpcec4.streamlit.app)

## ✨ Features
- 📤 Upload any PDF document
- 💬 Ask questions in natural language
- 🔍 Retrieves relevant chunks using vector search
- 🤖 Answers powered by OpenAI GPT
- ⚡ Fast responses with FAISS vector store

## 🛠️ Tech Stack
| Tool | Purpose |
|------|---------|
| LangChain | RAG pipeline |
| OpenAI GPT | Language model |
| FAISS | Vector store |
| Streamlit | Web UI |
| Python | Backend |

## 📁 Project Structure
rag-document-qa/
├── app.py            ← Main Streamlit app
├── requirements.txt  ← Dependencies
└── .gitignore

## ⚙️ Run Locally

# 1. Clone the repo
git clone https://github.com/samhithav2727/rag-document-qa.git
cd rag-document-qa

# 2. Install dependencies
pip install -r requirements.txt

# 3. Add your API key
echo "OPENAI_API_KEY=your-key-here" > .env

# 4. Run the app
streamlit run app.py

## 🔑 Environment Variables
| Variable | Description |
|----------|-------------|
| OPENAI_API_KEY | Your OpenAI API key from platform.openai.com |

## 💡 How It Works
1. User uploads a PDF document
2. Document is split into chunks
3. Chunks are embedded and stored in FAISS vector store
4. User asks a question
5. Relevant chunks are retrieved
6. OpenAI GPT generates an accurate answer

## 👨‍💻 Author
**samhithav2727** — [GitHub](https://github.com/samhithav2727)
