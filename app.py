import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import tempfile
 
# ── Load API key from .env ──────────────────────────────────────────
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
 
# ── Page config ─────────────────────────────────────────────────────
st.set_page_config(page_title="RAG Document Q&A", page_icon="📄")
st.title("📄 RAG Document Q&A")
st.caption("Built by Samhitha Chamarthi | GenAI / ML Engineer")
st.divider()
 
# ── Session state for chat history ──────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
 
if "retriever" not in st.session_state:
    st.session_state.retriever = None
 
# ── Sidebar ─────────────────────────────────────────────────────────
with st.sidebar:
    st.header("📂 Upload Your PDF")
    uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])
 
    if uploaded_file:
        st.success(f"✅ Uploaded: {uploaded_file.name}")
 
    st.divider()
    if st.button("🗑️ Clear Chat History"):
        st.session_state.chat_history = []
        st.success("Chat cleared!")
 
    st.divider()
    st.markdown("**How it works:**")
    st.markdown("1. Upload a PDF")
    st.markdown("2. Ask any question")
    st.markdown("3. AI answers from your document")
 
# ── Process PDF when uploaded ────────────────────────────────────────
if uploaded_file and st.session_state.retriever is None:
    with st.spinner("Reading and processing your PDF..."):
 
        # Save uploaded file to a temp location
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name
 
        # Step 1: Load the PDF
        loader = PyMuPDFLoader(tmp_path)
        documents = loader.load()
 
        # Step 2: Split into chunks
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )
        chunks = splitter.split_documents(documents)
 
        # Step 3: Create embeddings and store in ChromaDB
        embeddings = OpenAIEmbeddings(api_key=openai_api_key)
        vectorstore = FAISS.from_documents(chunks, embeddings)
 
        # Step 4: Create retriever
        st.session_state.retriever = vectorstore.as_retriever(
            search_kwargs={"k": 3}
        )
 
    st.success(f"✅ PDF processed! {len(chunks)} chunks created. Ask your question below.")
 
# ── Q&A Section ──────────────────────────────────────────────────────
if st.session_state.retriever:
 
    # Show chat history
    for chat in st.session_state.chat_history:
        with st.chat_message("user"):
            st.write(chat["question"])
        with st.chat_message("assistant"):
            st.write(chat["answer"])
            st.caption(f"📄 Source: {chat['source']}")
 
    # Question input
    question = st.chat_input("Ask a question about your document...")
 
    if question:
        # Build the prompt
        prompt = PromptTemplate(
            template="""Use the following context from the document to answer the question.
If you don't know the answer from the context, say "I couldn't find that in the document."
Always mention which part of the document the answer came from.
 
Context:
{context}
 
Question: {question}
 
Answer:""",
            input_variables=["context", "question"]
        )
 
        # Build the RAG chain
        llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            api_key=openai_api_key,
            temperature=0
        )
 
        def format_docs(docs):
            return "\n\n".join(
                f"[Page {doc.metadata.get('page', '?') + 1}]: {doc.page_content}"
                for doc in docs
            )
 
        rag_chain = (
            {"context": st.session_state.retriever | format_docs,
             "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
 
        # Get answer
        with st.spinner("Thinking..."):
            answer = rag_chain.invoke(question)
 
            # Get source pages
            source_docs = st.session_state.retriever.invoke(question)
            pages = list(set(
                str(doc.metadata.get('page', '?') + 1)
                for doc in source_docs
            ))
            source = f"Pages {', '.join(pages)}"
 
        # Show answer
        with st.chat_message("user"):
            st.write(question)
        with st.chat_message("assistant"):
            st.write(answer)
            st.caption(f"📄 Source: {source}")
 
        # Save to chat history
        st.session_state.chat_history.append({
            "question": question,
            "answer": answer,
            "source": source
        })
 
else:
    st.info("👈 Upload a PDF from the sidebar to get started!")