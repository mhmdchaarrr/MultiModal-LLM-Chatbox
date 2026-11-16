"""
RAG System - Streamlit Web Interface
BST-AMS15 Engineering Teamwork I

Run with: streamlit run streamlit_app.py
"""

import streamlit as st
from rag_system import RAGPipeline

# ==================================================
# PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="RAG System - BST-AMS15",
    page_icon="🤖",
    layout="wide"
)

# ==================================================
# INITIALIZE SESSION STATE
# ==================================================

@st.cache_resource
def get_rag_pipeline(model_name: str):
    """Create RAG pipeline with specified Ollama model"""
    return RAGPipeline(
        chunk_size=1200,
        chunk_overlap=150,
        embedding_model="sentence-transformers",
        ollama_model=model_name
    )

if "ollama_model" not in st.session_state:
    st.session_state.ollama_model = "llama3.2"

if "rag_pipeline" not in st.session_state:
    st.session_state.rag_pipeline = get_rag_pipeline(st.session_state.ollama_model)

if "messages" not in st.session_state:
    st.session_state.messages = []

if "documents_loaded" not in st.session_state:
    st.session_state.documents_loaded = []

# ==================================================
# SIDEBAR
# ==================================================

with st.sidebar:
    st.title("📚 RAG System")
    
    st.markdown("---")
    st.subheader("🦙 Ollama Model")
    
    ollama_model = st.selectbox(
        "Select model",
        ["llama3.2", "mistral", "phi3", "llama3.1:8b"],
        index=0
    )
    
    # Update model if changed
    if ollama_model != st.session_state.ollama_model:
        st.session_state.ollama_model = ollama_model
        st.session_state.rag_pipeline.ollama_model = ollama_model
        st.success(f"✅ Switched to {ollama_model}")
    
    st.caption("⚠️ Make sure Ollama is running: `ollama serve`")

    st.markdown("---")
    st.subheader("📄 Add Documents")

    uploaded_file = st.file_uploader(
        "Upload PDF or Text file",
        type=["pdf", "txt"]
    )

    if uploaded_file:
        temp_path = f"/tmp/{uploaded_file.name}"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        file_type = "pdf" if uploaded_file.name.lower().endswith(".pdf") else "text"

        if st.button(f"Process {uploaded_file.name}"):
            with st.spinner("Processing..."):
                try:
                    st.session_state.rag_pipeline.add_document(temp_path, file_type)
                    st.session_state.documents_loaded.append(uploaded_file.name)
                    st.success(f"✅ Added: {uploaded_file.name}")
                except Exception as e:
                    st.exception(e)

    st.markdown("---")
    st.subheader("🌐 Add Web / YouTube")

    url_input = st.text_input("Enter URL")
    url_type = st.selectbox("Type", ["web", "youtube"])

    if st.button("Add URL"):
        url = (url_input or "").strip()
        if not url:
            st.error("Enter a URL")
        elif url_type == "youtube" and "youtu" not in url:
            st.error("Enter a YouTube URL")
        else:
            with st.spinner("Fetching..."):
                try:
                    st.session_state.rag_pipeline.add_document(url, url_type)
                    st.session_state.documents_loaded.append(url[:50] + "...")
                    st.success(f"✅ Added!")
                except Exception as e:
                    st.exception(e)

    st.markdown("---")
    st.subheader("📋 Loaded")
    if st.session_state.documents_loaded:
        for doc in st.session_state.documents_loaded:
            st.write(f"• {doc}")
    else:
        st.write("No documents yet")

    st.markdown("---")
    st.subheader("⚙️ Settings")
    top_k = st.slider("Chunks to retrieve", 1, 10, 5)
    show_sources = st.checkbox("Show sources", value=True)

    if st.button("🗑️ Reset"):
        st.session_state.rag_pipeline.vector_db.clear()
        st.session_state.documents_loaded = []
        st.session_state.messages = []
        st.rerun()

# ==================================================
# MAIN CHAT
# ==================================================

st.title("🤖 RAG Chat System")
st.caption(f"BST-AMS15 | Using Ollama ({ollama_model}) - FREE, Local, No API Key")

# Chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message and show_sources:
            with st.expander("📖 Sources"):
                for src in message["sources"]:
                    st.markdown(f"**Score:** {src['score']:.3f} | **Source:** {src['source']}")
                    st.code(src['content'][:400] + "...")

# Chat input
if prompt := st.chat_input("Ask a question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        if not st.session_state.documents_loaded:
            response = "⚠️ Upload documents first!"
            st.markdown(response)
        else:
            with st.spinner("Thinking..."):
                results = st.session_state.rag_pipeline.retrieve(prompt, top_k)
                
                sources = [{"score": s, "source": d.source, "content": d.content} 
                          for d, s in results]
                
                response = st.session_state.rag_pipeline.generate_answer(prompt, top_k)
                st.markdown(response)

                if show_sources and sources:
                    with st.expander("📖 Sources"):
                        for src in sources:
                            st.markdown(f"**Score:** {src['score']:.3f} | **Source:** {src['source']}")
                            st.code(src['content'][:400] + "...")

        st.session_state.messages.append({
            "role": "assistant",
            "content": response,
            "sources": sources if st.session_state.documents_loaded else []
        })

# ==================================================
# FOOTER
# ==================================================

st.markdown("---")
c1, c2, c3 = st.columns(3)
c1.metric("Documents", len(st.session_state.documents_loaded))
c2.metric("Chunks", len(st.session_state.rag_pipeline.vector_db.documents))
c3.metric("Messages", len(st.session_state.messages))