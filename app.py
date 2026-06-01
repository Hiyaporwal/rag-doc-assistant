import os
import uuid

import streamlit as st

from ingest import ingest_documents
from rag_pipeline import ask_question, load_qa_chain


st.set_page_config(page_title="Document Q&A", layout="wide")
st.title("Chat with Your Documents")


def load_chain_safely():
    try:
        return load_qa_chain()
    except Exception:
        return None


with st.sidebar:
    st.header("Upload Documents")
    uploaded_files = st.file_uploader(
        "Upload PDFs",
        type="pdf",
        accept_multiple_files=True,
    )

    if uploaded_files:
        upload_fingerprint = tuple((file.name, file.size) for file in uploaded_files)
        if st.session_state.get("upload_fingerprint") != upload_fingerprint:
            upload_id = uuid.uuid4().hex
            data_path = os.path.join("data", upload_id)
            db_path = os.path.join("chroma_db", upload_id)
            os.makedirs(data_path, exist_ok=True)
            for file in uploaded_files:
                file.seek(0)
                with open(os.path.join(data_path, file.name), "wb") as f:
                    f.write(file.read())
            st.session_state.upload_fingerprint = upload_fingerprint
            st.session_state.data_path = data_path
            st.session_state.db_path = db_path
            st.session_state.chain = None
            st.session_state.messages = []
        st.success(f"{len(uploaded_files)} file(s) uploaded.")

    if st.button("Process Documents"):
        with st.spinner("Embedding documents..."):
            try:
                st.session_state.chain = None
                data_path = st.session_state.get("data_path", "data")
                db_path = st.session_state.get("db_path", "chroma_db")
                ingest_documents(data_path=data_path, db_path=db_path, recreate_db=False)
                st.session_state.chain = load_qa_chain(db_path=db_path)
            except Exception as exc:
                st.session_state.chain = None
                st.error(str(exc))
                st.stop()
        st.success("Ready to chat.")


if "messages" not in st.session_state:
    st.session_state.messages = []

if "chain" not in st.session_state:
    st.session_state.chain = load_chain_safely()

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

if question := st.chat_input("Ask a question about your documents..."):
    if not st.session_state.get("chain"):
        st.warning("Please upload and process at least one PDF before asking questions.")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                answer, sources = ask_question(st.session_state.chain, question)
            except Exception as exc:
                message = str(exc)
                if "invalid_api_key" in message or "Invalid API Key" in message:
                    st.error(
                        "Groq rejected your API key. Create a new Groq API key, update .env, "
                        "then restart Streamlit."
                    )
                else:
                    st.error(message)
                st.stop()
            st.write(answer)

            with st.expander("Sources"):
                seen_sources = set()
                for doc in sources:
                    source = os.path.basename(doc.metadata.get("source", "Unknown file"))
                    page = doc.metadata.get("page", "?")
                    source_key = (source, page, doc.page_content[:80])
                    if source_key in seen_sources:
                        continue
                    seen_sources.add(source_key)
                    preview = doc.page_content[:200]
                    st.write(f"- {source}, page {page}: {preview}...")

    st.session_state.messages.append({"role": "assistant", "content": answer})
