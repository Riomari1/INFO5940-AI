import streamlit as st
import os
from openai import OpenAI
from typing import List, Dict
from pypdf import PdfReader
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Chunking strategy justification:
# choose_chunk_size() adjusts chunk length based on total document size.
# Smaller docs (<50k chars) get larger chunks (1500) for richer context,
# while very large docs use smaller chunks (500–1000) for faster retrieval.
# This balances accuracy and performance by controlling the number of embeddings.
# Streamlit sidebar adds a dynamic option (checkbox + inputs) allowing users
# to manually override chunk size and overlap for flexible experimentation.



api_key = os.getenv("API_KEY")
if not api_key:
    st.error("Missing API key. Set `API_KEY` in environment.")
    st.stop()

client = OpenAI(
    api_key=api_key,
    base_url="https://api.ai.it.cornell.edu",
)

st.title("📝 File Q&A with OpenAI")

# Sidebar controls for chunking and retrieval
with st.sidebar:
    st.header("Settings")
    dynamic_chunks = st.checkbox("Dynamic chunk size", value=True)
    default_chunk = 1000
    chunk_size_input = st.number_input(
        "Chunk size", value=default_chunk, min_value=100, max_value=4000, step=100
    )
    chunk_overlap_input = st.number_input(
        "Chunk overlap", value=0, min_value=0, max_value=1000, step=50
    )
    top_k = st.slider("Top K chunks", min_value=1, max_value=20, value=5)
    if st.button("Reset index"):
        st.session_state["index"] = []
        st.session_state["loaded_filenames"] = set()
        st.success("Index cleared.")
uploaded_files = st.file_uploader(
    "Upload document(s)", type=("txt", "md", "pdf"), accept_multiple_files=True
)

question = st.chat_input(
    "Ask something about the documents",
    disabled=not uploaded_files,
)

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Upload files (.txt/.pdf) and ask a question."}
    ]
if "index" not in st.session_state:
    st.session_state["index"] = []  # list of {text, source, chunk_id}
if "loaded_filenames" not in st.session_state:
    st.session_state["loaded_filenames"] = set()
if "vectorstore" not in st.session_state:
    st.session_state["vectorstore"] = None
if "retriever" not in st.session_state:
    st.session_state["retriever"] = None

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

def read_file_to_text(file) -> str:
    name = file.name.lower()
    if name.endswith((".txt", ".md")):
        return file.read().decode("utf-8", errors="ignore")
    if name.endswith(".pdf"):
        try:
            reader = PdfReader(file)
            pages = [p.extract_text() or "" for p in reader.pages]
            return "\n".join(pages)
        except Exception as e:
            st.warning(f"Failed to parse PDF {file.name}: {e}")
            return ""
    return ""

def choose_chunk_size(total_chars: int) -> int:
    if total_chars <= 50_000:
        return 1500
    if total_chars <= 200_000:
        return 1000
    return 500

def split_recursive(text: str, chunk_size: int, chunk_overlap: int = 0) -> List[str]:
    # Similar to LangChain's RecursiveCharacterTextSplitter
    seps = ["\n\n", "\n", " ", ""]
    def split_by_sep(t: str, sep_idx: int) -> List[str]:
        sep = seps[sep_idx]
        if sep == "":
            return [t[i : i + chunk_size] for i in range(0, len(t), chunk_size)]
        parts = t.split(sep)
        out: List[str] = []
        cur = ""
        for p in parts:
            candidate = (cur + sep + p) if cur else p
            if len(candidate) <= chunk_size:
                cur = candidate
            else:
                if cur:
                    out.append(cur)
                if len(p) <= chunk_size:
                    cur = p
                else:
                    if sep_idx + 1 < len(seps):
                        subparts = split_by_sep(p, sep_idx + 1)
                        out.extend(subparts[:-1])
                        cur = subparts[-1] if subparts else ""
                    else:
                        # fallback
                        out.append(p[:chunk_size])
                        cur = p[chunk_size:]
        if cur:
            out.append(cur)
        # apply overlap if requested
        if chunk_overlap > 0 and len(out) > 1:
            overlapped: List[str] = []
            for i, ch in enumerate(out):
                if i == 0:
                    overlapped.append(ch)
                else:
                    prev_tail = out[i - 1][-chunk_overlap:]
                    overlapped.append(prev_tail + ch)
            return overlapped
        return out

    return split_by_sep(text, 0)

def add_documents_to_index(files: List) -> None:
    for f in files:
        if f.name in st.session_state["loaded_filenames"]:
            continue
        text = read_file_to_text(f)
        if not text.strip():
            continue
        # dynamic or manual chunk size
        size = choose_chunk_size(len(text)) if dynamic_chunks else int(chunk_size_input)
        overlap = int(chunk_overlap_input)
        chunks = split_recursive(text, chunk_size=size, chunk_overlap=overlap)
        for i, ch in enumerate(chunks):
            st.session_state["index"].append(
                {"text": ch, "source": f.name, "chunk_id": i}
            )
        st.session_state["loaded_filenames"].add(f.name)
    # display index summary
    if st.session_state["index"]:
        st.caption(
            f"Indexed {len(st.session_state['index'])} chunks from {len(st.session_state['loaded_filenames'])} file(s)."
        )
        with st.expander("Chunk samples"):
            for doc in st.session_state["index"][:2]:
                st.markdown(f"- `{doc['source']}#{doc['chunk_id']}`: {doc['text'][:300]}...")

def score_chunk(query: str, chunk: str) -> int:
    # simple token overlap score
    q_tokens = set(query.lower().split())
    c_tokens = set(chunk.lower().split())
    return len(q_tokens & c_tokens)

def retrieve(query: str, k: int = 5) -> List[Dict]:
    scored = [
        (score_chunk(query, item["text"]), item)
        for item in st.session_state["index"]
    ]
    scored.sort(key=lambda x: x[0], reverse=True)
    return [item for score, item in scored[:k] if score > 0] or [
        item for _, item in scored[:k]
    ]

if uploaded_files:
    add_documents_to_index(uploaded_files)

# LangChain + Chroma RAG pipeline
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_openai import OpenAIEmbeddings
    from langchain_community.vectorstores import Chroma
    have_langchain = True
except Exception:
    have_langchain = False

def ensure_vectorstore():
    if not have_langchain:
        return
    # Build splitter per current settings
    size = 1000
    if dynamic_chunks:
        # approximate based on combined text length
        total_chars = sum(len(it["text"]) for it in st.session_state["index"]) or 0
        size = choose_chunk_size(total_chars)
    else:
        size = int(chunk_size_input)
    overlap = int(chunk_overlap_input)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=size, chunk_overlap=overlap, separators=["\n\n", "\n", " ", ""]
    )

    # Convert our indexed chunks into LC Documents (no re-splitting, just build docs)
    from langchain_core.documents import Document
    docs = [
        Document(page_content=it["text"], metadata={"source": it["source"], "chunk_id": it["chunk_id"]})
        for it in st.session_state["index"]
    ]

    # Initialize embeddings + vectorstore (in-memory by default)
    embeddings = OpenAIEmbeddings(
        model="openai.text-embedding-3-small",
        api_key=api_key,
        base_url="https://api.ai.it.cornell.edu",
    )
    st.session_state["vectorstore"] = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        collection_name="session_index",
    )
    st.session_state["retriever"] = st.session_state["vectorstore"].as_retriever(
        search_type="similarity", search_kwargs={"k": top_k}
    )

# Ensure vectorstore is built when we have new docs and LangChain available
if uploaded_files and have_langchain:
    ensure_vectorstore()

if question and uploaded_files:

    # Append the user's question to the messages
    st.session_state.messages.append({"role": "user", "content": question})
    st.chat_message("user").write(question)

    # Retrieve relevant chunks and build context with sources
    top_chunks: List[Dict]
    if st.session_state.get("retriever") is not None:
        docs = st.session_state["retriever"].invoke(question)
        top_chunks = [
            {
                "text": d.page_content,
                "source": d.metadata.get("source", "unknown"),
                "chunk_id": d.metadata.get("chunk_id", 0),
            }
            for d in docs
        ]
    else:
        # Fallback to simple overlap retriever
        top_chunks = retrieve(question, k=top_k)
    sources_text = "\n\n".join(
        [
            f"[Source: {c['source']} | chunk {c['chunk_id']}]\n{c['text']}"[:2000]
            for c in top_chunks
        ]
    )
    system_preamble = (
        "You are a helpful assistant. Use only the provided sources to answer. "
        "If the answer is not in the sources, say you cannot find it."
    )

    with st.chat_message("assistant"):
        stream = client.chat.completions.create(
            model="openai.gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_preamble},
                {"role": "system", "content": f"SOURCES:\n{sources_text}"},
                *st.session_state.messages,
            ],
            stream=True,
        )
        response = st.write_stream(stream)

    # Append the assistant's response and show sources
    st.session_state.messages.append({"role": "assistant", "content": response})
    with st.expander("Show sources"):
        for c in top_chunks:
            st.markdown(f"- `{c['source']}#{c['chunk_id']}`")
