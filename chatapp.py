# chatapp.py — Clean GPT-style final answers + evidence-first verification
import os
import streamlit as st
from dotenv import load_dotenv
from pathlib import Path
from typing import List, Tuple
import re

from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

# ---------------- Config ----------------
API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    raise SystemExit("❌ GOOGLE_API_KEY not found in .env")

DOCS_DIR = Path("RAG-BOT/docs")    # put your PDFs here (or upload from sidebar)
FAISS_DIR = "faiss_index"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
TOP_K = 5

# ---------------- Models ----------------
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.1, google_api_key=API_KEY)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# ---------------- Helpers ----------------
def safe_text(resp) -> str:
    """
    Extract readable text from many possible LLM return formats.
    Returns a cleaned string.
    """
    if resp is None:
        return ""

    # If it's already a string
    if isinstance(resp, str):
        return clean_response_text(resp)

    # Check if it's a LangChain message object
    if hasattr(resp, 'content'):
        content = resp.content
        if isinstance(content, str):
            return clean_response_text(content)
        elif isinstance(content, list):
            # Handle case where content is a list of text blocks
            text_parts = []
            for item in content:
                if hasattr(item, 'text'):
                    text_parts.append(item.text)
                elif isinstance(item, dict) and 'text' in item:
                    text_parts.append(item['text'])
                elif isinstance(item, str):
                    text_parts.append(item)
            return clean_response_text(' '.join(text_parts))

    # If it's dict-like
    if isinstance(resp, dict):
        # common keys to try
        for k in ("content", "text", "output_text", "result", "answer"):
            if k in resp and resp[k]:
                return clean_response_text(str(resp[k]))
        
        # Some wrappers put candidates: [{ "content": "..."}]
        if "candidates" in resp and isinstance(resp["candidates"], (list, tuple)) and len(resp["candidates"]) > 0:
            first = resp["candidates"][0]
            if isinstance(first, dict):
                content = first.get("content") or first.get("text") or first.get("message", {}).get("content")
                if content:
                    return clean_response_text(str(content))
            return clean_response_text(str(first))
        
        # try nested message field
        if "message" in resp:
            msg = resp["message"]
            if isinstance(msg, str):
                return clean_response_text(msg)
            if isinstance(msg, dict):
                for k in ("content", "text"):
                    if k in msg and msg[k]:
                        return clean_response_text(str(msg[k]))

    # If object with attributes
    for attr in ("text", "content", "output_text", "result"):
        if hasattr(resp, attr):
            val = getattr(resp, attr)
            if isinstance(val, str):
                return clean_response_text(val)
            try:
                return clean_response_text(str(val))
            except Exception:
                pass

    # fallback to string conversion
    try:
        return clean_response_text(str(resp))
    except Exception:
        return ""

def clean_response_text(text: str) -> str:
    """Clean and format response text to make it more natural"""
    if not text:
        return ""
    
    # Remove common JSON-like patterns and metadata
    text = re.sub(r'<bound method.*?>', '', text, flags=re.DOTALL)
    text = re.sub(r'\{["\']?content["\']?:\s*["\']', '', text)
    text = re.sub(r'["\'],?\s*["\']?additional_kwargs["\']?:\s*\{.*?\}', '', text, flags=re.DOTALL)
    text = re.sub(r'["\'],?\s*["\']?response_metadata["\']?:\s*\{.*?\}', '', text, flags=re.DOTALL)
    text = re.sub(r'["\'],?\s*["\']?id["\']?:\s*["\']run-[a-f0-9-]+["\']', '', text)
    text = re.sub(r'["\'],?\s*["\']?usage_metadata["\']?:\s*\{.*?\}', '', text, flags=re.DOTALL)
    text = re.sub(r'\{["\']?input_tokens["\']?:\s*\d+.*?\}', '', text, flags=re.DOTALL)
    
    # Remove extra quotes and brackets
    text = re.sub(r'^["\'\[\{]+|["\'\]\}]+$', '', text.strip())
    text = re.sub(r'\\n', '\n', text)
    text = re.sub(r'\\t', '\t', text)
    text = re.sub(r'\\"', '"', text)
    
    # Clean up multiple spaces and newlines
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\n\s*\n', '\n\n', text)
    
    return text.strip()

def build_index_from_pdfs() -> FAISS:
    """
    Read PDFs from DOCS_DIR, chunk per page and create Documents with metadata,
    then build and save FAISS index.
    """
    if not DOCS_DIR.exists():
        raise FileNotFoundError(f"Docs folder not found: {DOCS_DIR.resolve()}")

    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    docs: List[Document] = []

    pdf_paths = sorted(DOCS_DIR.glob("*.pdf"))
    if not pdf_paths:
        raise FileNotFoundError("No PDFs found in docs folder.")

    for pdf_path in pdf_paths:
        try:
            reader = PdfReader(str(pdf_path))
        except Exception as e:
            st.warning(f"Could not open {pdf_path.name}: {e}")
            continue

        for page_idx, page in enumerate(reader.pages, start=1):
            page_text = page.extract_text() or ""
            if not page_text.strip():
                continue
            chunks = splitter.split_text(page_text)
            for chunk_idx, c in enumerate(chunks):
                meta = {"source": pdf_path.name, "page": page_idx, "chunk": chunk_idx}
                docs.append(Document(page_content=c, metadata=meta))

    if not docs:
        raise ValueError("No text extracted from PDFs to index.")

    db = FAISS.from_documents(docs, embeddings)
    db.save_local(FAISS_DIR)

    # keep small sample so UI can show evidence that metadata exists
    st.session_state["index_sample"] = [
        {"source": d.metadata.get("source"), "page": d.metadata.get("page"), "chunk_len": len(d.page_content)}
        for d in docs[:3]
    ]
    return db

def load_or_build_index() -> FAISS:
    """Load FAISS index if present, otherwise build it. Rebuild if load fails."""
    if Path(FAISS_DIR).exists():
        try:
            db = FAISS.load_local(FAISS_DIR, embeddings, allow_dangerous_deserialization=True)
            return db
        except Exception:
            # corrupted or built with different embeddings -> delete & rebuild
            try:
                import shutil
                shutil.rmtree(FAISS_DIR)
            except Exception:
                pass
    return build_index_from_pdfs()

def retrieve_evidence(db: FAISS, query: str, k: int = TOP_K):
    docs = db.similarity_search(query, k=k)
    evidences = []
    for d in docs:
        src = d.metadata.get("source", "unknown")
        page = d.metadata.get("page", "?")
        text = d.page_content.strip()
        if len(text) > 800:
            text = text[:800].rsplit(" ", 1)[0] + "..."
        evidences.append({"text": text, "source": src, "page": page})
    return evidences, docs

# ---------------- Improved Prompts ----------------
INITIAL_PROMPT = (
    "You are a helpful assistant. Answer the following question briefly and clearly using your general knowledge. "
    "Provide a direct, natural response without any technical formatting or metadata. "
    "Keep your answer concise and informative.\n\n"
    "Question: {question}\n\n"
    "Answer:"
)

VERIFY_PROMPT = (
    "You are a helpful AI assistant. Based on the question and the evidence from the uploaded documents, "
    "provide a comprehensive and accurate answer.\n\n"
    "Question: {question}\n\n"
    "Evidence from Documents:\n{evidence_block}\n\n"
    "INSTRUCTIONS:\n"
    "- Use the evidence from the documents to provide a complete and accurate answer\n"
    "- Focus on what the documents actually say about the topic\n"
    "- Present the information in a clear, natural, conversational way\n"
    "- If the evidence doesn't fully answer the question, mention what information is available\n"
    "- Do not reference or compare with any previous answers\n"
    "- Write as if this is the first and only answer you're providing\n"
    "- Keep your response under 300 words\n\n"
    "Answer:"
)

def format_evidence_for_prompt(evidences: List[dict]) -> str:
    lines = []
    for i, e in enumerate(evidences, start=1):
        lines.append(f"Evidence {i} (from {e['source']}, page {e['page']}):\n{e['text']}")
    return "\n\n".join(lines)

# ---------------- Core QA flow ----------------
def answer_with_verification(db: FAISS, question: str) -> Tuple[str, str, List[str], List[dict]]:
    # 1) initial LLM knowledge-only response
    try:
        init_resp = llm.invoke(INITIAL_PROMPT.format(question=question))
        initial_answer = safe_text(init_resp).strip()
    except Exception as e:
        st.error(f"Error getting initial answer: {e}")
        initial_answer = "Unable to generate initial answer"

    # 2) retrieve evidence
    evidences, raw_docs = retrieve_evidence(db, question, k=TOP_K)

    # 3) verification step using evidence
    evidence_block = format_evidence_for_prompt(evidences) if evidences else "No relevant evidence found in the documents."
    verify_text = VERIFY_PROMPT.format(
        question=question, 
        evidence_block=evidence_block
    )
    
    try:
        verify_resp = llm.invoke(verify_text)
        final_answer = safe_text(verify_resp).strip()
    except Exception as e:
        st.error(f"Error getting final answer: {e}")
        final_answer = initial_answer or "Unable to generate answer"

    # 4) unique sources
    unique_sources = []
    for e in evidences:
        s = f"{e['source']} (page {e['page']})"
        if s not in unique_sources:
            unique_sources.append(s)

    return initial_answer, final_answer, unique_sources, evidences

# ---------------- Streamlit UI ----------------
def main():
    st.set_page_config(page_title="Multi-PDF Chat Agent", layout="wide")
    st.title("📚 Multi-PDF Chat Agent — Evidence-first answers")

    # Initialize session state for better UX
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Sidebar: upload & build
    with st.sidebar:
        st.header("📁 PDF Files Management")
        uploaded = st.file_uploader("Upload PDFs (multiple files supported)", accept_multiple_files=True, type=["pdf"])
        
        if st.button("🔨 Submit & Build Index", type="primary"):
            if not uploaded:
                st.warning("⚠️ Please upload at least one PDF file.")
            else:
                with st.spinner("📤 Uploading files and building index..."):
                    try:
                        DOCS_DIR.mkdir(parents=True, exist_ok=True)
                        
                        # Save uploaded files
                        for f in uploaded:
                            out_path = DOCS_DIR / f.name
                            with open(out_path, "wb") as out:
                                out.write(f.read())
                        
                        # Delete old index and rebuild
                        if Path(FAISS_DIR).exists():
                            import shutil
                            shutil.rmtree(FAISS_DIR)
                        
                        db = build_index_from_pdfs()
                        st.success("✅ Index built successfully!")
                        
                        # Show sample metadata
                        sample = st.session_state.get("index_sample", [])
                        if sample:
                            st.info("📊 Index created with sample chunks:")
                            for s in sample:
                                st.write(f"• {s['source']} (Page {s['page']}) - {s['chunk_len']} chars")
                                
                    except Exception as e:
                        st.error(f"❌ Index build failed: {e}")

        st.divider()
        st.caption(f"📍 Index location: `{FAISS_DIR}`")
        st.caption(f"📁 Documents folder: `{DOCS_DIR}`")

    # Main chat interface
    st.markdown("### 💬 Ask a Question")
    question = st.text_input(
        "Enter your question about the uploaded documents:",
        placeholder="e.g., What is the main topic discussed in the documents?",
        key="question_input"
    )
    
    if question and question.strip():
        with st.spinner("🤔 Thinking..."):
            try:
                # Load or build index
                db = load_or_build_index()
                
                # Get answer with verification
                initial_answer, final_answer, sources, evidences = answer_with_verification(db, question.strip())
                
                # Display the clean, natural answer
                st.markdown("### 🎯 Answer")
                display_answer = final_answer or initial_answer or "I couldn't generate an answer for your question."
                
                # Clean up any remaining formatting issues
                display_answer = display_answer.replace("Final Answer:", "").strip()
                display_answer = display_answer.replace("Answer:", "").strip()
                
                st.markdown(display_answer)
                
                # Show sources if available
                if sources:
                    st.markdown("### 📚 Sources")
                    for source in sources:
                        st.markdown(f"• {source}")
                
                # Expandable sections for additional info
                with st.expander("🔍 Show Evidence Details", expanded=False):
                    if evidences:
                        for i, e in enumerate(evidences, start=1):
                            st.markdown(f"**Evidence {i}** - {e['source']} (Page {e['page']})")
                            st.text_area(f"Content {i}", e["text"], height=100, key=f"evidence_{i}")
                    else:
                        st.info("No specific evidence chunks found in the documents.")
                
                with st.expander("🧠 Show Initial Knowledge-Based Answer", expanded=False):
                    if initial_answer and initial_answer != final_answer:
                        st.write(initial_answer)
                    else:
                        st.info("The initial and final answers were the same or no initial answer was generated.")
                        
            except FileNotFoundError as e:
                st.error(f"📂 {e}")
                st.info("Please upload some PDF files using the sidebar and build the index first.")
            except Exception as e:
                st.error(f"❌ An error occurred: {e}")
                st.info("Please try again or check your PDF files and API configuration.")

    # Help section
    with st.expander("❓ How to use this app", expanded=False):
        st.markdown("""
        1. **Upload PDFs**: Use the sidebar to upload one or more PDF files
        2. **Build Index**: Click "Submit & Build Index" to process your documents
        3. **Ask Questions**: Type your question in the text input above
        4. **Get Answers**: The app will provide evidence-based answers from your documents
        
        **Features:**
        - Natural, conversational responses
        - Evidence-based answers with source citations
        - Support for multiple PDF files
        - Clean, readable formatting
        """)

if __name__ == "__main__":
    main()