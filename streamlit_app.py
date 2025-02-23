import streamlit as st
import pdfplumber
import httpx
import asyncio
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore

# --- Page Config ---
st.set_page_config(page_title="DeepSeek Companion", layout="wide", page_icon="📚")

# --- UI Styling ---
st.markdown(
    """
    <style>
        [data-testid="stSidebar"] { background-color: #1e1e2f; }
        .stChatMessage { border-radius: 10px; padding: 12px 16px; margin: 8px 0; }
        .stChatMessageUser { background-color: #4CAF50; color: white; }
        .stChatMessageAI { background-color: #303F9F; color: white; }
        .stButton>button { background-color: #FF6F61 !important; color: white !important; border-radius: 8px; }
        .stButton>button:hover { background-color: #FF3D33 !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- Title & Sidebar ---
st.title("DeepSeek Companion 📚")
st.caption("AI Pair Programmer with RAG-based Assistance")

with st.sidebar:
    st.header("🔍 Chat History")
    st.session_state.setdefault("chat_sessions", {})
    st.session_state.setdefault("current_session", None)
    
    if st.button("🌱 🆕 New Chat", use_container_width=True):
        st.session_state.current_session = None
        st.session_state.new_chat = True
        st.rerun()
    
    if st.session_state.chat_sessions:
        with st.expander("📜 View Chat History", expanded=False):
            for session_name in list(st.session_state.chat_sessions.keys()):
                if st.button(session_name, key=f"select_{session_name}"):
                    st.session_state.current_session = session_name
                    st.session_state.new_chat = False
                    st.rerun()

    st.header("⚙️ Configuration")
    try:
        response = httpx.get("http://localhost:11434/api/tags", timeout=3)
        available_models = [m["name"] for m in response.json().get("models", [])]
    except Exception:
        available_models = []
        st.error("⚠️ Failed to fetch models from Ollama. Ensure Ollama is running.")
        st.stop()
    
    selected_model = st.selectbox("Choose Model", available_models, index=0)
    uploaded_files = st.file_uploader("Upload documents", type=["pdf", "txt", "py", "md"], accept_multiple_files=True)

# --- Initialize LLM ---
llm_engine = ChatOllama(model=selected_model, base_url="http://localhost:11434", temperature=0.3)
embedding_model = OllamaEmbeddings(model=selected_model)
vector_store = InMemoryVectorStore(embedding_model)

# --- System Prompt ---
system_prompt = SystemMessagePromptTemplate.from_template(
    "You are an expert AI coding assistant. Provide concise, accurate solutions with debugging print statements where needed."
)

# --- Async File Processing ---
async def extract_text_from_file(file):
    """Extract text from supported files."""
    text = ""
    if file.type == "application/pdf":
        with pdfplumber.open(file) as pdf:
            text = "\n".join([page.extract_text() or "" for page in pdf.pages])
    else:
        text = file.getvalue().decode("utf-8")
    return text

async def process_uploaded_files(files):
    """Processes uploaded files asynchronously."""
    if not files:
        return
    
    tasks = [extract_text_from_file(file) for file in files]
    extracted_texts = await asyncio.gather(*tasks)
    full_text = "\n\n".join(extracted_texts)

    text_processor = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    documents = [Document(page_content=chunk) for chunk in text_processor.split_text(full_text)]
    vector_store.add_documents(documents)
    st.success("✅ Documents processed and indexed!")

if uploaded_files:
    asyncio.run(process_uploaded_files(uploaded_files))

# --- Optimized RAG Pipeline ---
def build_prompt_chain(user_query):
    """Creates a structured prompt chain incorporating retrieved documents."""
    prompt_sequence = [system_prompt]
    relevant_docs = vector_store.similarity_search(user_query, k=3)
    document_context = "\n\n".join([doc.page_content for doc in relevant_docs])
    context_prompt = f"Context:\n{document_context}\n\nQuery:\n{user_query}" if document_context else f"Query:\n{user_query}"
    prompt_sequence.append(HumanMessagePromptTemplate.from_template(context_prompt))
    return ChatPromptTemplate.from_messages(prompt_sequence)

async def generate_ai_response(user_query):
    """Generates response using LLM asynchronously."""
    prompt_chain = build_prompt_chain(user_query)
    chain = prompt_chain | llm_engine | StrOutputParser()
    return await asyncio.to_thread(chain.invoke, {"input": user_query})

# --- Display Chat ---
st.divider()

if st.session_state.current_session is None:
    st.session_state.new_chat = False
else:
    for message in st.session_state.chat_sessions[st.session_state.current_session]:
        chat_class = "stChatMessageAI" if message["role"] == "ai" else "stChatMessageUser"
        st.markdown(f'<div class="stChatMessage {chat_class}">{message["content"]}</div>', unsafe_allow_html=True)

# --- Process User Input ---
user_query = st.chat_input("Ask a question...")
if user_query:
    if st.session_state.current_session is None:
        session_name = user_query[:30] + "..." if len(user_query) > 30 else user_query
        st.session_state.current_session = session_name
        st.session_state.chat_sessions[session_name] = [{"role": "ai", "content": "Hello! How can I assist?"}]
    
    st.session_state.chat_sessions[st.session_state.current_session].append({"role": "user", "content": user_query})
    
    with st.spinner("🔍 Generating response..."):
        ai_response = asyncio.run(generate_ai_response(user_query))
    
    st.session_state.chat_sessions[st.session_state.current_session].append({"role": "ai", "content": ai_response})
    st.rerun()
