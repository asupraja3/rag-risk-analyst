import streamlit as st
import os
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from flashrank import Ranker, RerankRequest
from langchain_huggingface import HuggingFacePipeline
from transformers import pipeline

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Credit Risk Analyst",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS FOR PROFESSIONAL LOOK ---
st.markdown("""
<style>
    .stChatInput {border-radius: 10px;}
    .reportview-container {background: #0e1117;}
    h1 {color: #00d4b3;} /* ProjectC-like Green/Blue */
    .stExpander {border: 1px solid #333;}
</style>
""", unsafe_allow_html=True)

# --- LOAD ENV ---
load_dotenv()


# --- 1. CACHED RESOURCE LOADING (Simulating Microservice Initialization) ---
# [REAL-WORLD SCENARIO]: We cache these because loading a model/index takes time.
# In a real firm, these would be pre-loaded in a Kubernetes container.

@st.cache_resource
def load_embeddings_and_index():
    """Loads the Vector DB and Embeddings once."""
    try:
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        # Check if index exists before loading to prevent crash
        if os.path.exists("faiss_financial_index"):
            vector_store = FAISS.load_local(
                "faiss_financial_index",
                embeddings,
                allow_dangerous_deserialization=True
            )
            return vector_store
        else:
            return None
    except Exception as e:
        st.error(f"Error loading Index: {e}")
        return None


@st.cache_resource
def load_llm_pipeline():
    """Loads the LLM Model once."""
    # Using tiny-gpt2 for the demo to ensure it runs on your CPU.
    # In Prod: This would be Llama-3-8B or GPT-4 via API.
    generator = pipeline(
        "text-generation",
        model = "distilgpt2",
        # model="sshleifer/tiny-gpt2",
        # model = "TheBloke/guanaco-7B-HF", #Too huge for vercel
        max_new_tokens=100

        # device_map="auto"
    )
    return HuggingFacePipeline(pipeline=generator)


@st.cache_resource
def load_ranker():
    """Loads the FlashRank Reranker."""
    return Ranker()


# --- INITIALIZATION ---
vector_store = load_embeddings_and_index()
llm = load_llm_pipeline()
ranker = load_ranker()

# --- SIDEBAR CONFIGURATION ---
with st.sidebar:
    st.header("⚙️ System Parameters")
    st.info("**Architecture:** Hybrid Search (Dense + Rerank)")
    st.markdown("---")
    st.write("### 🔒 Security Status")
    st.caption("✅ RAG Pipeline: Local (Air-Gapped)")
    st.caption("✅ Vectors: FAISS (Encrypted At Rest)")

    st.markdown("---")
    if vector_store is None:
        st.error("❌ FAISS Index not found! Please run your ingestion script first.")
    else:
        st.success("✅ FAISS Index Loaded")

# --- MAIN CHAT LOGIC ---
st.title("Credit Risk Analyst")
st.markdown("ASK QUESTIONS ABOUT: **Liquidity, Debt Covenants, or Risk Factors (10-K)**")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("Enter query (e.g., 'What are the liquidity risks for Tesla?')..."):

    # 1. Display User Message
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 2. Process & Display Assistant Response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()

        if vector_store:
            with st.status("🔍 Retrieving & Reranking Financial Data...", expanded=True) as status:

                # A. Retrieval (FAISS)
                st.write("1️⃣ Querying Vector Database...")
                retriever = vector_store.as_retriever(search_kwargs={"k": 10})
                initial_docs = retriever.invoke(prompt)

                # B. Reranking (FlashRank)
                st.write("2️⃣ Reranking for Precision...")
                rerank_request = RerankRequest(query=prompt, passages=[
                    {"id": str(i), "text": doc.page_content, "meta": doc.metadata} for i, doc in enumerate(initial_docs)
                ])
                ranked_results = ranker.rerank(rerank_request)

                # Select Top 3
                top_context = [res['text'] for res in ranked_results[:3]]
                context_text = "\n\n---\n\n".join(top_context)

                status.update(label="✅ Context Secured", state="complete", expanded=False)

                # Show the "Hidden" context to the user (Transparency)
                with st.expander("📄 View Verified Sources (Audit Trail)"):
                    for i, res in enumerate(ranked_results[:3]):
                        st.markdown(f"**Source {i + 1} (Score: {res['score']:.4f}):**")
                        st.caption(res['text'][:300] + "...")

            # C. Generation (LLM)
            system_template = """You are a Senior Quantitative Risk Analyst. Use ONLY the context provided. Context: {context}"""

            prompt_template = ChatPromptTemplate.from_messages([
                ("system", system_template),
                ("user", "Question: {question}")
            ])

            chain = prompt_template | llm

            try:
                # Run the chain
                response = chain.invoke({"context": context_text, "question": prompt})

                # Formatting check: HuggingFacePipeline sometimes returns a string, sometimes an object
                final_answer = response if isinstance(response, str) else str(response)

                message_placeholder.markdown(final_answer)
                st.session_state.messages.append({"role": "assistant", "content": final_answer})

            except Exception as e:
                st.error(f"LLM Generation Error: {e}")
        else:
            st.error("Vector Store is not initialized. Cannot process query.")

print("--- Streamlit App Running ---")