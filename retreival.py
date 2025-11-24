import os
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
# from langchain.prompts import ChatPromptTemplate
from langchain_core.prompts import ChatPromptTemplate
from flashrank import Ranker, RerankRequest
from langchain_huggingface import HuggingFacePipeline
from transformers import pipeline
# from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
# from langchain.callbacks import CallbackManager
# from langchain.callbacks.base import BaseCallbackHandler



# ### [REAL-WORLD SCENARIO]:
# In production, keys are injected via a Secrets Manager (like HashiCorp Vault),
# never hardcoded or simple env vars.
load_dotenv()
open_api_key = os.getenv("OPENAI_API_KEY")
# os.environ["OPENAI_API_KEY"] = open_api_key

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_store = FAISS.load_local("faiss_financial_index", embeddings, allow_dangerous_deserialization=True)

# ### [REAL-WORLD SCENARIO]:
# System Prompts in finance are version-controlled (Git for Prompts).
# We would pull this text from a "Prompt Registry" to ensure
# compliance has approved the wording.
system_template = """
You are a Senior Quantitative Risk Analyst.
Use ONLY the context provided.
Context: {context}
"""


def get_answer(query):
    # ### [REAL-WORLD SCENARIO]:
    # We would track the "Latency" of this retrieval step.
    # If FAISS takes >200ms, an alert fires to the engineering team.
    retriever = vector_store.as_retriever(search_kwargs={"k": 10})
    initial_docs = retriever.invoke(query)

    ranker = Ranker()
    rerank_request = RerankRequest(query=query, passages=[
        {"id": str(i), "text": doc.page_content, "meta": doc.metadata} for i, doc in enumerate(initial_docs)
    ])
    ranked_results = ranker.rerank(rerank_request)

    top_context = [res['text'] for res in ranked_results[:3]]
    context_text = "\n\n---\n\n".join(top_context)

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_template),
        ("user", "Question: {question}")
    ])

    # ### [REAL-WORLD SCENARIO]:
    # We would use a self-hosted model (like Llama 3) on internal servers.
    # We would also enable `streaming=True` to send the answer
    # via WebSockets token-by-token.
    # llm = ChatOpenAI(model="gpt-4", temperature=0.0)
    # llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.0)

    generator = pipeline(
        "text-generation",
        # model="NousResearch/Nous-Hermes-13b",
        # model="TheBloke/guanaco-7B-HF",
        model="sshleifer/tiny-gpt2",
        max_new_tokens=50,
        device_map="auto"
    )
    llm = HuggingFacePipeline(pipeline=generator)

    # llm = HuggingFacePipeline(
    #     pipeline=generator,
    #     callback_manager=StreamingStdOutCallbackHandler(),
    #     verbose=True
    # )

    chain = prompt | llm


    response = chain.invoke({"context": context_text, "question": query})

    # return response.content
    return response


if __name__ == "__main__":
    # ### [REAL-WORLD SCENARIO]:
    # This 'while' loop would be replaced by a FastAPI or WebSocket server.
    # app = FastAPI()
    # @app.websocket("/ws")
    # async def websocket_endpoint(websocket: WebSocket): ...

    print("--- Fortress Financial Chatbot Online ---")
    while True:
        user_input = input("Analyst Query: ")
        if user_input.lower() == "exit": break
        print(get_answer(user_input))