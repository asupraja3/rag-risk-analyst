# Rag-Risk-Analyst: Institutional RAG System

## Project Overview
Rag-Risk-Analyst is a specialized Retrieval-Augmented Generation (RAG) chatbot designed for the **Credit & Risk Domain**. It ingests high-density financial filings (SEC 10-Ks), indexes them locally, and performs precision-based retrieval to answer questions about corporate debt, liquidity, and risk factors.

## Architecture Flow

```mermaid
graph TD
    A[User Query] --> B[Embedding Model]
    B --> C[FAISS Vector Search]
    C -->|Top 10 Matches| D[FlashRank Reranker]
    D -->|Top 3 Verified Contexts| E[LLM - GPT-4]
    E -->|System Prompt Enforcement| F[Final Analyst Report]
