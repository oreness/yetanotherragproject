# yetanotherragproject

Two PDF tools built with LangChain, Groq, and Streamlit.

## Tools

**Chat** — Upload a PDF and ask questions about it. Uses RAG (FAISS + sentence-transformers) to retrieve relevant context before answering.

**Review** — Upload a document and get a structured analysis: content and format scores, strengths, and concrete improvement suggestions. Includes a follow-up chat to ask questions about the analysis.

## Stack

- [LangChain](https://github.com/langchain-ai/langchain) — orchestration
- [Groq](https://groq.com) — LLM inference (LLaMA 3.3 70B)
- [FAISS](https://github.com/facebookresearch/faiss) — vector store
- [sentence-transformers](https://www.sbert.net) — embeddings (all-MiniLM-L6-v2)
- [Streamlit](https://streamlit.io) — UI

## Setup

```bash
git clone https://github.com/oreness/yetanotherragproject.git
cd yetanotherragproject
pip install -r requirements.txt
```

Create a `.env` file in the root:

```
GROQ_API_KEY=your_key_here
```

Get a free API key at [console.groq.com](https://console.groq.com).

```bash
streamlit run app.py
```

## Notes

- The Chat tab indexes the full PDF into chunks and retrieves the 4 most relevant ones per query.
- The Review tab sends up to 8000 characters to the LLM for analysis and returns structured JSON.
