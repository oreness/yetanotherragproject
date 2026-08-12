import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import tempfile
import os
import json
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="yetanotherragproject", layout="wide")

st.markdown("""
<style>
    .block-container { padding-top: 2rem; }
    .stTabs [data-baseweb="tab"] { font-size: 14px; }
    .metric-bar { height: 6px; border-radius: 3px; margin-top: 4px; margin-bottom: 12px; }
    h1 { font-weight: 400; letter-spacing: -1px; }
    .stAlert { border-radius: 6px; }
</style>
""", unsafe_allow_html=True)

st.title("yetanotherragproject")
st.caption("PDF tools powered by Groq + LangChain")

tab1, tab2 = st.tabs(["Chat", "Review"])

# ── helpers ──────────────────────────────────────────────────────────────────

@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def get_llm():
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=os.getenv("GROQ_API_KEY")
    )

def load_pdf(uploaded):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
        f.write(uploaded.read())
        return f.name

def score_color(score):
    if score >= 7:
        return "#4ade80"
    elif score >= 5:
        return "#facc15"
    return "#f87171"

# ── TAB 1: CHAT ───────────────────────────────────────────────────────────────

with tab1:
    st.markdown("#### Chat with your PDF")
    st.caption("Upload a document and ask anything about it.")

    uploaded_chat = st.file_uploader("Upload PDF", type="pdf", key="chat_upload")

    if uploaded_chat:
        embeddings = load_embeddings()
        tmp = load_pdf(uploaded_chat)

        with st.spinner("Processing..."):
            loader = PyPDFLoader(tmp)
            docs = loader.load()
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            chunks = splitter.split_documents(docs)
            vectorstore = FAISS.from_documents(chunks, embeddings)
            retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
            llm = get_llm()

            prompt = ChatPromptTemplate.from_template("""
Answer the question based only on the following context:

{context}

Question: {question}

Answer:""")

            chain = (
                {"context": retriever, "question": RunnablePassthrough()}
                | prompt
                | llm
                | StrOutputParser()
            )

        os.unlink(tmp)
        st.caption(f"{len(chunks)} chunks indexed")

        if "chat_messages" not in st.session_state:
            st.session_state.chat_messages = []

        chat_container = st.container(height=400)
        with chat_container:
            for msg in st.session_state.chat_messages:
                with st.chat_message(msg["role"]):
                    st.write(msg["content"])

        if user_input := st.chat_input("Ask something...", key="chat_input"):
            st.session_state.chat_messages.append({"role": "user", "content": user_input})
            with chat_container:
                with st.chat_message("user"):
                    st.write(user_input)
                with st.chat_message("assistant"):
                    with st.spinner(""):
                        answer = chain.invoke(user_input)
                    st.write(answer)
            st.session_state.chat_messages.append({"role": "assistant", "content": answer})

# ── TAB 2: REVIEW ─────────────────────────────────────────────────────────────

REVIEW_PROMPT = """You are an expert in scientific writing and academic document analysis.
Analyze the following document and evaluate it across two dimensions:

CONTENT (0-10):
- Coherence: ideas are well connected
- Relevance: important points receive appropriate weight
- Conciseness: no unnecessarily over-developed sections
- Repetition: identifies unnecessarily repeated ideas

FORMAT (0-10):
- Citations and references: correctly formatted and consistent
- Punctuation and formal register
- Figure references: coherent with the text
- Paragraph order and flow
- Logical conclusion
- Structure: the document has a clear and logical organization

Return the result in this exact JSON format, no additional text, no markdown:
{{
    "score_content": X,
    "score_format": X,
    "score_global": X,
    "strengths": ["...", "...", "..."],
    "improvements": ["...", "...", "..."]
}}

DOCUMENT:
{doc_text}"""

with tab2:
    st.markdown("#### Document Review")
    st.caption("Upload a document and get a structured analysis.")

    uploaded_review = st.file_uploader("Upload PDF", type="pdf", key="review_upload")

    if uploaded_review:
        tmp = load_pdf(uploaded_review)

        with st.spinner("Analyzing..."):
            loader = PyPDFLoader(tmp)
            docs = loader.load()
            doc_text = "\n".join([d.page_content for d in docs])
            os.unlink(tmp)

            llm = get_llm()
            prompt = ChatPromptTemplate.from_template(REVIEW_PROMPT)
            chain = prompt | llm | StrOutputParser()
            result = chain.invoke({"doc_text": doc_text[:8000]})

        try:
            clean = result.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
            data = json.loads(clean)

            st.session_state.doc_text = doc_text
            st.session_state.review_data = data

            if "review_messages" not in st.session_state:
                st.session_state.review_messages = []

            col_left, col_right = st.columns([1, 1])

            with col_left:
                c1, c2, c3 = st.columns(3)
                for col, label, key in [
                    (c1, "Content", "score_content"),
                    (c2, "Format", "score_format"),
                    (c3, "Global", "score_global")
                ]:
                    with col:
                        score = data[key]
                        st.metric(label, f"{score}/10")
                        st.markdown(
                            f'<div class="metric-bar" style="background:{score_color(score)};width:{score*10}%"></div>',
                            unsafe_allow_html=True
                        )

                st.divider()

                st.markdown("**Strengths**")
                for s in data["strengths"]:
                    st.markdown(f"- {s}")

                st.divider()

                st.markdown("**Improvements**")
                for i in data["improvements"]:
                    st.markdown(f"- {i}")

            with col_right:
                st.markdown("**Questions about the analysis**")

                chat_container = st.container(height=450)
                with chat_container:
                    for msg in st.session_state.review_messages:
                        with st.chat_message(msg["role"]):
                            st.write(msg["content"])

                if question := st.chat_input("Ask something...", key="review_input"):
                    st.session_state.review_messages.append({"role": "user", "content": question})

                    analisis = st.session_state.review_data
                    texto = st.session_state.doc_text

                    chat_prompt = f"""You are an expert in scientific writing. You have analyzed the following document:

DOCUMENT (excerpt):
{texto[:4000]}

YOUR ANALYSIS:
- Content score: {analisis['score_content']}/10
- Format score: {analisis['score_format']}/10
- Global score: {analisis['score_global']}/10

Strengths:
{chr(10).join(f"- {s}" for s in analisis['strengths'])}

Improvements:
{chr(10).join(f"- {i}" for i in analisis['improvements'])}

Answer the user's question clearly and concretely, quoting the document when relevant.

Question: {question}"""

                    with chat_container:
                        with st.chat_message("assistant"):
                            with st.spinner(""):
                                response = llm.invoke(chat_prompt)
                                answer = response.content
                            st.write(answer)

                    st.session_state.review_messages.append({"role": "assistant", "content": answer})

        except json.JSONDecodeError:
            st.error("Could not parse the response. Try again.")
            st.code(result)