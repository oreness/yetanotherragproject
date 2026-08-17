CHAT_PROMPT = """Answer the question based only on the following context:

{context}

Question: {question}

Answer:"""

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