"""RAG pipeline: retrieve context, build a strict prompt, call Qwen3-0.6B via Ollama."""

from langchain_ollama import ChatOllama

from rag_tool import RagTool

LLM_MODEL = "qwen3:0.6b"
OLLAMA_URL = "http://127.0.0.1:11434"

SYSTEM_PROMPT = """You are a university IT support assistant.
Answer ONLY using the context below.
If the answer is not in the context, you MUST say you do not know.
Do NOT invent phone numbers, emails, URLs, or any other facts.
Do NOT give instructions not supported by the context.
Keep the answer short and direct.

Context:
{context}
"""


class RagPipeline:
    def __init__(self):
        self.retriever = RagTool()
        self.llm = ChatOllama(
            model=LLM_MODEL,
            temperature=0.0,
            base_url=OLLAMA_URL,
        )

    def answer(self, question):
        docs = self.retriever.retrieve(question)
        context = "\n\n---\n\n".join(
            f"[{d.metadata.get('policy_id', '?')} | {d.metadata.get('topic', '?')}]\n{d.page_content}"
            for d in docs
        )
        prompt = SYSTEM_PROMPT.format(context=context) + f"\n\nUser question: {question}\nAnswer:"
        response = self.llm.invoke(prompt)
        return response.content if hasattr(response, "content") else str(response)