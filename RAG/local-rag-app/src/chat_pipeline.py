from typing import List

from src.retriever import Retriever

REFUSAL_MESSAGE = "I cannot answer that based on the uploaded documents."
PROMPT_TEMPLATE = """Use ONLY the provided context to answer the question.
Do NOT use prior knowledge.
If unsure, say you cannot answer.

Context:
{context}

Conversation History:
{history}

Question:
{question}

Answer concisely and only using the context. If the answer cannot be found in the provided context, say: I cannot answer that based on the uploaded documents.
"""


def format_history(history: List[dict]) -> str:
    if not history:
        return "No prior conversation."
    formatted = []
    for turn in history:
        role = "User" if turn["role"] == "user" else "Assistant"
        formatted.append(f"{role}: {turn['message']}")
    return "\n".join(formatted)


class ChatPipeline:
    def __init__(self, llm, retriever: Retriever):
        self.llm = llm
        self.retriever = retriever

    def build_prompt(self, context: str, history: List[dict], question: str) -> str:
        history_text = format_history(history)
        return PROMPT_TEMPLATE.format(context=context, history=history_text, question=question)

    def run(self, query: str, chat_history: List[dict] = None) -> str:
        chat_history = chat_history or []
        relevant_docs = self.retriever.retrieve(query)
        if not relevant_docs:
            return REFUSAL_MESSAGE

        context = "\n\n".join([doc.page_content for doc in relevant_docs if doc.page_content.strip()])
        if not context.strip():
            return REFUSAL_MESSAGE

        prompt = self.build_prompt(context=context, history=chat_history, question=query)
        answer = self.llm.generate(prompt)
        if not answer:
            return REFUSAL_MESSAGE

        return answer.strip()
