"""Entry point: ask the RAG pipeline the critical phone-number question."""

from rag import RagPipeline

QUESTIONS = [
    "What exact phone number should I call for the University IT Service Desk?",
    "How do I reset my university password?",
    "My VPN client keeps failing. What should I do?",
]


def main():
    rag = RagPipeline()
    for q in QUESTIONS:
        print("\n" + "=" * 60)
        print("QUESTION:", q)
        print("-" * 60)
        answer = rag.answer(q)
        print("ANSWER:", answer)


if __name__ == "__main__":
    main()