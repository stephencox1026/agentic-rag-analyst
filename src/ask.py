"""CLI: python -m src.ask "your question" """
import sys

from .rag import RagAnalyst


def main():
    if len(sys.argv) < 2:
        print('Usage: python -m src.ask "your question"')
        return
    question = " ".join(sys.argv[1:])
    analyst = RagAnalyst()
    ans = analyst.ask(question)
    print("\n" + ans.text + "\n")
    print("Sources: " + ", ".join(ans.sources))


if __name__ == "__main__":
    main()
