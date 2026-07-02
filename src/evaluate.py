"""Lightweight eval harness: retrieval hit-rate + LLM-as-judge faithfulness.

Reads data/eval/qa.jsonl with rows: {"question": ..., "expected_source": ..., "ideal": ...}
Usage: python -m src.evaluate
"""
import json
import pathlib

from openai import OpenAI

from . import config
from .rag import RagAnalyst

JUDGE_PROMPT = (
    "Score the answer's faithfulness to the context from 1 (unsupported) to 5 "
    "(fully supported). Reply with only the integer."
)


def judge(client: OpenAI, answer: str, context: str) -> int:
    resp = client.chat.completions.create(
        model=config.CHAT_MODEL,
        temperature=0,
        messages=[
            {"role": "system", "content": JUDGE_PROMPT},
            {"role": "user", "content": f"Context:\n{context}\n\nAnswer:\n{answer}"},
        ],
    )
    try:
        return int("".join(ch for ch in resp.choices[0].message.content if ch.isdigit())[:1])
    except (ValueError, IndexError):
        return 0


def main():
    path = pathlib.Path("data/eval/qa.jsonl")
    if not path.exists():
        print("No eval set at data/eval/qa.jsonl")
        return
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    analyst = RagAnalyst()
    client = OpenAI(api_key=config.OPENAI_API_KEY)

    hits, faith = 0, []
    for row in rows:
        ans = analyst.ask(row["question"])
        retrieved_sources = {s.split("#")[0] for s in ans.sources}
        if row.get("expected_source") in retrieved_sources:
            hits += 1
        faith.append(judge(client, ans.text, "\n\n".join(ans.contexts)))

    n = len(rows)
    print(f"Questions evaluated: {n}")
    print(f"Retrieval hit-rate:  {hits}/{n} = {hits / n:.0%}")
    print(f"Mean faithfulness:   {sum(faith) / n:.2f} / 5")


if __name__ == "__main__":
    main()
