"""Retrieval + grounded generation with citations."""
from dataclasses import dataclass

import chromadb
from openai import OpenAI

from . import config

SYSTEM_PROMPT = (
    "You are a precise analyst. Answer ONLY from the provided context. "
    "Cite sources inline using [source] tags that match the context labels. "
    "If the context does not contain the answer, say so explicitly."
)


@dataclass
class Answer:
    text: str
    sources: list
    contexts: list


class RagAnalyst:
    def __init__(self):
        self.client = OpenAI(api_key=config.OPENAI_API_KEY)
        self.coll = chromadb.PersistentClient(path=config.CHROMA_DIR).get_collection(
            config.COLLECTION
        )

    def retrieve(self, question: str, k: int = config.TOP_K):
        qvec = self.client.embeddings.create(
            model=config.EMBEDDING_MODEL, input=[question]
        ).data[0].embedding
        res = self.coll.query(query_embeddings=[qvec], n_results=k)
        docs = res["documents"][0]
        metas = res["metadatas"][0]
        return list(zip(docs, metas))

    def ask(self, question: str, k: int = config.TOP_K) -> Answer:
        hits = self.retrieve(question, k)
        context_blocks = []
        sources = []
        for doc, meta in hits:
            label = f"{meta['source']}#{meta['chunk']}"
            sources.append(label)
            context_blocks.append(f"[{label}]\n{doc}")
        context = "\n\n".join(context_blocks)

        resp = self.client.chat.completions.create(
            model=config.CHAT_MODEL,
            temperature=0.1,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": f"Context:\n{context}\n\nQuestion: {question}",
                },
            ],
        )
        return Answer(
            text=resp.choices[0].message.content,
            sources=sources,
            contexts=[d for d, _ in hits],
        )
