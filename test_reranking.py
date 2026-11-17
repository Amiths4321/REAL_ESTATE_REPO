
from rag import hybrid_search, rerank_documents

question = "Does Skyline project have covered parking?"

documents = hybrid_search(question, top_k=10)

print("\nBEFORE RERANKING")
for doc in documents:
    print(doc.get("source"), doc.get("text", "")[:150])

reranked = rerank_documents(
    question,
    documents,
    top_k=5
)

print("\nAFTER RERANKING")
for doc in reranked:
    print(doc.get("source"), doc.get("text", "")[:150])
