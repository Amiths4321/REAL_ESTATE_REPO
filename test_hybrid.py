
from rag import hybrid_search

question = "Swimming pool and covered parking in Powai"

results = hybrid_search(question, top_k=5)

for result in results:

    print("\nSource:", result["source"])
    print("Score:", result["hybrid_score"])
    print("Content:", result["text"][:300])
