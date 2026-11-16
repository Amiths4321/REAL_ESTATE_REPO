
from rag import retrieve_documents_v2


question = "Which amenities are available?"


print("\nSEARCHING ALL PROJECTS")

results = retrieve_documents_v2(
    question,
    top_k=5,
    min_similarity=0.30
)

for result in results:

    print("\nProject:", result["project_name"])
    print("Source:", result["source"])
    print("Similarity:", round(result["similarity"], 3))
    print("Text:", result["text"])


print("\nSEARCHING ONLY PROJECT A")

results = retrieve_documents_v2(
    question,
    top_k=5,
    min_similarity=0.30,
    project_name="project_a"
)

for result in results:

    print("\nProject:", result["project_name"])
    print("Similarity:", round(result["similarity"], 3))
    print("Text:", result["text"])


print("\nTESTING AN UNRELATED QUESTION")

results = retrieve_documents_v2(
    "What is the recipe for making pizza?",
    top_k=3,
    min_similarity=0.30
)

if not results:
    print("No sufficiently relevant documents found.")

else:
    print("Retrieved:", len(results), "documents")
