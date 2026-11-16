
from rag import (
    index_documents_v2,
    retrieve_documents_v2
)

index_documents_v2()

question = "Which project has a swimming pool?"

results = retrieve_documents_v2(question)

print("\nRETRIEVED DOCUMENTS\n")

for result in results:

    print("Project:", result["project_name"])
    print("Source:", result["source"])
    print("Page:", result["page_number"])
    print("Similarity:", round(result["similarity"], 3))
    print("Information:", result["text"])

    print("-" * 50)
