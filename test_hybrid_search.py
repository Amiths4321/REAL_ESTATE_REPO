from rag import setup_keyword_search, index_keyword_documents, keyword_search

setup_keyword_search()
index_keyword_documents()

results = keyword_search("Powai swimming pool parking")

for result in results:
    print(result["source"])
    print(result["text"])