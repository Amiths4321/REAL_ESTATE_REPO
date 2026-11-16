
from rag import retrieve_documents_v2
from app import ask_qwen

from evaluation_dataset import EVALUATION_DATASET

import json


def evaluate_answers():

    for test in EVALUATION_DATASET:

        question = test["question"]

        expected_keywords = test["expected_keywords"]

        documents = retrieve_documents_v2(
            question,
            top_k=3
        )

        context = [
            {
                "source": doc["source"],
                "text": doc["text"]
            }
            for doc in documents
        ]

        prompt = f"""
Answer this question using only the supplied documents.

Question:
{question}

Documents:
{json.dumps(context)}

Do not invent information.
"""

        answer = ask_qwen([
            {
                "role": "user",
                "content": prompt
            }
        ])

        answer_lower = answer.lower()

        matched = [
            keyword
            for keyword in expected_keywords
            if keyword.lower() in answer_lower
        ]

        print("\nQuestion:", question)

        print("AI Answer:", answer)

        print("Expected keywords:", expected_keywords)

        print("Matched keywords:", matched)

        if len(matched) == len(expected_keywords):

            print("Result: PASS")

        else:

            print("Result: REVIEW REQUIRED")

        print("-" * 60)


if __name__ == "__main__":
    evaluate_answers()
