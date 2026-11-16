
from rag import retrieve_documents_v2

from evaluation_dataset import EVALUATION_DATASET


def evaluate_retrieval():

    total = len(EVALUATION_DATASET)

    correct = 0

    for test in EVALUATION_DATASET:

        question = test["question"]

        expected_source = test["expected_source"]

        results = retrieve_documents_v2(
            question,
            top_k=3
        )

        retrieved_sources = [
            result["source"]
            for result in results
        ]

        passed = expected_source in retrieved_sources

        if passed:
            correct += 1

        print("\nQuestion:", question)

        print("Expected:", expected_source)

        print("Retrieved:", retrieved_sources)

        print("Result:", "PASS" if passed else "FAIL")

        print("-" * 50)

    accuracy = (correct / total) * 100

    print("\nRETRIEVAL EVALUATION")

    print("Total questions:", total)

    print("Correct retrievals:", correct)

    print("Retrieval accuracy:", round(accuracy, 2), "%")


if __name__ == "__main__":
    evaluate_retrieval()
