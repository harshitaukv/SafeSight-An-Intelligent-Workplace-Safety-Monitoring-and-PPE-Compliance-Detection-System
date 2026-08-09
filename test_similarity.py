from similarity_search import search

query = input("Ask a question: ")

results = search(query)

print("\nTop Results\n")

for i, result in enumerate(results, 1):

    print(f"Result {i}")
    print(f"Similarity Score : {result['score']:.4f}")
    print(result["document"]["text"])
    print("-" * 50)