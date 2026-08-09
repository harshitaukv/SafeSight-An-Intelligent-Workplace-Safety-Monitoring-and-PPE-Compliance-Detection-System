from data_processing import prepare_rag_data

docs = prepare_rag_data()

print("Total Documents:", len(docs))

for d in docs[:5]:
    print("=" * 50)
    print(d["type"])
    print(d["text"][:250])