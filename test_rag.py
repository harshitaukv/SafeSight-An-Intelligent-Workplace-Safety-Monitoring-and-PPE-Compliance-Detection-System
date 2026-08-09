from rag_pipeline import ask_question

while True:

    question = input("\nAsk: ")

    if question.lower() == "exit":
        break

    answer = ask_question(question)

    print("\nAnswer:\n")
    print(answer)