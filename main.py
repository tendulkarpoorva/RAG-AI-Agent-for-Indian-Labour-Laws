from langchain_ollama.llms import OllamaLLM
from langchain.prompts import ChatPromptTemplate
from vector import main

retriever = main()
model = OllamaLLM(model="llama3.2")

template = """
You are an expert in answering Indian Labour Law related questions.

Here is some relevant context from legal documents:
{context}

Here is the question to answer: {question}
"""
prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

while True:
    print("\n\n-------------------------------")
    question = input("Ask your question (q to quit): ")
    print("\n\n")
    if question == "q":
        break

    docs = retriever.invoke(question)
    # Combine the content of retrieved documents for context
    context = "\n\n".join([doc.page_content for doc in docs])
    result = chain.invoke({"context": context, "question": question})
    print(result)