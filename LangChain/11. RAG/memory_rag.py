import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from typing import List
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from utils.import_loader import import_loader

load_dotenv()

MyPdfLoader = import_loader("pdf_loader")

loader = MyPdfLoader("LangChain/11. RAG/documents/tfm-david.pdf")
documents = loader.load()

llm = ChatGroq(model="llama-3.3-70b-versatile")

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500,
    chunk_overlap=150
)

chunks = splitter.split_documents(documents)

embedding_model = HuggingFaceEmbeddings(
    model="sentence-transformers/all-miniLM-L6-v2"
)

vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embedding_model
)

retriever = vector_store.as_retriever(
    search_kwargs={
        "k": 5
    }
)


system_message = """
You are a helpful AI Assistant.
Answer strictly from the provided context or with the chat history.
If the answer is not present in the context, just say you don't know
"""

prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessage(
            content=system_message
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        (
            "human", "Context: \n {context}.\n\nQuestion:\n{input}"
        )
    ]   
)

parser = StrOutputParser()

def conversational_rag(user_input: str, chat_history: List[BaseMessage]):
    docs = retriever.invoke(user_input)
    context = "\n\n".join(
        f"[Page {d.metadata.get('page', 'N/A')}]\n{d.page_content}" for d in docs
    )

    messages = {
        "input": user_input,
        "context": context,
        "chat_history": chat_history
    }
    

    chain = prompt | llm | parser
    ai_answer = chain.invoke(messages)

    return ai_answer, docs

chat_history: List[BaseMessage] = []

print("Conversational RAG")

while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        break

    ai_answer, sources = conversational_rag(user_input, chat_history)
    chat_history.append(HumanMessage(content=user_input))
    chat_history.append(AIMessage(content=ai_answer))

    print(f"\nAI: {ai_answer}")

