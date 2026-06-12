import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from typing import List
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

from utils.import_loader import import_loader

load_dotenv()
DOC_FOLDER = "11. RAG/documents/"

MyPDFLoader = import_loader("pdf_loader")
MyDirectoryLoader = import_loader("directory_loader")

loader = MyDirectoryLoader(
    path=DOC_FOLDER,
    loaders = {
        ".pdf": MyPDFLoader
    }
)

all_documents: List[Document] = loader.load()
origins = set([d.metadata.get('source', '').replace(DOC_FOLDER, '') for d in all_documents])
print("Origins: ", origins)


splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100
)

chunks = splitter.split_documents(all_documents)

print("Total chunks created: ", len(chunks))

embedding_model = HuggingFaceEmbeddings(
    model="sentence-transformers/all-miniLM-L6-v2"
)

vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embedding_model,
    collection_name="multi_doc_rag"
)

retriever = vector_store.as_retriever(
    search_kwargs={
        "k": 5
    }
)

llm = ChatGroq(
    model="llama-3.3-70b-versatile"
)
parser = StrOutputParser()

prompt = ChatPromptTemplate.from_messages([
    SystemMessage(
        content=(
            """
            You are an expert assistant.
            Answer only using the provided context.
            If multiple documents are involved, clearly mention them.
            If the answer is not present in any document, just say that you don't know
            """
        )
    ),
    (
        "human",
        """
        Context: {context}.
        Question: {question}
        """
    )
])

def multi_document_rag(query: str) -> None:
    docs = retriever.invoke(query)

    context = "\n\n".join(
        f"""
        [Document: {d.metadata.get('source', '').replace(DOC_FOLDER, '')}] | [Page: {d.metadata.get('page')}]
        {d.page_content}
        """
        for d in docs
    )

    messages = {
        "context": context,
        "question": query
    }

    chain = prompt | llm | parser
    return chain.invoke(messages), docs

print("MULTI DOCUMENT RAG, type 'exit' to quit")
while True:
    query = input("You: ")
    if query.lower() == "exit":
        break

    answer, sources = multi_document_rag(query)
    print(f"\nAI Answer: {answer}")

    print("\nDocuments retrieved:")
    for i, d in enumerate(sources):
        print(f"{i+1}. {d.metadata.get('source', '').replace(DOC_FOLDER, '')}")