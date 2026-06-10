import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from langchain_text_splitters import RecursiveCharacterTextSplitter

from utils.import_loader import import_loader

MyPdfLoader = import_loader("pdf_loader")
loader = MyPdfLoader("7. TextSplitters/documents/example.pdf")

docs = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=20,
)

result = splitter.split_documents(docs)

print("Chunks len:")
print(len(result))
print("\nChunk content:")
print(result[10].page_content)
print("\nChunk metadata:")
print(result[10].metadata)