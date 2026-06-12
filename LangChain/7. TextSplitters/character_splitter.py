import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from langchain_text_splitters import CharacterTextSplitter

from utils.import_loader import import_loader

MyPdfLoader = import_loader("pdf_loader")
loader = MyPdfLoader("LangChain/7. TextSplitters/documents/example.pdf")

docs = loader.load()

splitter = CharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separator=""
)

result = splitter.split_documents(docs)

print(result[50].page_content)
print(result[50].metadata)