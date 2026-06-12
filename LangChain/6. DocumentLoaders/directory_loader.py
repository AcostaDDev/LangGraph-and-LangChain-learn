from loaders.directory_loader import MyDirectoryLoader
from loaders.pdf_loader import MyPdfLoader
from loaders.text_loader import MyTextLoader
from loaders.csv_loader import MyCSVLoader

loader = MyDirectoryLoader(
    path="6. DocumentLoaders/documents",
    loaders={
        ".pdf": MyPdfLoader,
        ".txt": MyTextLoader,
        ".csv": MyCSVLoader,
    }
)

# docs = loader.load()
# print(len(docs))
# print(docs[0].metadata)
# print(docs[1].metadata)

docs = loader.lazy_load()

for document in docs:
    print(document.metadata.get("page") if document.metadata.get("page") != '' else f"The document {document.metadata.get('source')} has no page")