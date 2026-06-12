from loaders.pdf_loader import MyPdfLoader

loader = MyPdfLoader("LangChain/6. DocumentLoaders/documents/tfm-david.pdf")
docs = loader.load()

# print(docs[0])
print(docs[0].metadata)