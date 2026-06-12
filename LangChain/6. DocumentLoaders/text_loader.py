from loaders.text_loader import MyTextLoader


loader = MyTextLoader("LangChain/6. DocumentLoaders/documents/cricket.txt")

docs = loader.load()

# print(docs)
# print(type(docs))
# print(docs[0])

print(docs[0].page_content)