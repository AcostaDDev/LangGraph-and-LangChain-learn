from loaders.csv_loader import MyCSVLoader

loader = MyCSVLoader(file_path="6. DocumentLoaders/documents/example.csv")

data=loader.load()

print(data[0])