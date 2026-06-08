from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document


class MyTextLoader(BaseLoader):
    def __init__(self, file_path):
        self.file_path = file_path

    def lazy_load(self):
        with open(self.file_path, "r", encoding='utf-8') as f:
            text = f.read()
        
        yield Document(
            page_content=text,
            metadata={
                'producer': '',
                'creator': '',
                'creationdate': '',
                'title': '',
                'source': self.file_path,
                'total_pages': 1,
                'page': 0,
                'page_label': '1'
            }
        )