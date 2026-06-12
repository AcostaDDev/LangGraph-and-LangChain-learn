import fitz
from typing import Iterator
from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document


class MyPdfLoader(BaseLoader):
    def __init__(self, file_path):
        self.file_path = file_path

    def lazy_load(self) -> Iterator[Document]:
        doc = fitz.open(self.file_path)
        total_pages = doc.page_count

        pdf_meta = doc.metadata or {}

        producer = pdf_meta.get("producer", "")
        creator = pdf_meta.get("creator", "")
        creation_date = pdf_meta.get("creationDate", "")
        title = pdf_meta.get("title", "")

        for n_page, page in enumerate(doc):
            text = page.get_text()
            if text.strip():
                yield Document(
                    page_content=text,
                    metadata={
                        'producer': producer,
                        'creator': creator,
                        'creationdate': creation_date,
                        'title': title,
                        'source': self.file_path,
                        'total_pages': total_pages,
                        'page': n_page,
                        'page_label': str(n_page + 1)
                    }
                )
        doc.close()