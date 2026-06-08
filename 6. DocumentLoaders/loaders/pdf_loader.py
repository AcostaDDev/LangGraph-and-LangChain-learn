from typing import Iterator
from pypdf import PdfReader
from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document


class MyPdfLoader(BaseLoader):
    def __init__(self, file_path):
        self.file_path = file_path

    def lazy_load(self) -> Iterator[Document]:
        reader = PdfReader(self.file_path)
        total_pages = len(reader.pages)

        pdf_meta = reader.metadata or {}

        producer = pdf_meta.get("/Producer", "")
        creator = pdf_meta.get("/Creator", "")
        creation_date = pdf_meta.get("/CreationDate", "")
        title = pdf_meta.get("/Title", "")

        for n_page, page in enumerate(reader.pages):
            text = page.extract_text()
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