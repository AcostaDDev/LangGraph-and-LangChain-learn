import csv

from typing import Iterator, Optional

from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document

class MyCSVLoader(BaseLoader):
    def __init__(
            self,
            file_path: str,
            source_column: Optional[str] = None,
            encoding: str = "utf-8"
    ) -> None:
        self.file_path = file_path
        self.source_column = source_column
        self.encoding = encoding

    def lazy_load(self) -> Iterator[Document]:
        with open(self.file_path, mode="r", encoding=self.encoding) as csv_file:
            reader = csv.DictReader(csv_file)

            for index, row in enumerate(reader):
                text_content = "\n".join(
                    f"{key.strip()}: {value.strip()}"
                    for key, value in row.items()
                    if key and value
                )

                source = self.file_path
                if self.source_column and self.source_column in row:
                    source = row[self.source_column]
                
                yield Document(
                    page_content=text_content,
                    metadata={
                        "source": source,
                        "row": index
                    }
                )