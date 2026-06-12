from pathlib import Path
from typing import Iterator, Type, Dict

from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document

class MyDirectoryLoader(BaseLoader):
    def __init__(
            self,
            path: str,
            loaders: Dict[str, Type[BaseLoader]],
            recursive: bool=True
    ) -> None:
        self.path = Path(path)
        self.loaders = {ext.lower(): cls for ext, cls in loaders.items()}
        self.recursive = recursive
        
    def lazy_load(self) -> Iterator[Document]:
        files = self.path.rglob("*") if self.recursive else self.path.iterdir()

        for file_path in files:
            if not file_path.is_file():
                continue

            ext = file_path.suffix.lower()
            if not ext in self.loaders:
                continue
            
            loader_cls = self.loaders[ext]
            try:
                loader = loader_cls(str(file_path))
                yield from loader.lazy_load()

            except Exception as e:
                print(f"Error processing {file_path.name}: ", e)