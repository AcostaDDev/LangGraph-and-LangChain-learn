import sys
import importlib.util

from typing import Type
from langchain_core.document_loaders import BaseLoader


def import_loader(mod_name: str) -> Type[BaseLoader]:
    
    file_path = f"6. DocumentLoaders/loaders/{mod_name}.py"

    spec = importlib.util.spec_from_file_location(mod_name, file_path)
    module = importlib.util.module_from_spec(spec)

    sys.modules[mod_name] = module
    spec.loader.exec_module(module)

    loader = None
    if mod_name == "pdf_loader":
        loader = module.MyPdfLoader
    elif mod_name == "text_loader":
        loader = module.MyTextLoader
    elif mod_name == "web_loader":
        loader = module.MyWebLoader
    elif mod_name == "directory_loader":
        loader = module.MyDirectoryLoader
    elif mod_name == "csv_loader":
        loader = module.MyCSVLoader

    return loader
