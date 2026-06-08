import requests
from bs4 import BeautifulSoup
from typing import Iterator, Optional
from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document


class MyWebLoader(BaseLoader):
    def __init__(
            self,
            url: str,
            timeout: Optional[int] = 10
    ) -> None:
        self.url = url
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }
        self.timeout = timeout

    def lazy_load(self):
        try:
            response = requests.get(self.url, headers=self.headers, timeout=self.timeout)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, 'html.parser')
            for script in soup(["script", "style", "noscript"]):
                script.extract()

            title = soup.title.string.strip() if soup.title else "No Title"
            clean_text = soup.get_text(separator="\n", strip=True)

            yield Document(
                page_content=clean_text,
                metadata={
                    "source": self.url,
                    "title": title,
                    "language": soup.html.get("lang", "unknown") if soup.html else "unknown"
                }
            )

        except requests.exceptions.RequestException as e:
            print(f"Error when trying to access to {self.url}: {e}")