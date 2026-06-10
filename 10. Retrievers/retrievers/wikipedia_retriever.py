import warnings
from bs4 import GuessedAtParserWarning

warnings.filterwarnings("ignore", category=GuessedAtParserWarning)

import wikipedia
from typing import List, Optional
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain_core.callbacks import CallbackManagerForRetrieverRun


class MyWikipediaRetriever(BaseRetriever):

    lang: str = "en"
    top_k_results: int = 3
    doc_content_chars_max: int = 4000
    user_agent: str = "MyLangChainTutorApp/1.0 (estudiante@ejemplo.com)"

    auto_suggest: bool = False

    def _get_relevant_documents(
            self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        wikipedia.set_user_agent(self.user_agent)
        wikipedia.set_lang(self.lang)
        docs = []

        try:
            search_results = wikipedia.search(query, results=self.top_k_results)

            for title in search_results:
                try:
                    page = wikipedia.page(title, self.auto_suggest)
                    content = page.content

                    docs.append(Document(
                        page_content=content,
                        metadata={
                            "title": page.url,
                            "source": page.url,
                            "summary": page.summary
                        }
                    ))
                except wikipedia.exceptions.DisambiguationError: 
                    continue
                except wikipedia.exceptions.PageError:
                    continue
        except Exception as e:
            print(f"Error when trying to connect with Wikipedia: {e}")

        return docs
