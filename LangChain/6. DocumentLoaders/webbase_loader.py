from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from dotenv import load_dotenv

from loaders.web_loader import MyWebLoader


load_dotenv()
model = ChatGroq(model="llama-3.3-70b-versatile")

prompt = PromptTemplate(
    template="Answer the following question: {question}, from the following text: {text}",
    input_variables=['question', 'text']
)

url = "https://techlekh.com/dongfeng-nammi-vigo-price-nepal/"

loader = MyWebLoader(url=url)
docs = loader.load()

parser = StrOutputParser()

chain = prompt | model | parser

result = chain.invoke({'question': 'what is the article talking about?', 'text': docs[0].page_content})
print(result)