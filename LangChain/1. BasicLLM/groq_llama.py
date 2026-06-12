from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()
llm = ChatGroq(model="llama-3.1-8b-instant")
response = llm.invoke("Hello, explain yourself in one sentence.")
print(response.content)