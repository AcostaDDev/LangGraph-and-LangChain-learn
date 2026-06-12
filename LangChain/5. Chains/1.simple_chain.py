from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()
model = ChatGroq(model="llama-3.3-70b-versatile")

prompt = PromptTemplate(
    template="generate 3 facts about a topic: {topic}",
    input_variables=['topic']
)

parser = StrOutputParser()

chain = prompt | model | parser

response = chain.invoke({"topic": "Aliens"})
print(response)