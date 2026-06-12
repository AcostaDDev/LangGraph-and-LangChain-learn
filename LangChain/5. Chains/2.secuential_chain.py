from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv


load_dotenv()
model = ChatGroq(model="llama-3.3-70b-versatile")

prompt1 = PromptTemplate(
    template="Generate detailed report on {topic}",
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template="Generatea 3 point summary of the following text: {text}",
    input_variables=['text']
)

parser = StrOutputParser()

chain = prompt1 | model | parser | prompt2 | model | parser

response = chain.invoke({"topic": "Aliens"})
print(response)