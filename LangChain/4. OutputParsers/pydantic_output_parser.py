from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()
model = ChatGroq(model="llama-3.3-70b-versatile")

class Person(BaseModel):
    name: str = Field(description="The person's full name")
    age: int = Field(gt=18, lt=60, description="The person's, age. Must not be greater than 18 and lower than 60")
    city: str = Field(description="The ciry where the person lives in")
    
parser = PydanticOutputParser(pydantic_object=Person)

template = PromptTemplate(
    template="Give me the name, age and city of a fictional {place} person. {format_instruction}",
    input_variables=['place'],
    partial_variables={"format_instruction": parser.get_format_instructions()}
)

# template2 = PromptTemplate(
#     template="Write a 5 line template on the following {text}",
#     input_variables=['text']
# )


chain = template | model | parser

response = chain.invoke({"place": "European"})
print(response)