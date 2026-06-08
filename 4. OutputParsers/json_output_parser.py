from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from dotenv import load_dotenv

load_dotenv()
model = ChatGroq(model="llama-3.3-70b-versatile")
parser = JsonOutputParser()

template = PromptTemplate(
    template="Give me the name, age and city of a fictional person. The name and the city has to be spanish and different from Sofía. {format_instruction}",
    input_variables=[],
    partial_variables={"format_instruction": parser.get_format_instructions()}
)

# template2 = PromptTemplate(
#     template="Write a 5 line template on the following {text}",
#     input_variables=['text']
# )


chain = template | model | parser

response = chain.invoke({})
print(response)