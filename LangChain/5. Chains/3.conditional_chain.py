from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain_core.runnables import RunnableBranch, RunnableLambda

from pydantic import BaseModel, Field
from typing import Literal

from dotenv import load_dotenv


load_dotenv()
model = ChatGroq(model="llama-3.3-70b-versatile")

str_parser = StrOutputParser()

class Feedback(BaseModel):
    sentiment: Literal['positive', 'negative'] = Field(description="The sentiment of the feedback, must be either positive or negative")

pydantic_parser = PydanticOutputParser(pydantic_object=Feedback)

prompt1 = PromptTemplate(
    template="Classify the sentiment of the following text into positive or negative: {feedback}. {format_instruction}",
    input_variables=['feedback'],
    partial_variables={'format_instruction': pydantic_parser.get_format_instructions()}
)

classifier_chain = prompt1 | model | pydantic_parser

prompt_possitive = PromptTemplate(
    template="Write an appropiate response to this possitive feedback: {feedback}",
    input_variables=['feedback']
)

prompt_negative = PromptTemplate(
    template="Write an appropiate response to this negative feedback: {feedback}",
    input_variables=['feedback']
)

branch_chain = RunnableBranch(
    (lambda x: x.sentiment=="positive", prompt_possitive | model | str_parser),
    (lambda x: x.sentiment=="negative", prompt_negative | model | str_parser),
    RunnableLambda(lambda x: "No valid sentiment found")
)

chain = classifier_chain | branch_chain

result = chain.invoke({"feedback": "This place sucks"})
print(result)
