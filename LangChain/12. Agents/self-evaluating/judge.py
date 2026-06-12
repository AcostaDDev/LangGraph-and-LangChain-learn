from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.output_parsers import JsonOutputParser

from prompts import JUDGE_PROMPT

load_dotenv()
judge_llm = ChatGroq(
    model="llama-3.3-70b-versatile"
)
parser = JsonOutputParser()

def evaluate_answer(question: str, answer: str) -> dict:
    chain = JUDGE_PROMPT | judge_llm | parser
    return chain.invoke({
        "question": question,
        "answer": answer
    })

# question = "Which is the capital of France"
# answer = "It is Berlin"

# result = evaluate_answer(question, answer)
# import json
# print(json.dumps(result, indent=4))