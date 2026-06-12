import json

from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser

from prompts import QUESTION_PROMPT
from judge import evaluate_answer

load_dotenv()
llm = ChatGroq(
    model="llama-3.3-70b-versatile"
)
parser = StrOutputParser()

def generate_answer(question: str) -> str:
    chain = QUESTION_PROMPT | llm | parser
    return chain.invoke(question)

def ask_until_good(question: str, threshold: int=8, max_attempts: int=3):
    attempts = 0
    while attempts<max_attempts:
        answer = generate_answer(question),
        evaluation = evaluate_answer(question, answer)
        # print("AI Answer: ", answer)
        # print(f"Judge Answer:\n{json.dumps(evaluation, indent=4)}")
        # print()

        if evaluation.get('accuracy', 0) >= threshold and not evaluation.get('hallucination', True):
            print("Answer meets the threshold")
            print(f"Final answer: {answer}")
            print("\n\n")
            break
        else:
            print("Answer is below the threshold, regenerating...")
            print("\n")
            max_attempts+=1



questions = [
    "Which is the capital of France?",
    "Who wrote the book: 'Harry Potter'",
    "Name the largest but dense city in the world",
]

for question in questions:
    ask_until_good(question)