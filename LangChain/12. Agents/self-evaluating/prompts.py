from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser


parser = JsonOutputParser()

QUESTION_PROMPT = PromptTemplate(
    template=(
        """
        Answer the following question as accurately and briefly as possible:
        {question}
        """
    ),
    input_variables=['question']
)

JUDGE_PROMPT = PromptTemplate(
    template=(
        """
        You are an expert AI judge.
        Evaluate the following answer to the question
        
        Question: {question}.
        Answer: {answer}

        Assess the answer based on the following criteria:
        1. accuracy (0-10)
        2. hallucination (true/false)
        3. feedback (brief comment)

        {format_instructions}
        """
    ),
    input_variables=['question', 'answer'],
    partial_variables={'format_instructions': parser.get_format_instructions()}
)