from langchain_core.prompts import PromptTemplate

static_template = PromptTemplate(
    input_variables=[],
    template="Write a short fun fact about AI"
)

prompt_text = static_template.format()
print(prompt_text)