from langchain_core.prompts import PromptTemplate

dynamic_template = PromptTemplate(
    input_variables=["topic"],
    template="Write a short paragaph about {topic} in a {style} style"
)

prompt_text = dynamic_template.format(topic="artificial intelligence", style="humorous")
print(prompt_text)