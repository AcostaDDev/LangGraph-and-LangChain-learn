from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate

chat_prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template("You are a helpful assistant that provides information about {subjet}"),
    HumanMessagePromptTemplate.from_template("Can you tell me something interesting about {subjet}?")
])

prompt_text = chat_prompt.format_messages(subjet="quentum computing")
print(prompt_text)