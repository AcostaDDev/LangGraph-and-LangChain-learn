from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

chat_model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

print("Blog post generator")
print("Provide ideas or topics for the blog post. Type 'exit' to finish")

topic = input("Enter blog post topic: ")

chat_prompt_template = ChatPromptTemplate([
    SystemMessagePromptTemplate.from_template("You are a professional blog writer. Help generate informative, engaging and well structured blog post about a {topic}"),
    HumanMessagePromptTemplate.from_template("Write a detailed blog post about {topic}")
])

chat_history = []
while True:
    user_input = input("Ideas or instructions or type exit: ")

    if user_input == "exit":
        break

    messages = chat_prompt_template.format_messages(topic=topic)
    
    for prev in chat_history:
        messages.append(prev)

    messages.append(HumanMessagePromptTemplate.from_template(user_input).format_messages()[0])

    response = chat_model.invoke(messages)
    print("Blog Post Content:\n", response.content)