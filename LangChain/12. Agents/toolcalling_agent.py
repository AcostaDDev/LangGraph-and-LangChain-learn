from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq
from langchain.tools import tool
from langchain.agents import create_agent


llm = ChatGroq(
    model="llama-3.3-70b-versatile"
)

@tool
def multipy(a: float, b: float) -> float:
    """Multiply two numbers"""
    return a*b

@tool
def add(a: float, b: float) -> float:
    """Add two numbers"""
    return a+b

tools: list = [add, multipy]

system_prompt = """
You are a helpful AI Agent. Use tools when necessary. If no tool is required, answer directly
"""

agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt=system_prompt
)

print("LangChain Basic Agent. Type 'exit' to quit.")
while True:
    query = input("You: ")
    if query.lower() == "exit":
        break

    response = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": query
                }
            ]
        }
    )

    print("\nAI: ", response["messages"][-1].content)
    print("\n==========AI AGENT DEBUG MODE==========\n")
    print(response)
    print("\n\n")
    
