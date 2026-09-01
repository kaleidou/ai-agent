from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool

load_dotenv()

model = ChatOpenAI(model="gpt-5-mini")

@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    return f"The weather in {city} is sunny."

@tool
def calculate(expression: str) -> str:
    """Calculate a mathematical expression."""
    return str(eval(expression))

tools = [get_weather, calculate]

from langchain.agents import create_agent

agent = create_agent(
    model = model,
    tools = tools,
    system_prompt="You are a helpful AI assistant."
)
result = agent.invoke({
    "messages": [
        {
            "role": "user",
            "content": "Whats the weather in LA and whats 39*39"
        }
    ]
})
print (result["messages"][-1].content)