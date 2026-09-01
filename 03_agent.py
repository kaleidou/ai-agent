from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage

load_dotenv()

model = ChatOpenAI(
    model="gpt-5-mini"
)
@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    return f"The weather in {city} is sunny."
@tool
def calculate(expression:str) ->str:
    """Calculate a mathematical expression."""
    return str(eval(expression))
tools = [get_weather, calculate]
model_with_tools = model.bind_tools(tools)
tools_by_name = {
    tool.name:tool for tool in tools
}
messages = [
    HumanMessage(content="Whats the weather in LA and whats 39*39")
]
while True:
    response = model_with_tools.invoke(messages)
    messages.append(response)
    if not response.tool_calls:
        print(response.content)
        break
    for tool_call in response.tool_calls:
        selected_tool = tools_by_name[tool_call["name"]]
        tool_result = selected_tool.invoke(tool_call["args"])
        messages.append(
            ToolMessage(content=str(tool_result),
                        tool_call_id = tool_call["id"])
        )