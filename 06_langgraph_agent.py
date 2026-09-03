from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from typing import TypedDict, Annotated

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
load_dotenv()
@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    return f"The weather in {city} is sunny."

@tool
def calculate(expression: str) -> str:
    """Calculate a mathematical expression."""
    return str(eval(expression))

tools = [get_weather, calculate]

model = ChatOpenAI(
    model="gpt-5-mini"
)

model_with_tools = model.bind_tools(tools)

def call_model(state: AgentState):
    response = model_with_tools.invoke(state["messages"])
    return {
        "messages": [response]
    }

tools_by_name = {
    tool.name: tool
    for tool in tools
}
from langchain_core.messages import ToolMessage

def call_tools(state: AgentState):
    last_message = state["messages"][-1]
    tool_messages = []
    for tool_call in last_message.tool_calls:
        tool_result = tools_by_name[tool_call["name"]].invoke(tool_call["args"])
        tool_messages.append(ToolMessage(
            content = str(tool_result),
            tool_call_id = tool_call["id"]
        ))
    return {
        "messages": tool_messages
    }

def should_continue(state: AgentState):
    last_message = state["messages"][-1]

    if last_message.tool_calls:
        return "tools"
    else:
        return "end"

graph = StateGraph(AgentState)
from langgraph.prebuilt import tools_condition

from langgraph.prebuilt import ToolNode
tool_node = ToolNode(tools)

graph.add_node("model", call_model)
graph.add_node("tools", tool_node)
graph.add_edge(START, "model")
graph.add_conditional_edges(
    "model",
    tools_condition
)
graph.add_edge("tools","model")




app = graph.compile()
from langchain_core.messages import HumanMessage
response = app.invoke({
    "messages": [HumanMessage(
    content="What's the weather in LA and what is 39 * 39?")
]
})
print(response["messages"][-1].content)
