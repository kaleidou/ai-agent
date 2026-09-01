from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
load_dotenv()

model = ChatOpenAI(
    model="gpt-5-mini"
)
@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    return f"The weather in {city} is sunny"

model_with_tools = model.bind_tools([get_weather])

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI assitant."),
     ("human", "what's the weather in {city}")
])
parser = StrOutputParser()
chain = prompt|model_with_tools
response = chain.invoke({
    "city": "LA"
})
#print (response)
tool_call = response.tool_calls[0]
tool_response = get_weather.invoke(tool_call["args"])
print(tool_response)
from langchain_core.messages import ToolMessage
tool_message = ToolMessage(
    content = tool_response,
    tool_call_id = tool_call["id"]
)
messages = [
    SystemMessage(content="You are a helpful AI assitant."),
    HumanMessage(content="Whats the weather in LA")
]
messages.append(response)
messages.append(tool_message)
final_response = model_with_tools.invoke(messages)
print(final_response.content)