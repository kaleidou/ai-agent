from dotenv import load_dotenv

from typing import TypedDict, Annotated

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_core.tools import tool
from langchain_core.messages import BaseMessage

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition


load_dotenv()

documents = [
    Document(
        page_content="LangChain is a framework for building applications powered by language models."
    ),
    Document(
        page_content="LangGraph is a framework for building stateful, multi-step AI applications using graphs."
    ),
    Document(
        page_content="RAG combines information retrieval with language generation."
    )
]

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=50,
    chunk_overlap=10
)

chunks = text_splitter.split_documents(documents)

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small"
)

vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings
)

retriever = vector_store.as_retriever(
    search_kwargs={"k": 2}
)

@tool
def search_knowledge_base(query: str) -> str:
    """Search the knowledge base for information about LangChain, LangGraph, and RAG."""

    docs = retriever.invoke(query)

    return "\n".join(
        doc.page_content for doc in docs
    )

tools = [search_knowledge_base]

model = ChatOpenAI(
    model="gpt-5-mini"
)

model_with_tools = model.bind_tools(tools)

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

from langchain_core.messages import SystemMessage

def call_model(state: AgentState):

    messages = [
        SystemMessage(
            content=(
                "You are a helpful assistant. "
                "Use the knowledge base when necessary. "
                "Do not call the knowledge base repeatedly "
                "for the same question."
            )
        )
    ] + state["messages"]

    response = model_with_tools.invoke(messages)

    return {
        "messages": [response]
    }

tool_node = ToolNode(tools)

graph = StateGraph(AgentState)

graph.add_node("model", call_model)
graph.add_node("tools", tool_node)

graph.add_edge(START, "model")

graph.add_conditional_edges(
    "model",
    tools_condition
)

graph.add_edge("tools", "model")

app = graph.compile()
result = app.invoke({
    "messages": [
        ("user", "What is LangGraph?")
    ]
})
for message in result["messages"]:
    print(type(message).__name__)
    print(message.content)
    print("-----")