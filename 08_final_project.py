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
        page_content= ("LangChain is a framework for building applications powered by language models. "
            "It provides abstractions for prompts, models, tools, retrievers, and agents."
        )
    ),
    Document(
    page_content=(
        "Project Aurora is an internal AI project. "
        "It uses LangGraph for orchestration and Chroma for vector storage. "
        "The project is scheduled to launch in December 2026."
        "The project is scheduled to launch on December 15, 2026."
        )
    ),
    Document(
        page_content=(
            "LangGraph is a framework for building stateful and multi-step AI applications. "
            "It represents application workflows as graphs containing nodes, edges, and shared state."
        )
    ),
     Document(
        page_content=(
            "Retrieval-Augmented Generation, or RAG, combines information retrieval with language generation. "
            "Relevant documents are retrieved from a knowledge base and provided to the language model as context."
        )
    ),

    Document(
        page_content=(
            "An AI agent uses a language model to make decisions and can interact with external tools. "
            "Agents can repeatedly call tools, observe their results, and decide what action to take next."
        )
    ),

    Document(
        page_content=(
            "Embeddings convert text into numerical vectors that represent semantic meaning. "
            "Vector databases can compare these vectors to find documents that are semantically similar to a query."
        )
    )
]

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 50,
    chunk_overlap = 12
)

chunks = text_splitter.split_documents(documents)

embeddings = OpenAIEmbeddings(
    model = "text-embedding-3-small"
)

vector_store = Chroma.from_documents(documents=chunks, embedding=embeddings)

retriever = vector_store.as_retriever(
    search_kwargs={"k":3}
)
@tool
def search_knowledge_base(query:str) ->str:
    """Search the internal knowledge base for information about projects, AI frameworks, and technical concepts. Use this tool when the user asks about information that may be stored in the internal knowledge base."""
    docs = retriever.invoke(query)
    return "\n".join(doc.page_content for doc in docs)
@tool
def calculate(expression: str) -> str:
    """Calculate a mathematical expression."""
    return str(eval(expression))

tools = [search_knowledge_base, calculate]
model = ChatOpenAI(
    model ="gpt-5-mini"
)
model_with_tools = model.bind_tools(tools)

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

def call_model(state: AgentState):
    response = model_with_tools.invoke(state["messages"])
    return {"messages": [response]}
graph = StateGraph(AgentState)
graph.add_node("model", call_model)
tool_node = ToolNode(tools)
graph.add_node("tools", tool_node)
graph.add_edge(START, "model")
graph.add_conditional_edges(
    "model",
    tools_condition
)
graph.add_edge("tools","model")
app = graph.compile()
result = app.invoke({
    "messages": [
        ("user", "Project Aurora launches in how many days from now? and what is 39*39")
    ]
})

#print (result["messages"][-1].content)
for message in result["messages"]:
    print(type(message).__name__)
    print(message.content)
    print("-----")