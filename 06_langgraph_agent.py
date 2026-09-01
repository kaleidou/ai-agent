from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

from typing import TypedDict, Annotated

load_dotenv()