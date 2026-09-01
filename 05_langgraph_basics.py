from typing import TypedDict
from langgraph.graph import StateGraph, START, END
class MyState(TypedDict):
    message: str
    add_exclamation: bool
graph = StateGraph(MyState)

def decide_next(state: MyState):
    if state["add_exclamation"]:
        return "add_exclamation"
    else: return "end"
def add_hello(state: MyState):
    return {
        "message": "Hello " + state["message"]
    }
def add_exclamation(state: MyState):
    return {
        "message": state["message"] +"!"
    }
graph.add_node("add_hello", add_hello)
graph.add_node("add_exclamation", add_exclamation)
graph.add_edge(START, "add_hello")
graph.add_conditional_edges(
    "add_hello",
    decide_next,
    {"add_exclamation": "add_exclamation",
     "end":END
     }
)
graph.add_edge("add_exclamation", END)
app = graph.compile()
result = app.invoke({
    "message": "Jack",
    "add_exclamation": True
})
print(result)
# ----- Reducer example -----
from typing import TypedDict, Annotated
import operator
class MessageState(TypedDict):
    messages: Annotated[list[str], operator.add]
graph2 = StateGraph(MessageState)
def node_a(state: MessageState):
    return {
        "messages": ["A"]
    }

def node_b(state: MessageState):
    return {
        "messages": ["B"]
    }
graph2.add_node("node_a", node_a)
graph2.add_node("node_b", node_b)

graph2.add_edge(START, "node_a")
graph2.add_edge("node_a", "node_b")
graph2.add_edge("node_b", END)
app2 = graph2.compile()
result2 = app2.invoke({
    "messages":["SSSS"]
})
print (result2)