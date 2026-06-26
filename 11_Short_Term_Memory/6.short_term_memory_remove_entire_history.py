from langgraph.graph import StateGraph, MessagesState, START, END 
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.messages import HumanMessage, RemoveMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from decouple import config

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=config("GOOGLE_GEMINI_API_KEY"))

def chat_node(state: MessagesState):
    response = llm.invoke(state["messages"])
    return {"messages": [response]}

def clear_history_node(state: MessagesState):
    """
    This wipes EVERY message currently in state.
    REMOVE_ALL_MESSAGES is a special sentinel — the add_messages
    reducer detects it and deletes all existing messages before
    adding anything new.
    """
    return {"messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES)]}

# Routing: decide whether user wants to clear chat
def router(state: MessagesState):
    last_msg = state["messages"][-1].content.strip()
    if last_msg.strip() in ["clear", "clear history", "reset", "/reset"]:
        return "clear_history"
    return "chat"

builder = StateGraph(MessagesState)
builder.add_node("chat", chat_node)
builder.add_node("clear_history", clear_history_node)

builder.add_conditional_edges(START, router, {"chat": "chat", "clear_history": "clear_history"})
builder.add_edge("chat", END)
builder.add_edge("clear_history", END)

checkpointer = InMemorySaver()
graph = builder.compile(checkpointer=checkpointer)

config = {"configurable": {"thread_id": "user-123"}}

# Normal turns — history builds up
graph.invoke({"messages": [HumanMessage(content="Hi, I'm Harshil")]}, config)
graph.invoke({"messages": [HumanMessage(content="What's LangGraph?")]}, config)

state = graph.get_state(config)
print("Before clear:", len(state.values["messages"]), "messages")

# Trigger full history wipe
graph.invoke({"messages": [HumanMessage(content="clear")]}, config)

state = graph.get_state(config)
print("After clear:", len(state.values["messages"]), "messages")  # → 0
