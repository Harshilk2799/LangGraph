from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.messages import RemoveMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from decouple import config

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=config("GOOGLE_GEMINI_API_KEY"))

def chat(state: MessagesState):
    response = llm.invoke(state["messages"])
    return {"messages": [response]}

def delete_old_messages(state: MessagesState):
    msgs = state["messages"]

    # if more than 10 messages, delete the earliest 6
    if len(msgs) > 10:
        to_remove = msgs[:6]
        return {"messages": [RemoveMessage(id=m.id) for m in to_remove]}
    return {}

builder = StateGraph(MessagesState)
builder.add_node("chat", chat)
builder.add_node("cleanup", delete_old_messages)

builder.add_edge(START, "chat")
builder.add_edge("chat", "cleanup")
builder.add_edge("cleanup", END)

graph = builder.compile(checkpointer=InMemorySaver())

config = {"configurable": {"thread_id": "t1"}}

# Run multiple turns
print(graph.invoke({"messages": [{"role": "user", "content": "Hi, I'm Nitish"}]}, config))
print(graph.invoke({"messages": [{"role": "user", "content": "Tell me about LangGraph"}]}, config))
print(graph.invoke({"messages": [{"role": "user", "content": "Now explain checkpointers"}]}, config))
print(graph.invoke({"messages": [{"role": "user", "content": "What is Langchain"}]}, config))
print(graph.invoke({"messages": [{"role": "user", "content": "What is Quantum Mechanics"}]}, config))
print(graph.invoke({"messages": [{"role": "user", "content": "What is Gen AI"}]}, config))
print(graph.invoke({"messages": [{"role": "user", "content": "What is my name"}]}, config))

snap = graph.get_state(config)
print("Stored messages after cleanup:", len(snap.values["messages"]))
