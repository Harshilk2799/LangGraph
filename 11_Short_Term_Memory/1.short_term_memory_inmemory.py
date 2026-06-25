from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.checkpoint.memory import InMemorySaver
from langchain_google_genai import ChatGoogleGenerativeAI
from decouple import config

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=config("GOOGLE_GEMINI_API_KEY"))

def call_model(state: MessagesState):
    response = llm.invoke(state["messages"])
    return {"messages": [response]}

builder = StateGraph(MessagesState)
builder.add_node("call_model", call_model)

builder.add_edge(START, "call_model")
builder.add_edge("call_model", END)

# Before memory
# graph = builder.compile()
# print(graph.invoke({"messages": [{"role": "user", "content": "Hi! My name is Nitish."}]}))
# print(graph.invoke({"messages": [{"role": "user", "content": "What is my name?"}]}))


# After memory
checkpointer = InMemorySaver()
graph = builder.compile(checkpointer=checkpointer)

config = {"configurable": {"thread_id": "thread-1"}}
config2 = {"configurable": {"thread_id": "thread-2"}}

print(graph.invoke({"messages": [{"role": "user", "content": "Hi! My name is Nitish."}]}, config))
print(graph.invoke({"messages": [{"role": "user", "content": "What is my name?"}]}, config2))

print()

snap = graph.get_state(config)
vals = snap.values
for m in vals.get("messages", []):
        print("-", type(m).__name__, ":", m.content)