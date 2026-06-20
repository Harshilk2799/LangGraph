from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.messages import BaseMessage, AIMessage
from typing import Annotated, TypedDict
from langgraph.types import interrupt, Command
from decouple import config
from langchain_groq import ChatGroq


# interrupt(payload) = Graph ko pause karta hai, payload UI ko bhej deta hai
# Command(resume=value) = value wapas interrupt() ka return banta hai, execution continue


# Rules of interrupts
# 1. Node re-run hota hai, resume nahi hota
# 2. Side-effects ko idempotent rakho (ya interrupt() ke baad daalo)
# 3. Checkpointer zaroori hai
# 4. Resume karne ke liye Command use karo, raw value nahi
# 5. Try/except mein interrupt() ko mat wrap karo
# 6. Multiple interrupt() ek node mein — order se match hote hain

chat_model = ChatGroq(model="llama-3.1-8b-instant", api_key=config("GROQ_API_KEY"))

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

def chat_node(state: ChatState):

    decision = interrupt({
        "type": "approval",
        "reason": "Model is abouot to answer a user question.",
        "question": state["messages"][-1].content,
        "instruction": "Approve this question? Yes/no"
    })

    if decision["approved"] == "no":
        return {"messages": [AIMessage(content="Not approved.")]}
    else:
        response = chat_model.invoke(state["messages"])
        return {"messages": [response.content]}
    
builder = StateGraph(ChatState)

builder.add_node("chat", chat_node)

builder.add_edge(START, "chat")
builder.add_edge("chat", END)

checkpointer = InMemorySaver()
graph = builder.compile(checkpointer=checkpointer)

config = {"configurable": {"thread_id": "123"}}

initial_input = {
    "messages": [
        ("user", "Explain gradient desccent in very simple terms.")
    ]
}

result = graph.invoke(initial_input, config=config)

print(result)

message = result["__interrupt__"][0].value 
print(message)


user_input = input("\nBackend message - {message} \n Approve this question? (y/n): ")

final_result = graph.invoke(
    Command(resume={"approved": user_input}),
    config=config
)

print(final_result)