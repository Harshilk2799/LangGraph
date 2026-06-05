from typing import List, Annotated, TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_google_genai import ChatGoogleGenerativeAI
from decouple import config

chat_model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=config("GOOGLE_GEMINI_API_KEY"))

# Define State
class ChatState(TypedDict):
    messages: Annotated[list, add_messages]

def chatbot(state: ChatState)-> ChatState:
    response = chat_model.invoke(state["messages"])
    return {"messages": [response]}

graph_builder = StateGraph(ChatState)
graph_builder.add_node("chatbot", chatbot)

graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)


# Context manager handles connection lifecycle
with SqliteSaver.from_conn_string("./my_checkpoints.db") as checkpointer:

    workflow = graph_builder.compile(checkpointer=checkpointer)

    config = {"configurable": {"thread_id": "harshil-session-001"}}

    while True:
        user_input = input("You: ")

        if user_input in ["exit", "bye", "quit"]:
            print("Bot: Goodbye!")
            break 

        result = workflow.invoke(
            {
                "messages": [
                    {"role": "user", "content": user_input}
                ]
            },
            config=config
        )
        print("AI: ", result["messages"][-1].content)
