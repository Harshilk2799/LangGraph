import os
import sqlite3
from decouple import config
from dotenv import load_dotenv
from tavily import TavilyClient
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, START, END 
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.sqlite import SqliteSaver

client = TavilyClient(api_key=os.environ.get("TAVILY_API_KEY"))

@tool
def search(query: str) -> list[dict]:
    """
    Search the web for relevant information based on a natural language query.
    Use this when you need up-to-date or factual information not available 
    in your training data.
    
    Args:
        query: The search query string.
    
    Returns:
        A list of dicts, each containing 'title', 'url', and 'snippet'.
        Returns an empty list if the search fails or no results are found.
    """
    if not query or not query.strip():
        return []

    try:
        response = client.search(
            query=query,
            max_results=5,
            search_depth="advanced",
        )
    except Exception as e:
        print(f"[search] Tavily search failed: {e}")
        return []

    results = []
    for r in response.get("results", []):
        snippet = r.get("content", "").strip()
        truncated = (
            snippet[:400].rsplit(" ", 1)[0] + "..."
            if len(snippet) > 400 else snippet
        )
        results.append({
            "title": r.get("title", "Untitled"),
            "url": r.get("url", ""),
            "snippet": truncated,
        })

    return results

@tool
def calculator(first_num: float, second_num: float, operation: str) -> dict:
    """
    Perform a basic arithmetic operation on two numbers.
    
    Args:
        first_num: The first number.
        second_num: The second number.
        operation: One of 'add', 'sub', 'mul', 'div'.
    
    Returns:
        A dict with key 'result' on success, or 'error' on failure.
    """
    try:
        first_num = float(first_num)
        second_num = float(second_num)
    except (TypeError, ValueError):
        return {"error": "Both inputs must be valid numbers."}

    if operation == "add":
        result = first_num + second_num
    elif operation == "sub":
        result = first_num - second_num
    elif operation == "mul":
        result = first_num * second_num
    elif operation == "div":
        if second_num == 0:
            return {"error": "Division by zero is not allowed!"}
        result = first_num / second_num
    else:
        return {"error": f"Unsupported operation `{operation}`. Use add, sub, mul, or div."}

    return {"result": result}

load_dotenv()
chat_model = ChatGroq(model="qwen/qwen3-32b", api_key=config("GROQ_API_KEY"))


# Make tool list
tools = [search, calculator]

# Make the LLM tool-aware
llm_with_tools = chat_model.bind_tools(tools)
tool_node = ToolNode(tools) # Executes tool calls


class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

def chat_node(state: ChatState):
    message = state["messages"]
    response = llm_with_tools.invoke(message)
    return {"messages": [response]}


graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_node("tools", tool_node)

graph.add_edge(START, "chat_node")

# If the LLM asked for a tool, go to ToolNode; else finish
graph.add_conditional_edges("chat_node", tools_condition)
graph.add_edge("tools", "chat_node")    # tool result wapas LLM ko jaaye

conn = sqlite3.connect(database='chatbot.db', check_same_thread=False)

# Checkpointer
checkpointer = SqliteSaver(conn=conn)

chatbot = graph.compile(checkpointer=checkpointer)

thread_id = "thread-1"
while True:
    user_input = input("You: ").strip()

    if user_input.lower() in ["exit", "quit", "q"]:
        print("Chatbot band ho gaya. Bye!")
        break

    res = chatbot.invoke({"messages": [HumanMessage(content=user_input)]},
        config={"configurable": {"thread_id": thread_id}}
    )
    print("Bot:", res["messages"][-1].content)
    print()
