from langcache import LangCache
from dotenv import load_dotenv
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_google_genai import ChatGoogleGenerativeAI
from decouple import config

load_dotenv()

thread_config = {"configurable": {"thread_id": "Jay"}}

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", api_key=config("GOOGLE_GEMINI_API_KEY"))

lang_cache = LangCache(
    server_url=config("LANGCACHE_HOST"),
    cache_id=config("LANGCACHE_CACHE_ID"),
    api_key=config("LANGCACHE_API_KEY")
)

# 0.85-0.90 is a good starting point for chatbot Q&A. Lower it (e.g. 0.7)
# for more aggressive matching, raise it if you get false-positive hits.
SIMILARITY_THRESHOLD = 0.85


class ChatState(TypedDict):
    messages: Annotated[list, add_messages]
    user_input: str 
    cache_hit: bool 
    cached_response: str 

def check_cache(state: ChatState):
    """Search LangCache for a semantically similar prompt."""
    results = lang_cache.search(
        prompt=state["user_input"],
        similarity_threshold=SIMILARITY_THRESHOLD
    )

    if results.data:
        best_match = results.data[0]
        # print("Best Match: ", best_match)
        return {"cache_hit": True, "cached_response": best_match.response}
    return {"cache_hit": False, "cached_response": ""}

def route_after_cache(state: ChatState):
    return "use_cache" if state["cache_hit"] else "call_llm"

def use_cache(state: ChatState):
    """Reuse the cached answer instead of calling the LLM."""
    return {"messages": [AIMessage(content=state["cached_response"])]}

def call_llm(state: ChatState):
    """Cache miss -> call the LLM, then write the pair back to LangCache."""
    ai_message = llm.invoke(state["messages"])

    lang_cache.set(
        prompt=state["user_input"],
        response=ai_message.content
    )

    return {"messages": [ai_message]}

builder = StateGraph(ChatState)
builder.add_node("check_cache", check_cache)
builder.add_node("use_cache", use_cache)
builder.add_node("call_llm", call_llm)

builder.add_edge(START, "check_cache")
builder.add_conditional_edges("check_cache", route_after_cache, {"use_cache": "use_cache", "call_llm": "call_llm"})
builder.add_edge("use_cache", END)
builder.add_edge("call_llm", END)

def chat():
    print("LangGraph + Redis LangCache chatbot. Type 'exit' to quit.\n")

    with SqliteSaver.from_conn_string("checkpoints.db") as checkpointer:
        graph = builder.compile(checkpointer=checkpointer)
    
        while True:
            user_input = input("You: ").strip()
            if user_input.lower() in ["exit", "quit"]:
                break

            result = graph.invoke(
                {
                    "messages": [HumanMessage(content=user_input)],
                    "user_input": user_input,
                    "cache_hit": False,
                    "cached_response": "",
                },
                config=thread_config,
            )

            tag = "⚡ cache" if result["cache_hit"] else "🧠 llm"
            print(f"Bot [{tag}]: {result['messages'][-1].content}\n")


if __name__ == "__main__":
    chat()
