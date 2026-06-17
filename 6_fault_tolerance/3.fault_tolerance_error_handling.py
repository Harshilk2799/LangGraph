from typing import TypedDict, Optional
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_groq import ChatGroq
from decouple import config
from langchain_google_genai import ChatGoogleGenerativeAI

# Pillar 3 — Error Handling with Fallback Nodes
# => Sometimes retrying isn't enough. You need a fallback — if node A keeps failing, route to node B as a backup.
# => Retry ke baad bhi fail ho jaye, toh fallback node pe redirect karo using conditional edges.


# State includes an error field
class State(TypedDict):
    query: str
    response: str
    error: Optional[str]   # ← key for fault tolerance routing

# Primary LLM (fast, might fail)
groq_llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=config("GROQ_API_KEY"))

# Fallback LLM (reliable backup)
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=config("GOOGLE_GEMINI_API_KEY"))

# Node 1: Try primary LLM
def primary_llm(state: State) -> State:
    try:
        print("Trying primary LLM (Groq)...")
        result = groq_llm.invoke(state["query"])
        return {"response": result.content, "error": None}
    except Exception as e:
        print(f"Primary LLM failed: {e}")
        return {"response": "", "error": str(e)}

# Node 2: Fallback LLM
def fallback_llm(state: State) -> State:
    print("Falling back to OpenAI GPT-4o-mini...")
    result = llm.invoke(state["query"])
    return {"response": result.content, "error": None}

# Router: decide where to go based on error
def route_after_primary(state: State) -> str:
    if state.get("error"):
        return "fallback_llm"   # go to fallback
    return END                  # success, finish

# Build graph
builder = StateGraph(State)
builder.add_node("primary_llm", primary_llm)
builder.add_node("fallback_llm", fallback_llm)

builder.add_edge(START, "primary_llm")

# Conditional routing
builder.add_conditional_edges(
    "primary_llm",
    route_after_primary,
    {
        "fallback_llm": "fallback_llm",
        END: END
    }
)
builder.add_edge("fallback_llm", END)

graph = builder.compile(checkpointer=MemorySaver())

# Run
config = {"configurable": {"thread_id": "fault-test-1"}}
result = graph.invoke(
    {"query": "What are 3 tourist spots in Paris?", "error": None},
    config
)
print("Response:", result["response"])