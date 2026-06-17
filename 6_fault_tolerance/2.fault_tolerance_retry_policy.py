from typing import TypedDict
from decouple import config
from langgraph.func import task
from langgraph.types import RetryPolicy
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.pregel.main import RetryPolicy
from langchain_google_genai import ChatGoogleGenerativeAI

# Pillar 2 — Retry Policy
# => Retry policy tells LangGraph: "If a node fails, don't give up — try again automatically with a delay."

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=config("GOOGLE_GEMINI_API_KEY"))

class State(TypedDict):
    query: str 
    response: str 

attempt_count = 0 

# Way 1
# @task(retry_policy=RetryPolicy(max_attempts=3, initial_interval=1.0, backoff_factor=2.0, retry_on=Exception))
# def call_llm(state: State):
#     global attempt_count

#     attempt_count += 1 

#     if attempt_count < 3:
#         print(f"Attempt {attempt_count}: Simulating RateLimitError...")
#         raise Exception("RateLimitError: Too many requests")
    
#     print(f"Attempt {attempt_count}: LLM call SUCCESS")
#     result = llm.invoke(state["query"])
#     return {"response": result.content}


def call_llm(state: State):
    global attempt_count

    attempt_count += 1 

    if attempt_count < 3:
        print(f"Attempt {attempt_count}: Simulating RateLimitError...")
        raise Exception("RateLimitError: Too many requests")
    
    print(f"Attempt {attempt_count}: LLM call SUCCESS")
    result = llm.invoke(state["query"])
    return {"response": result.content}

builder = StateGraph(State)
# builder.add_node("call_llm", call_llm)

# Way 2
builder.add_node("call_llm", 
                call_llm, 
                retry_policy=
                RetryPolicy(
                    max_attempts=3,          # max 3 baar try karega
                    initial_interval=1.0,    # pehli retry ke pehle 1 second wait
                    backoff_factor=2.0,      # exponential backoff: 1s, 2s, 4s
                    max_interval=10.0,       # maximum 10s tak wait karega
                    jitter=True              # random jitter add karta hai (thundering herd avoid)
                )
            )

builder.add_edge(START, "call_llm")
builder.add_edge("call_llm", END)

with SqliteSaver.from_conn_string("./my_checkpoints.db") as checkpointer:

    graph = builder.compile(checkpointer=checkpointer)
    config = {"configurable": {"thread_id": "q1"}}

    result = graph.invoke({"query": "What is the capital of France?"}, config)
    print("Final response:", result["response"])
