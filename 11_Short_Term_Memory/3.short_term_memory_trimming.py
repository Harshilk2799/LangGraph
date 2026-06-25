from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.messages.utils import trim_messages, count_tokens_approximately
from langchain_google_genai import ChatGoogleGenerativeAI
from decouple import config

# LangGraph short-term memory: trimming messages before the LLM call.

# Problem this solves:
# Without intervention, every turn sends the FULL history to the LLM:
# 1. context window overflow on long conversations
# 2. degraded answer quality ("distracted" by stale turns)
# 3. higher cost per call (every old token, every time)


llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=config("GOOGLE_GEMINI_API_KEY"))

MAX_TOKENS = 150

def call_model(state: MessagesState):
    # Trim conversation history = last N messages that fit within the token budget 
    messages = trim_messages(
        messages=state["messages"],
        strategy="last",
        token_counter=count_tokens_approximately,
        max_tokens=MAX_TOKENS
    )

    print('Current Token Count ->', count_tokens_approximately(messages=messages))
    
    for message in messages:
        print(message.content)

    response = llm.invoke(messages)
    return {"messages": [response]}

builder = StateGraph(MessagesState)
builder.add_node("call_model", call_model)

builder.add_edge(START, "call_model")
builder.add_edge("call_model", END)

checkpointer = InMemorySaver()

graph = builder.compile(checkpointer=checkpointer)

config = {"configurable": {"thread_id": "chat-1"}}

result = graph.invoke({"messages": [{"role": "user", "content": "Hi, My name is Harshil"}]}, config)
print(result["messages"][-1].content, "\n\n")

result = graph.invoke({"messages": [{"role": "user", "content": "I am learning LangGraph"}]}, config)
print(result["messages"][-1].content, "\n\n")

result = graph.invoke({"messages": [{"role": "user", "content": "Can you explain short term memory?"}]}, config)
print(result["messages"][-1].content, "\n\n")

result = graph.invoke({"messages": [{"role": "user", "content": "What is my name?"}]}, config)
print(result["messages"][-1].content, "\n\n")

for item in graph.get_state(config).values['messages']:
    print(item.content)
    print('-'*120)