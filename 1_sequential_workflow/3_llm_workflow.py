from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI
from typing import TypedDict
from decouple import config

# Define State
class LLMState(TypedDict):
    question: str 
    answer: str

chat_model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=config("GOOGLE_GEMINI_API_KEY"))

# Define your graph
graph = StateGraph(LLMState)

def llm_qa(state: LLMState)->LLMState:
    # extract the question from state
    question = state["question"]

    # form a prompt 
    prompt = f"Answer the following question {question}"

    # ask that question to the LLM 
    answer = chat_model.invoke(prompt).content

    # update the answer in the state
    state["answer"] = answer
    return state

# add nodes to your graph 
graph.add_node("llm_qa", llm_qa)

# add edges to your graph
graph.add_edge(START, "llm_qa")
graph.add_edge("llm_qa", END)

# compile the graph
workflow = graph.compile()

# execute the graph
initial_state = {"question": "What is Python programming"}
final_state = workflow.invoke(initial_state)
print(final_state)