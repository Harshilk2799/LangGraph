from langgraph.graph import StateGraph, START, END
from typing import TypedDict
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from decouple import config
from langgraph.checkpoint.memory import InMemorySaver

chat_model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=config("GOOGLE_GEMINI_API_KEY"))


class JokeState(TypedDict):
    topic: str 
    joke: str 
    explanation: str 

def generate_joke(state: JokeState):
    topic = state["topic"]
    prompt = PromptTemplate(
        template="Generate a joke on the topic {topic}",
        input_variables=["topic"]
    )
    joke_prompt = prompt.format(topic=topic)
    response = chat_model.invoke(joke_prompt).content
    return {"joke": response}

def generate_explanation(state: JokeState):
    joke = state["joke"]
    prompt = PromptTemplate(
        template="Write an explanation for the joke - {joke}",
        input_variables=["joke"]
    )
    joke_explanation_prompt = prompt.format(joke=joke)
    response = chat_model.invoke(joke_explanation_prompt).content
    return {"explanation": response}


graph = StateGraph(JokeState)

graph.add_node("generate_joke", generate_joke)
graph.add_node("generate_explanation", generate_explanation)

graph.add_edge(START, "generate_joke")
graph.add_edge("generate_joke", "generate_explanation")
graph.add_edge("generate_explanation", END)

checkpointer = InMemorySaver()

workflow = graph.compile(checkpointer=checkpointer)

configure1 = {"configurable": {"thread_id": "1"}}

response = workflow.invoke({"topic": "Pizza"}, config=configure1)

print(response)


# Get the latest state snapshot for a thread
snapshot = workflow.get_state(configure1)
print(snapshot)

print(snapshot.values)        # current state dict
print(snapshot.next)          # which node would run next
print(snapshot.metadata)      # step count, timestamps


# Iterate all checkpoints (newest first)
history = list(workflow.get_state_history(configure1))
print(history)