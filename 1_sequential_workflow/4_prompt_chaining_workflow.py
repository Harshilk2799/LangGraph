from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from typing import TypedDict
from decouple import config

# Define State
class BlogState(TypedDict):
    topic: str 
    outline: str
    content: str

chat_model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=config("GOOGLE_GEMINI_API_KEY"))

# Define your graph
graph = StateGraph(BlogState)

def generate_outline(state: BlogState)->BlogState:
    topic = state["topic"]

    prompt = PromptTemplate(
        template="Generate a detailed outline for a blog on the topic - {topic}",
        input_variables=["topic"]
    )
    outline_prompt = prompt.format(topic=topic)
    outline_gen = chat_model.invoke(outline_prompt).content
    state["outline"] = outline_gen
    return state
    

def generate_blog(state: BlogState)->BlogState:
    topic = state["topic"]
    outline = state["outline"]

    prompt = PromptTemplate(
        template="Write a detailed blog on the topic - {topic} using the following outline \n {outline}",
        input_variables=["topic", "outline"]
    )
    blog_prompt = prompt.format(topic=topic, outline=outline)
    blog_gen = chat_model.invoke(blog_prompt).content
    state["content"] = blog_gen
    return state

# add nodes to your graph 
graph.add_node("generate_outline", generate_outline)
graph.add_node("generate_blog", generate_blog)

# add edges to your graph
graph.add_edge(START, "generate_outline")
graph.add_edge("generate_outline", "generate_blog")
graph.add_edge("generate_blog", END)

# compile the graph
workflow = graph.compile()

# execute the graph
initial_state = {"topic": "What is Python programming"}
final_state = workflow.invoke(initial_state)
print(final_state)