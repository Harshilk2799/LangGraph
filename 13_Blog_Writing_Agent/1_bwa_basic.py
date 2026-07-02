import operator
from typing import List, TypedDict, Annotated
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_groq import ChatGroq
from decouple import config
from pathlib import Path

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=config("GROQ_API_KEY")
)
class Task(BaseModel):
    id: int 
    title: str
    brief: str = Field(..., description="What to cover")

class Plan(BaseModel):
    blog_title: str 
    tasks: List[Task]

class State(TypedDict):
    topic: str 
    plan: Plan 
    # reducer: results from workers get concatenated automatically
    sections: Annotated[List[str], operator.add]
    final: str 

def orchestrator(state: State):
    plan = llm.with_structured_output(Plan).invoke(
        [
            SystemMessage(
                content="Create a blog plan with 5-7 sections on the following topic."
            ),
            HumanMessage(
                content=f"Topic: {state["topic"]}"
            )
        ]
    )
    return {"plan": plan}

def fanout(state: State):
    return [
        Send("worker", {"task": task, "topic": state["topic"], "plan": state["plan"]})
        for task in state["plan"].tasks
    ]

def worker(payload: dict)-> dict:

    # payload contains what we sent
    task = payload["task"]
    topic = payload["topic"]
    plan = payload["plan"]

    blog_title = plan.blog_title

    section_md = llm.invoke(
        [
            SystemMessage(
                content="Write one clean Markdown section."
            ),
            HumanMessage(
                content=(
                    f"Blog: {blog_title}\n"
                    f"Topic: {topic}\n"
                    f"Section: {task.title}\n"
                    f"Brief: {task.brief}\n"
                    "Return only the section content in markdown."
                )
            )
        ]
    ).content.strip()

    return {"sections": [section_md]}


def reducer(state: State):

    title = state["plan"].blog_title
    body = "\n\n".join(state["sections"]).strip()

    final_md = f"# {title}\n\n{body}\n"

    finalname = title.lower().replace(" ", "_") + ".md"
    output_path = Path(finalname)
    output_path.write_text(final_md, encoding="utf-8")

    return {"final": final_md}

builder = StateGraph(State)
builder.add_node("orchestrator", orchestrator)
builder.add_node("worker", worker)
builder.add_node("reducer", reducer)

builder.add_edge(START, "orchestrator")
builder.add_conditional_edges("orchestrator", fanout, ["worker"])
builder.add_edge("worker", "reducer")
builder.add_edge("reducer", END)

graph = builder.compile()

out = graph.invoke({"topic": "Write a blog on Self Attention", "sections": []})
print(out)