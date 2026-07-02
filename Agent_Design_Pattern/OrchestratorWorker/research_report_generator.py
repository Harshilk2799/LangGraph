from langgraph.graph import StateGraph, START, END 
from langgraph.types import Send
from operator import add 
from typing import TypedDict, List, Annotated
from pydantic import BaseModel, Field  
from langchain_google_genai import ChatGoogleGenerativeAI
from decouple import config

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=config("GOOGLE_GEMINI_API_KEY"))

class OverallState(TypedDict):
    research_topic: str 
    sources: List[str]
    worker_findings: Annotated[List[str], add]
    final_report: str 

class WorkerState(TypedDict):
    source: str 
    worker_id: int 
    research_topic: str 


class ResearchPlan(BaseModel):
    sources: List[str] = Field(
        description="List of specific research sources/aspects to investigate.",
        max_length=5
    )
    reasoning: str = Field(
        description="Brief explanation of why these sources where chosen."
    )


def plan_research(state: OverallState):

    print("\n" + "="*70)
    print("ORCHESTRATOR: Planning research strategy.")
    print("="*70)
    print(f"Topic: {state["research_topic"]}\n")

    planner_llm = llm.with_structured_output(ResearchPlan)

    prompt = f"""
    You are a research strategist planning a comprehensive investigation.

    Research topic: {state["research_topic"]}

    CRITICAL INSTRUCTION: Generate between 3-5 specific research sources or aspects to investigate.
    DO NOT generate more than 5 sources.

    Each source should be:
    - Specifi and focused on a distinct aspect
    - Relevant to the overall topic
    - Complementary to other souces (minimal overlap)
    - Concrete enough to guide targeted research

    Examples of good sources:
    - "Clinical trial results and efficacy data."
    - "Economic impact and cost-benefit analysis."
    - "Regulatory framework and compliance requirements."
    - "Patient outcomes and quality of life metrics."
    - "Industry adoption rates and market trends."

    Generate sources that will provide comprehensive coverage of the topic.
    """

    research_plan = planner_llm.invoke(prompt)

    print(f"Orchestrator generated {len(research_plan.sources)} research sources")

    for i, source in enumerate(research_plan.sources, 1):
        print(f"{i}, {source}")

    print(f"\n Reasoning: {research_plan.reasoning}")
    print("Preparing to dispatch to nodes...")

    return {"sources": research_plan.sources}

def research_worker(state: WorkerState):

    worker_id = state["worker_id"]
    source = state["source"]

    print(f"WORKER {worker_id}: Researching '{source}'...")

    prompt = f"""
    You are a specialized researcher investigating: {state["research_topic"]}

    Your specific focus area: {source}

    Conduct thorough research on this aspect and provides:

    1. KEY FINDINGS (3-5 specific points)
    - What are the most important discoveries or facts ?

    2. DATA & STATISTICS 
    - Relevant numbers, percentages, or quantitative information

    3. INSIGHTS & ANALYSIS 
    - What does this information mean?
    - How does it relate to the broader topic?

    4. SOURCES & CREDIBILITY
    - Types of sources you would consult (academic, industry, goverment, etc.)

    5. IMPLICATIONS
    - Why does this matter for understanding the overall topic?

    Be specific, factual, and provide depth on this particular aspect.
    """

    response = llm.invoke(prompt).content 

    findings = {
        "worker_id": worker_id,
        "source": source,
        "content": response
    }

    print("WORKER {worker_id}: Research Complete\n")

    return {
        "worker_findings": [findings]
    }

def synthesize_report(state: OverallState):
    print("="*70)
    print("SYNTHESIZER: Combining insights from all workers")
    print("="*70)

    print(f"Processing findings from {len(state["worker_findings"])} research sources")

    all_findings = "\n\n" + "="*70 + "\n\n"
    all_findings += "\n\n".join(
        f"RESEARCH AREA {f['worker_id']}: {f['source']}\n"
        f"{'-' * 70}\n"
        f"{f['content']}"
        for f in state["worker_findings"]
    )

    prompt = f"""
    You are synthesizing a comprehensive research report on: {state["research_topic"]}

    You have received detailed findings from {len(state["worker_findings"])}
    Specialized researchers, each investigating a different aspect of this topic.

    RESEARCH FINDINGS:
    {all_findings}

    Create a cohesive, well-structured research report (500-700 words) with the
    following sections:

    1. EXECUTIVE SUMMARY (2-3 sentences)
    - Provide a high-level overview of the key takeaways

    2. INTRODUCTION
    - Context and importance of this topic
    - Scope of this research 

    3. KEY FINDINGS
    - Integrate insights from all research areas
    - Organize thematically rather than by source 
    - Use specific data and examples

    4. ANALYSIS & SYNTHESIS
    - Identify patterns across different research areas
    - Highlight connections and relationships
    - Note any contradictions or tensions 

    5. IMPLICATIONS
    - What do these findings mean?
    - Who is affected and how?

    6. CONCLUSIONS
    - Main takeaways
    - Areas for further research

    IMPORTANT:
    - Write this as a unified, flowing report, NOT as separate sections from each researcher.
    - Integrate findings natually across themes
    - Use specific examples and data from the research
    - Make it professional and authoritatives
    """

    final_report = llm.invoke(prompt).content 

    print("SYNTHESIZER: Final report Complete\n")

    return {"final_report": final_report}


def create_research_workers(state: OverallState):

    print("DISPATCHER: Creating research workers dynamically...")

    return [
        Send("research_worker", {
            "source": source,
            "worker_id": i+1,
            "research_topic": state["research_topic"]
        })
        for i, source in enumerate(state["sources"])
    ]

builder = StateGraph(OverallState)

builder.add_node("orchestrator", plan_research)
builder.add_node("research_worker", research_worker)
builder.add_node("synthesizer", synthesize_report)

builder.add_edge(START, "orchestrator")
builder.add_conditional_edges("orchestrator", create_research_workers, ["research_worker"])
builder.add_edge("research_worker", "synthesizer")
builder.add_edge("synthesizer", END)

graph = builder.compile()

healthcare_topic = "The impact of artificial intelligence on healthcare delivery and patient outcomes"
environmental_topic = "Renewable energy adoption barriers in developing countries."

topic = healthcare_topic

print("="*70)
print(f"Topic: {topic}")
print("="*70)

result = graph.invoke({
    "research_topic": topic,
    "sources": [],
    "worker_findings":[]
})

print("="*70)
print("FINAL SYNTHESIZED RESULT")
print("="*70)
print(result["final_report"])