from langgraph.graph import StateGraph, START, END 
from typing import TypedDict
from decouple import config
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate

class BatsmanState(TypedDict):
    runs: int 
    balls: int 
    fours: int 
    sixes: int 

    strike_rate: float
    ball_per_boundary: float
    boundary_percent: float
    summary: str

chat_model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=config("GOOGLE_GEMINI_API_KEY"))


def calculate_strike_rate(state: BatsmanState):
    sr = (state["runs"]/state["balls"])*100
    # Below return call partial update for parallel workflow otherwise error
    return {"strike_rate": sr}

def calculate_ball_per_boundary(state: BatsmanState):
    bpb = state["balls"]/(state["fours"] + state["sixes"]) 
    state["ball_per_boundary"] = bpb
    return {"ball_per_boundary": bpb}

def calculate_boundary_percent(state: BatsmanState):
   boundary_per = (((state["fours"] * 4) + (state["sixes"] * 6)) / state["runs"])*100
   return {"boundary_percent": boundary_per}

def summary(state: BatsmanState):
    strick_rate = state["strike_rate"]
    bpb = state["ball_per_boundary"]
    boundary_per = state["boundary_percent"]

    prompt = PromptTemplate(
        template="""
        Strike rate - {strick_rate} \n
        Balls per boundary - {bpb} \n
        Boundary percent - {boundary_per}
    """,
    input_variables=["strick_rate", "bpb", "boundary_per"]
    )

    summary_prompt = prompt.format(strick_rate=strick_rate, bpb=bpb, boundary_per=boundary_per)
    summary = chat_model.invoke(summary_prompt).content
    return {"summary": summary}

graph = StateGraph(BatsmanState)

graph.add_node("calculate_strike_rate", calculate_strike_rate)
graph.add_node("calculate_ball_per_boundary", calculate_ball_per_boundary)
graph.add_node("calculate_boundary_percent", calculate_boundary_percent)
graph.add_node("summary", summary)

graph.add_edge(START, "calculate_strike_rate")
graph.add_edge(START, "calculate_ball_per_boundary")
graph.add_edge(START, "calculate_boundary_percent")
graph.add_edge("calculate_strike_rate", "summary")
graph.add_edge("calculate_ball_per_boundary", "summary")
graph.add_edge("calculate_boundary_percent", "summary")
graph.add_edge("summary", END)

workflow = graph.compile()

initial_state = {"runs": 100, "balls": 50, "fours": 6, "sixes": 4}
final_state = workflow.invoke(initial_state)
print(final_state)