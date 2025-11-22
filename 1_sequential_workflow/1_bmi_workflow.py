from langgraph.graph import StateGraph, START, END
from typing import TypedDict

# 1. Define State
class BMIState(TypedDict):
    weight_kg: float
    height_m: float
    bmi: float

# 2. Define your graph 
graph = StateGraph(BMIState)

def calculate_bmi(state: BMIState) -> BMIState:
    weight = state["weight_kg"]
    height = state["height_m"]
    
    bmi = weight / (height ** 2)
    state["bmi"] = round(bmi, 2)
    return state

# 3. add nodes to your graph
graph.add_node("calculate_bmi", calculate_bmi)

# 4. add edges to your graph 
graph.add_edge(START, "calculate_bmi")
graph.add_edge("calculate_bmi", END)

# 5. compile the graph
workflow = graph.compile()

# 6. execute the graph
initial_state = {"weight_kg": 80, "height_m": 1.73}
final_state = workflow.invoke(initial_state)
print(final_state)