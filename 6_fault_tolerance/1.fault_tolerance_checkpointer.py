import time
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver

# Pillar 1 = Checkpointing (State Persistence)
# => Checkpointing is the foundation.
# => After every node executes, LangGraph saves the full graph state to a checkpointer.
# => If something fails later, you can resume from the last saved checkpoint instead of restarting from scratch.

class CrashState(TypedDict):
    input: str 
    step1: str 
    step2: str 
    step3: str 

def step_1(state: CrashState)-> CrashState:
    print("Stap 1 executed!")
    return {"step1": "done", "input": state["input"]}

def step_2(state: CrashState)-> CrashState:
    print("Stap 2 handing...now manually interrupt from the notebook toolbar (STOP button)!")
    time.sleep(30)  # Simulate long-running hand
    return {"step2": "done"}

def step_3(state: CrashState)-> CrashState:
    print("Stap 3 executed!")
    return {"step3": "done"}

graph_builder = StateGraph(CrashState)

graph_builder.add_node("step_1", step_1)
graph_builder.add_node("step_2", step_2)
graph_builder.add_node("step_3", step_3)

graph_builder.add_edge(START, "step_1")
graph_builder.add_edge("step_1", "step_2")
graph_builder.add_edge("step_2", "step_3")
graph_builder.add_edge("step_3", END)

with SqliteSaver.from_conn_string("./my_checkpoints.db") as checkpointer:

    workflow = graph_builder.compile(checkpointer=checkpointer)

    try:
        print("Running graph: please manually interrupt during step 2...")
        workflow.invoke({"input": "start"}, config={"configurable": {"thread_id": "1"}})
    except KeyboardInterrupt:
        print("Kernal manully interrupt (crash simulated!)")

    print(workflow.get_state({"configurable": {"thread_id": "1"}}))

    history = list(workflow.get_state_history({"configurable": {"thread_id": "1"}}))
    print("\n\nHistory: ",history)

    # Re-run to show fault-tolerant resume
    final_state = workflow.invoke(None, {"configurable": {"thread_id": "1"}})
    print(workflow.get_state({"configurable": {"thread_id": "1"}}))

    # Iterate all checkpoints (newest first)
    history = list(workflow.get_state_history({"configurable": {"thread_id": "1"}}))
    print(history)