from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver
from typing import TypedDict, Annotated
import operator

class AgentState(TypedDict):
    messages: Annotated[list[str], operator.add]
    count: int 

def node_a(state: AgentState):
    print(f"[Node A] Count = {state['count']}")
    return {
        "messages": ["Node A executed!"],
        "count": state["count"] + 1
    }

def node_b(state: AgentState):
    print(f"[Node B] Count = {state['count']}")
    return {
        "messages": ["Node B executed"],
        "count": state["count"] + 1
    }

def node_c(state: AgentState):
    print(f"[Node C] Count = {state['count']}")
    return {
        "messages": ["Node C executed"],
        "count": state["count"] + 1
    }

with SqliteSaver.from_conn_string("./my_checkpoints.db") as checkpointer:
    builder = StateGraph(AgentState)

    builder.add_node("node_a", node_a)
    builder.add_node("node_b", node_b)
    builder.add_node("node_c", node_c)

    builder.add_edge(START, "node_a")
    builder.add_edge("node_a", "node_b")
    builder.add_edge("node_b", "node_c")
    builder.add_edge("node_c", END)

    graph = builder.compile(checkpointer=checkpointer)

    # Normal Run

    # Thread config — ye har run ke liye unique hona chahiye
    config = {"configurable": {"thread_id": "thread-42"}}

    # Initial state ke saath invoke karo
    result = graph.invoke(
        {"messages": [], "count": 0},
        config=config
    )

    print("Final State:", result)


    # get_state_history()
    # history = list(graph.get_state_history(config))

    # print(f"Total checkpoints: {len(history)}\n")

    # for i, checkpoint in enumerate(history):
    #     print(f"--- Checkpoint {i} ---")
    #     print(f"  checkpoint_id : {checkpoint.config['configurable']['checkpoint_id']}")
    #     print(f"  next node     : {checkpoint.next}")        # Aage kya chalega
    #     print(f"  state.count   : {checkpoint.values['count']}")
    #     print(f"  state.messages: {checkpoint.values['messages']}")
    #     print()


    # Replay (Kisi Past Checkpoint Se Wapas Run Karo)

    history = list(graph.get_state_history(config))

    target_checkpoint = history[1]
    target_config = target_checkpoint.config

    print("Replaying from checkpoint:")
    print(f"  State: {target_checkpoint.values}")
    print(f"  Next : {target_checkpoint.next}")

    replay_result = graph.invoke(None, config=target_config)

    print("\nResult after replay:", replay_result)
