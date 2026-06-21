from typing import Optional, TypedDict, Literal
from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt, Command
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3
from pprint import pprint

class ExpenseState(TypedDict):
    employee: str 
    amount: float 
    reason: str 
    status: Optional[str]   #  "approved/rejected/auto_approved"
    reviewer_comment: Optional[str]

def check_expense(state: ExpenseState):
    print(f"Checking expense: ₹{state['amount']} by {state['employee']}")

    if state["amount"] <= 5000:
        # Chhoti amount — auto approve, human ki zaroorat nahi
        return {"status": "auto_approved"}
    
    # Bada amount — human approval chahiye, isliye kuch nahi return
    # routing decide karega ki human_approval node pe jana hai
    return {}

def human_approval(state: ExpenseState):
    decision = interrupt({
        "employee": state["employee"],
        "amount": state["amount"],
        "reason": state["reason"],
        "ask": "Approve or Reject this expense?"
    })

    # decision yahan ek dict hoga jo human/resume ne bheja
    if decision["action"] == "approve":
        return {
            "status": "approved",
            "reviewer_comment": decision.get("comment", "")
        }
    else:
        return {
            "status": "rejected",
            "reviewer_comment": decision.get("comment", "Rejected by reviewer")
        }
    
def finalize(state: ExpenseState):
    print(f"\n--- FINAL RESULT ---")
    print(f"Employee: {state['employee']}")
    print(f"Amount: ₹{state['amount']}")
    print(f"Status: {state['status']}")
    if state.get("reviewer_comment"):
        print(f"Comment: {state['reviewer_comment']}")
    return {}

def route_after_check(state: ExpenseState)-> Literal["human_approval", "finalize"]:
    if state.get("status") == "auto_approved":
        return "finalize"
    return "human_approval"

builder = StateGraph(ExpenseState)

builder.add_node("check_expense", check_expense)
builder.add_node("human_approval", human_approval)
builder.add_node("finalize", finalize)

builder.add_edge(START, "check_expense")
builder.add_conditional_edges("check_expense", route_after_check)
builder.add_edge("human_approval", "finalize")
builder.add_edge("finalize", END)

config = {"configurable": {"thread_id": "expense-001"}}

conn = sqlite3.connect(database='chatbot.db', check_same_thread=False)

# Checkpointer
checkpointer = SqliteSaver(conn=conn)

graph = builder.compile(checkpointer=checkpointer)

result = graph.invoke(
    {
        "employee": "Harshil",
        "amount": 15000,
        "reason": "Conference travel ticket"
    },
    config=config
)

pprint(result)

# Case 1: Manager approve karta hai
# result = graph.invoke(
#     Command(resume={"action": "approve", "comment": "Looks valid, approved"}),
#     config=config
# )   
# pprint(result)

# Case 2: Manager reject karta hai
# result = graph.invoke(
#     Command(resume={"action": "reject", "comment": "Not a business expense"}),
#     config=config
# )
# pprint(result)


# Case 3: Auto-approve Case 
result = graph.invoke(
    {"employee": "Priya", "amount": 2000, "reason": "Office stationery"},
    config={"configurable": {"thread_id": "expense-002"}}
)
pprint(result)