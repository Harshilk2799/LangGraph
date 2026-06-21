from typing import TypedDict, Optional
from langgraph.graph import StateGraph, START, END 
from langgraph.types import interrupt, Command
from langgraph.checkpoint.memory import MemorySaver

class ContentState(TypedDict):
    customer_query: str 
    ai_draft: str 
    final_content: Optional[str]
    was_edited: Optional[str]
    edit_count: Optional[int]

def generate_draft(state: ContentState):
    query = state["customer_query"]

    draft = f"Dear Customer,\n\nThank you for reaching out about '{query}'. " \
            f"We're looking into this and will resolve it within 24 hours.\n\nRegards,\nSupport Team"
    
    return {"ai_draft": draft, "edit_count": 0}

def review_and_edit(state: ContentState):
    response = interrupt({
        "action_needed": "review_draft",
        "ai_draft": state["ai_draft"],
        "instructions": "Edit the content below if needed, then submit."
    })

    # response mein human ka final edited text aayega
    edited_text = response["edited_content"]
    print("Edited Text: ", edited_text)

    # Check karo ki human ne actually change kiya ya same rakha
    was_changed = edited_text.strip() != state["ai_draft"].strip()
    print("Was Changed: ", was_changed)

    return {
        "final_content": edited_text,
        "was_edited": was_changed,
        "edit_count": state.get("edit_count", 0) + (1 if was_changed else 0)
    }

def send_content(state: ContentState):
    print("\n--- SENDING TO CUSTOMER ---")
    print(state["final_content"])
    print(f"\nWas edited by human: {state['was_edited']}")
    return {}

graph = StateGraph(ContentState)

graph.add_node("generate_draft", generate_draft)
graph.add_node("review_and_edit", review_and_edit)
graph.add_node("send_content", send_content)

graph.add_edge(START, "generate_draft")
graph.add_edge("generate_draft", "review_and_edit")
graph.add_edge("review_and_edit", "send_content")
graph.add_edge("send_content", END)

checkpointer = MemorySaver()
app = graph.compile(checkpointer=checkpointer)

config = {"configurable": {"thread_id": "content-001"}}

result = app.invoke(
    {"customer_query": "My order hasn't arrived yet"},
    config=config
)

print(result["__interrupt__"])

# Case 1: Human Edits the Content

# edited_version = (
#     "Dear Customer,\n\n"
#     "We sincerely apologize for the delay with your order. Our logistics team "
#     "has confirmed it will arrive within 48 hours, and we're including a 10% "
#     "discount code for the inconvenience: SORRY10\n\n"
#     "Warm regards,\nSupport Team"
# )

# result = app.invoke(
#     Command(resume={"edited_content": edited_version}),
#     config=config
# )
# print(result)


# Case 2: Human Draft as-is Accept Kare

result = app.invoke(
    Command(resume={"edited_content": app.get_state(config).values["ai_draft"]}),
    config=config
)
print("=====================")
print(result)