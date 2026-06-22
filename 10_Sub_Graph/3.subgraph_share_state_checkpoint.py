from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import InMemorySaver
from decouple import config

class ParentState(TypedDict):
    question: str 
    answer_eng: str 
    answer_hin: str 

parent_llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=config("GOOGLE_GEMINI_API_KEY"))
subgraph_llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=config("GOOGLE_GEMINI_API_KEY"))

def translate_text(state: ParentState):

    prompt = f"""
Translate the following text to Hindi.
Keep it natural and clear. Do not add extra content.

Text:
{state["answer_eng"]}
""".strip()
    
    translated_text = subgraph_llm.invoke(prompt).content

    return {'answer_hin': translated_text}

subgraph_builder = StateGraph(ParentState)

subgraph_builder.add_node('translate_text', translate_text)

subgraph_builder.add_edge(START, 'translate_text')
subgraph_builder.add_edge('translate_text', END)

# Yahan checkpointer NAHI diya — kyunki parent isko handle karega
subgraph = subgraph_builder.compile()

def generate_answer(state: ParentState):

    answer = parent_llm.invoke(f"You are a helpful assistant. Answer clearly.\n\nQuestion: {state['question']}").content
    return {'answer_eng': answer}

parent_builder = StateGraph(ParentState)

parent_builder.add_node("answer", generate_answer)
parent_builder.add_node("translate", subgraph)

parent_builder.add_edge(START, 'answer')
parent_builder.add_edge('answer', 'translate')
parent_builder.add_edge('translate', END)

checkpointer = InMemorySaver()

# Checkpointer SIRF parent compile karte time pass kiya
graph = parent_builder.compile(checkpointer=checkpointer)

config = {"configurable": {"thread_id": "trip-101"}}
result = graph.invoke({'question': 'What is quantum physics'}, config=config)

print(result)   

# Note:
# Kya ho raha hai internally: Parent graph apna checkpoint save karta hai 
# AUR subgraph ke har step ka bhi checkpoint save hota hai — automatically, 
# same checkpointer instance use karke, lekin alag namespace mein.


# State Inspect karna — Namespace Samajhna
# Yahi sabse confusing part hai logon ke liye. Subgraph ke checkpoints ek nested namespace (checkpoint_ns) mein store hote hain.

# Parent level state dekho
state = graph.get_state(config)
print("Parent state:", state.values)
print("Next node:", state.next)

# Subgraph ke andar ka state dekhna hai? get_state_history use karo
print("\n--- Full history (parent + subgraph) ---")
for snapshot in graph.get_state_history(config):
    ns = snapshot.config["configurable"].get("checkpoint_ns", "")
    print(f"Namespace: '{ns}' | Next: {snapshot.next} | Values: {snapshot.values.get('research_data', '')}")