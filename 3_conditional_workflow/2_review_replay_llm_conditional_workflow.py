from langgraph.graph import StateGraph, START, END 
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from decouple import config
from pydantic import BaseModel, Field
from typing import TypedDict, Literal

chat_model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=config("GOOGLE_GEMINI_API_KEY"))

class SentimentSchema(BaseModel):
    sentiment: Literal["positive", "negative"] = Field(description="Sentiment of the review")

class DiagnosisSchema(BaseModel):
    issue_type: Literal["UX", "Performance", "Bug", "Support", "Other"] = Field(description="The category of issue mentioned in the review")
    tone: Literal["angry", "frustrated", "disappointed", "calm"] = Field(description='The emotional tone expressed by the user')
    urgency: Literal["low", "medium", "high"] = Field(description='How urgent or critical the issue appears to be')

structured_model = chat_model.with_structured_output(SentimentSchema)
structured_model2 = chat_model.with_structured_output(DiagnosisSchema)

# prompt = 'What is the sentiment of the following review - The software too bad'
# response = structured_model.invoke(prompt)
# print(response.sentiment)


class ReviewState(TypedDict):
    review: str 
    sentiment: Literal["positive", "negative"]
    diagnosis: dict 
    response: str 

graph = StateGraph(ReviewState)

def find_sentiment(state: ReviewState):
    prompt = PromptTemplate(
        template="For the following review find out the sentiment \n {review}",
        input_variables=["review"]
    )
    sentiment_prompt = prompt.format(review=state["review"])
    sentiment = structured_model.invoke(sentiment_prompt).sentiment
    return {"sentiment": sentiment}

def positive_response(state: ReviewState):
    prompt = PromptTemplate(
        template="""Write a warm thank-you message in response to this review:
        \n\n\"{review}\"\n
        Also, kindly ask the user to leave feedback on our website.""",
        input_variables=["review"]
    )
    positive_prompt = prompt.format(review=state["review"])
    positive = chat_model.invoke(positive_prompt).content
    return {"response": positive}

def run_diagnosis(state: ReviewState):
    prompt = PromptTemplate(
        template="Diagnose this negative review:\n\n{review}\n"
        "Return issue_type, tone, and urgency.",
        input_variables=["review"]
    )
    diagnosis_prompt = prompt.format(review=state["review"])
    diagnosis = structured_model2.invoke(diagnosis_prompt)
    return {"diagnosis": diagnosis.model_dump()}

def negative_response(state: ReviewState):
    diagnosis = state["diagnosis"]

    prompt = PromptTemplate(
        template="""You are a support assistant.
        The user had a {issue_type} issue, sounded {tone}, and marked urgency as {urgency}'.
        Write an empathetic, helpful resolution message.""",
        input_variables=["issue_type", "tone", "urgency"]
    )
    negative_prompt = prompt.format(
        issue_type=diagnosis["issue_type"], 
        tone=diagnosis["tone"],
        urgency=diagnosis["urgency"]
    )
    negative = chat_model.invoke(negative_prompt).content
    return {"response": negative}


def check_sentiment(state: ReviewState)->Literal["positive_response", "run_diagnosis"]:
    if state["sentiment"] == "positive":
        return "positive_response"
    else:
        return "run_diagnosis"

graph.add_node("find_sentiment", find_sentiment)
graph.add_node("positive_response", positive_response)
graph.add_node("run_diagnosis", run_diagnosis)
graph.add_node("negative_response", negative_response)

graph.add_edge(START, "find_sentiment")
graph.add_conditional_edges("find_sentiment", check_sentiment)
graph.add_edge("positive_response", END)
graph.add_edge("run_diagnosis", "negative_response")
graph.add_edge("negative_response", END)

workflow = graph.compile()

initial_state={
    'review': "I’ve been trying to log in for over an hour now, and the app keeps freezing on the authentication screen. I even tried reinstalling it, but no luck. This kind of bug is unacceptable, especially when it affects basic functionality."
}
final_state = workflow.invoke(initial_state)
print(final_state)