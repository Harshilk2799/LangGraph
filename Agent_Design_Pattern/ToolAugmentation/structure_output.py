from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field 
from typing import List
from decouple import config
import json 

llm = ChatOpenAI(
    api_key=config("OPENROUTER_API_KEY"),
    model="openai/gpt-4o-mini",
    base_url="https://openrouter.ai/api/v1")

class ProductReview(BaseModel):
    """Structured product review analysis."""
    product_name: str = Field(description="Name of the product.")
    sentiment: str = Field(description="Overall sentiment: positive, negative, or neutral")
    rating: int = Field(description="Rating from 1-5", ge=1, le=5)
    pros: List[str] = Field(description="List of positive aspects")
    cons: List[str] = Field(description="List of negative aspects")
    summary: str = Field(description="Bried summary of the review.")

structured_llm = llm.with_structured_output(ProductReview)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a product review analyzer, Extract structured information from reviews."),
    ("user", "{review_text}")
])

chain = prompt | structured_llm

review_text = """
I bought the SonicWave X200 wireless earbuds last month. The sound quality
is excellent and the battery lasts a full day, which I love. However,
the touch controls are overly sensitive and the carrying case feels cheap.
Overall I'm satisfied but wouldn't call it perfect. I'd give it a 4 out of 5.
"""

result = chain.invoke({"review_text": review_text})

print(json.dumps(result.model_dump(), indent=2))