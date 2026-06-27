from langgraph.store.memory import InMemoryStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from decouple import config

embedding_model = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001", google_api_key=config("GOOGLE_GEMINI_API_KEY"))

store = InMemoryStore(index={"embed": embedding_model, "dims": 1536})

namespace = ("users", "u1")

store.put(namespace, "1", {"data": "User prefers concise answers over long explanations"})
store.put(namespace, "2", {"data": "User likes examples in Python"})
store.put(namespace, "3", {"data": "User usually works late at night"})
store.put(namespace, "4", {"data": "User prefers dark mode in applications"})
store.put(namespace, "5", {"data": "User is learning machine learning"})
store.put(namespace, "6", {"data": "User dislikes overly theoretical explanations"})
store.put(namespace, "7", {"data": "User prefers step-by-step reasoning"})
store.put(namespace, "8", {"data": "User is based in India"})
store.put(namespace, "9", {"data": "User likes real-world analogies"})
store.put(namespace, "10", {"data": "User prefers bullet points over paragraphs"})

items = store.search(namespace, query="What is the currently learning", limit=1)

for item in items:
    print(item.value)

items = store.search(namespace, query="What are user's preferences", limit=3)

for item in items:
    print(item.value)