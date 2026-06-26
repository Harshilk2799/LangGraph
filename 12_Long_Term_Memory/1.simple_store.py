from langgraph.store.memory import InMemoryStore

# Creating a store
store = InMemoryStore()

# Creating a namespace
namespace = ("user", "u1")

# Creating a Memories

# Adding memories
store.put(namespace, "1", {"data": "User likes pizza"})
store.put(namespace, "2", {"data": "User prefers dark mode"})

# Another namespace
namespace2 = ("user", "u2")

# Adding Memories
store.put(namespace2, "1", {"data": "User likes pasta"})
store.put(namespace2, "2", {"data": "User prefers grid style navigation"})


# Retriving memories
print(store.get(namespace, "2"))

# Retrieving all the memories
items = store.search(namespace2)

for item in items:
    print(item.value)

