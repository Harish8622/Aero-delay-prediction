from langgraph.graph import StateGraph, START, END
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage
from typing import TypedDict

# ---- 1. Define State ----
class ChatState(TypedDict):
    messages: list  # stores chat history

# ---- 2. Setup the LLM ----
llm = ChatOpenAI(model="gpt-3.5-turbo")  # or swap in a free model

# ---- 3. Define Node Functions ----
def chatbot_node(state: ChatState):
    """Chat node: takes user input + history, returns assistant response."""
    history = state["messages"]
    user_msg = history[-1]  # last message is the latest user input
    
    # Call the LLM
    response = llm([HumanMessage(content=user_msg)])
    
    # Append response to history
    history.append(response.content)
    return {"messages": history}

def start_node(state: ChatState):
    """Start node: just passes through initial state."""
    return state

def end_node(state: ChatState):
    """End node: simply returns the final state."""
    return state

# ---- 4. Build the Graph ----
graph = StateGraph(ChatState)

# Add nodes
graph.add_node("start", start_node)
graph.add_node("chatbot", chatbot_node)
graph.add_node("end", end_node)

# Define edges
graph.add_edge(START, "start")        # Start → start_node
graph.add_edge("start", "chatbot")    # start_node → chatbot
graph.add_edge("chatbot", "end")      # chatbot → end_node
graph.add_edge("end", END)            # end_node → END

# Compile graph
app = graph.compile()

# ---- 5. Run the Graph ----
if __name__ == "__main__":
    user_input = input("You: ")
    result = app.invoke({"messages": [user_input]})
    print("Bot:", result["messages"][-1])

    # neeed to add tools so initially it asks customer for flight details
    # then uses tools to determine the required params
    # this is conditional edge depending on if there is enough info
    # use pydanbtic to ensure correct params
    # then uses tool to call inference
    # then returns to user