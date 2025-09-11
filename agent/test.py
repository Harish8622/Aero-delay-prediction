print("starting test.py")
import os
print("importing packages")
from dotenv import load_dotenv
print("imported dotenv")
from typing import Annotated
from typing_extensions import TypedDict
from langchain_openai import ChatOpenAI
print("imported Annotated")
from langgraph.graph import StateGraph, START, END

from langgraph.graph.message import add_messages


# Load API key
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Initialize OpenAI chat model
llm = ChatOpenAI(model="gpt-3.5-turbo")

# Define graph state
class State(TypedDict):
    messages: Annotated[list, add_messages]

# Create graph builder
graph_builder = StateGraph(State)

# Define chatbot node
def chatbot_response(state: State):
    user_message = state["messages"][-1]["content"]
    response = llm.invoke(user_message)
    return {"messages": [{"role": "assistant", "content": response.content}]}

# Add chatbot node to graph
graph_builder.add_node("chatbot", chatbot_response)
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)

# Compile graph
chatbot_agent = graph_builder.compile()

# Run chatbot
while True:
    user_input = input("\nYou: ")
    if user_input.lower() in ["exit", "quit"]:
        print("👋 Goodbye!")
        break

    initial_state = {"messages": [{"role": "user", "content": user_input}]}
    state = chatbot_agent.invoke(initial_state)
    print("🤖 Bot:", state["messages"][-1]["content"])