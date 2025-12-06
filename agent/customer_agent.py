import sys
import os

import requests
import holidays
from dotenv import load_dotenv
from typing import Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool

from typing_extensions import TypedDict
from pydantic import BaseModel, Field, field_validator, ValidationError
import pandas as pd
import json
from datetime import datetime
from agent.model_config import llm
from agent.helpers.tools import tools

# define state
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]



# define call model node
def agent_node(state: AgentState) -> AgentState:
    """
    calls the model with bound tools
    """

    response = client_with_tools.invoke(state["messages"])
    # state["messages"].append(AIMessage(content=response.content))
    return {"messages": state["messages"] + [response]}

# define conditional edge logic
def should_continue(state: AgentState) -> bool:
    """
    determine if we should continue to tools or the END node
    """

    last_message = state["messages"][-1]

    # check if last message has tool calls

    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    else:   
        return END


# define tools node


client_with_tools = llm.bind_tools(tools)
tool_node = ToolNode(tools)


graph = StateGraph(AgentState)

graph.add_node("agent", agent_node)
graph.add_node("tools", tool_node)

graph.add_edge(START, "agent") # always go from start to agent
# next conditional logic, only tools or end if no further tool needed
graph.add_conditional_edges(
    "agent",
    should_continue,
    {"tools": "tools",
    END: END
    }
)

# must define loop back to agent 
graph.add_edge("tools", "agent")

app = graph.compile()


# define the system message

system_message = """

you are also capable of assisting with flight route confirmations only using the tools provided

<<YOU MUST ALWAYS USE TOOLS WHENEVER POSSIBLE, DO NOT MAKE UP ANSWERS EVEB IF IT SEEMS OBVIOUS
IF A TOOL CAN BE USED, USE IT, IF SOMETHING IS OUT OF SCOPE, SAY YOU CANNOT HELP. DO NOT MAKE UP ANSWERS>>

<<<IF THE USER MENTIONS YOU ARE IN "TEST MODE" OR INFERS THIS, RETURN YOUR ANSWER BASED ON FOLLOWING INSTRUCTIONS>>>

<<<IF THE USER MENTIONS FLIGHT QUERY YOU MUST ATTEMPT TO GET DISTANCE BETWEEN ORIGIN AND DESTINATION and TEMPORAL FEATURES and WEATHER CONDITIONS>>>

<<<INSTRUCTIONS FOR TEST MODE:
    - ALways start by listing the tools you have access to like this:
        <<TOOLS AVAILABLE>>
          [toolx, tooly]
    - For each tool mention why or why you did not use it and if you use a tool, show the input next to it and the output from the tool:
    - You must say exactly what you gave the tool and what it returned like this:
        <<TOOL USAGE>>
          [toolx: used because of xyz, input: {input}, output: {output}]
    - Finally return your final answer:
        <<FINAL ANSWER>>
          [your final answer here]
    >>> End of test mode instructions


<<< IF THE USER DOES NOT MENTION TEST MODE, ANSWER HOW YOU SEE FIT, JUST DO NOT MAKE UP TOOL USAGE>>>

If user tries to leave with exit or quit leave a nice goodbye message and end the conversation.

"""

conversation_state = {
    "messages": []}

conversation_state["messages"].append(SystemMessage(content=system_message))

user_message = ""

while user_message.lower() not in ["exit", "quit"]:
    user_message = input("I am a sentient calculator. Ask me anything!:")
    conversation_state["messages"].append(HumanMessage(content= user_message))
    result_state = app.invoke(conversation_state)
    response = result_state["messages"][-1]
    print(f"AI: {response.content}")
    conversation_state = result_state                                        
