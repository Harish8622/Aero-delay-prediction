from typing import Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, SystemMessage

from typing_extensions import TypedDict
from src.agent_core import (
    llm,
    tools,
    agent_node_prompt,
)



def build_agent_app():
    class AgentState(TypedDict):
        messages: Annotated[list, add_messages]

    def agent_node(state: AgentState) -> AgentState:
        """
        calls the model with bound tools
        """
        response = client_with_tools.invoke(state["messages"])
        return {"messages": state["messages"] + [response]}

    def should_continue(state: AgentState) -> bool:
        """
        determine if we should continue to tools or the END node
        """
        last_message = state["messages"][-1]
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "tools"
        return END

    client_with_tools = llm.bind_tools(tools)
    tool_node = ToolNode(tools)

    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node)
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
    graph.add_edge("tools", "agent")
    return graph.compile()


if __name__ == "__main__":
    app = build_agent_app()
    conversation_state = {"messages": [SystemMessage(content=agent_node_prompt)]}
    user_message = ""

    while user_message.lower() not in ["exit", "quit"]:
        user_message = input("I am here to give you insights on your upcoing flight!  :  ")
        conversation_state["messages"].append(HumanMessage(content=user_message))
        result_state = app.invoke(conversation_state)
        response = result_state["messages"][-1]
        print(f"\n AI: {response.content} \n")
        conversation_state = result_state
