import os 
from dotenv import load_dotenv
from langchain_core.outputs import llm_result
from langchain_google_genai import ChatGoogleGenerativeAI
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage
from langgraph.prebuilt import ToolNode
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import tools_condition
from tools import tools
from retriever import guest_info_tool
load_dotenv(override=True)


chat = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0.1,
)

tools.append(guest_info_tool) 
chat_with_tools = chat.bind_tools(tools)

class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    llm_result: str 
from langchain_core.messages import SystemMessage
  
def assistant(state: AgentState):
    
    
    system_message = SystemMessage(content="You should  always reformulate the  tool output  into a human readable format and then answer the question.")
    messages = [system_message] + state["messages"]
    
    response = chat_with_tools.invoke(messages)
    
    return {
        "messages": [response],
        "llm_result": response.content
    }

builder = StateGraph(AgentState)

builder.add_node("assistant", assistant)
builder.add_node("tools", ToolNode(tools))

builder.add_edge(START, "assistant")
builder.add_conditional_edges("assistant",  tools_condition)
builder.add_edge("tools", "assistant")
# builder.add_edge("assistant" , END)
alfred = builder.compile()


image_bytes = alfred.get_graph(xray=True).draw_mermaid_png()

with open("langgraph.png", "wb") as f:
    f.write(image_bytes)

print("Diagram saved as 'langgraph_diagram.png'")

response = alfred.invoke({"messages": [HumanMessage(content="What is the weather in Al Hoceima, MA?")] , llm_result:""})
print(response)
print(response['llm_result'])