from pydantic import BaseModel

class AirbnbSearchInput(BaseModel):
    user_query: str



from fastapi import FastAPI
from fastapi.responses import JSONResponse
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from fastapi_mcp import  
from fastapi_mcp.agent import MCPAgent

app = FastAPI()

# Load environment variables
os.environ["GOOGLE_API_KEY"] = os.getenv("GEMINI_API_KEY")
import google.generativeai as genai

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
# Define the system prompt
mcp_agent_system_message = """
You are a helpful assistant that helps users search for Airbnb apartments.
"""

@app.post("/search_airbnb")
async def search_airbnb(input: AirbnbSearchInput):
    # Initialize the MCP client
    client = MCPClient.from_config_file("./browser_mcp.json")

    # Initialize the language model
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash-001",
        temperature=0.5,
    )

    # Create the MCP agent
    agent = MCPAgent(
        llm=llm,
        client=client,
        max_steps=15,
        memory_enabled=False,
        verbose=True,
        system_prompt=mcp_agent_system_message,
    )

    # Execute the agent with the user query
    if hasattr(agent, 'arun'):
        response = await agent.arun(input.user_query)
    else:
        response = await agent.run(input.user_query)

    return JSONResponse(content={"result": response})


from fastapi_mcp import FastApiMCP

# Initialize the MCP server
mcp = FastApiMCP(
    app,
    name="Airbnb Search MCP",
    description="MCP server for Airbnb apartment search",
    base_url="http://localhost:8000",  # Replace with your actual base URL
    include_operations=["search_airbnb"]  # Expose only the custom endpoint
)

# Mount the MCP server
mcp.mount()



if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)





