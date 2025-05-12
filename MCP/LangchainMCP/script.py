import asyncio
from dotenv import load_dotenv
load_dotenv(override=True)
from langchain_groq import  ChatGroq
from mcp_use import MCPAgent , MCPClient
from langchain_google_genai import ChatGoogleGenerativeAI

import  os

mcp_agent_system_message = "\n".join([
    "You are an AI assistant specialized in providing information about apartments in Al Hoceima, Morocco.",
    "Only answer questions related to Al Hoceima; politely decline to respond to queries about other locations or topics.",
    "Ensure your responses are consistent in format and tone.",
    "Do not include links to individual apartment listings.",
    "Provide only the URL of the search page containing the relevant results.",
    "Summarize key details like location, price, and features for each apartment.",
    "Maintain a professional and helpful demeanor."
])

load_dotenv(override=True)
async def run_memory_chat():
          # os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
          os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

          config_file="./browser_mcp.json"
          client=MCPClient.from_config_file(config_file)
          # llm=ChatGroq(model="qwen-qwq-32b")

          llm = ChatGoogleGenerativeAI(
          model="gemini-2.0-flash-001",
          temperature=0.5,
          )

          agent = MCPAgent(
                  llm=llm ,
                  client=client,
                  max_steps=15,
                  memory_enabled=False,
                  verbose=True,
                  system_prompt=mcp_agent_system_message,

          )

          print(await agent.run("i dont have "))

if __name__ == "__main__":
          asyncio.run(run_memory_chat())


          
