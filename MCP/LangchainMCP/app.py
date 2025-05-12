import asyncio
from dotenv import load_dotenv
load_dotenv(override=True)
from langchain_groq import  ChatGroq
from mcp_use import MCPAgent , MCPClient

import  os

load_dotenv(override=True)
async def run_memory_chat():
          os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

          config_file="./browser_mcp.json"
          client=MCPClient.from_config_file(config_file)
          llm=ChatGroq(model="qwen-qwq-32b")

          agent = MCPAgent(
                  llm=llm ,
                  client=client,
                  max_steps=15,
                  memory_enabled=True
          )

          try:
                  while True:
                              user_input = input("\nYou: ")
                              if user_input.lower() == "exit":
                                          break
                              response = await agent.run(user_input)
                              print(f"Agent: {response}")
                              if user_input == "clear":
                                          agent.clear_conversation_history()
          except KeyboardInterrupt:
                  print("\nExiting...")
          except Exception as e:
                  print(f"An error occurred: {e}")
          # finally:
          #         await agent.close()
          #         await client.close()


if __name__ == "__main__":
          asyncio.run(run_memory_chat())


          
