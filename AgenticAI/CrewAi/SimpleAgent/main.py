from crewai import Agent, Task, Crew , LLM
import os
from dotenv import load_dotenv
load_dotenv(override=True)
os.environ["SERPER_API_KEY"] = os.getenv("SERPER_API_KEY")


os.environ["GEMINI_API_KEY"] = os.getenv("GOOGLE_API_KEY")

llm=LLM(
        model="gemini/gemini-1.5-flash",
        temperature=0.5,
        # verbose=True,    
        )

from crewai_tools import ScrapeWebsiteTool, SerperDevTool, YoutubeVideoSearchTool,PDFSearchTool

search_tool = SerperDevTool()
scrape_tool = ScrapeWebsiteTool()
youtube_tool = YoutubeVideoSearchTool(
     config=dict(
        llm=dict(
            provider="google", # or google, openai, anthropic, llama2, ...
            config=dict(
                model="gemini-1.5-flash",
                temperature=0.5,
                
                # top_p=1,
                # stream=true,
            ),
        ),
        embedder=dict(
            provider="google", # or openai, ollama, ...
            config=dict(
                model="models/embedding-001",
                task_type="retrieval_document",
                title="Embeddings",
            ),
        ),
    )
)

data_analyst_agent = Agent(
    role="CIH BAnk Agent",
    goal="Answer a customer ({question}) according to the ({url})",
    backstory="Specializing in guidance consumers, this agent "
              "uses the ({url})  cih bank website which can be searched online"
              "to provide crucial insights to  help costumers make informed decisions "
              "and navigate the complex world of banking services. ",
    verbose=True,
    allow_delegation=False,
    tools = [scrape_tool, search_tool] ,
    llm=llm
)

youtube_agent = Agent(
    role="Youtube Video Search Agent",
    goal="the user ask  ({question}) according to the ({url})",
    backstory="This agent help  the user to  get  the answer from youtube video  ",
    verbose=True,
    allow_delegation=False,
    tools = [youtube_tool  ],
    llm=llm
)


# Task for Data Analyst Agent: Analyze Market Data
data_analysis_task = Task(
    description=(
        "Read and understand the cih bank website. When user ask ({question}), provide them answer promptly according to provided url ({url}) if  the  url not contain a good result , search the internet for the answer , u should also inclde a link to the source "
    ),
    expected_output=(
        "answer from user({question}) "
    ),  
    agent=data_analyst_agent,
)

youtube_task = Task(
    description=(
        "Read and understand the video. When user ask ({question}), provide them answer promptly according to provided url ({url})  "
    ),
    expected_output=(
        "answer from user({question}) "
    ),
    agent=youtube_agent,
)

from crewai import Crew, Process
from langchain_openai import ChatOpenAI

# Define the crew with agents and tasks
refund_crew = Crew(
    agents=[youtube_agent ],

    tasks=[youtube_task ],
    process=Process.hierarchical,
    verbose=True ,
    manager_llm=llm
)
refund_crew_inputs = {
'question': 'give me  an answer  to  this  question  ',
'url' :'https://www.youtube.com/watch?v=pkfrtRyp1cs'
}
### this execution will take some time to run
result = refund_crew.kickoff(inputs=refund_crew_inputs)























