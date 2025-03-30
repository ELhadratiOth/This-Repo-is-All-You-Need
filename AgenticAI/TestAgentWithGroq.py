from crewai import Agent, Task, Crew, LLM

# Initialize Large Language Model (LLM) of your choice (see all models on our Models page)
llm = LLM(model="groq/llama-3.1-8b-instant")

# Create your CrewAI agents with role, main goal/objective, and backstory/personality
summarizer = Agent(
    role='Documentation Summarizer', # Agent's job title/function
    goal='Create concise summaries of technical documentation', # Agent's main objective
    backstory='Technical writer who excels at simplifying complex concepts', # Agent's background/expertise
    llm=llm, # LLM that powers your agent
    verbose=True # Show agent's thought process as it completes its task
)

translator = Agent(
    role='Technical Translator',
    goal='Translate technical documentation to other languages',
    backstory='Technical translator specializing in software documentation',
    llm=llm,
    verbose=True
)

# Define your agents' tasks
summary_task = Task(
    description='Summarize this {input}',
    expected_output="A clear, concise summary of the hook's functionality",
    agent=summarizer # Agent assigned to task
)

translation_task = Task(
    description='Translate this : {input} to french',
    expected_output="french translation of the hook documentation",
    agent=translator,
)

# Create crew to manage agents and task workflow
crew = Crew(
    agents=[summarizer, translator], # Agents to include in your crew
    tasks=[summary_task, translation_task], # Tasks in execution order
    verbose=True ,
    manager_llm=llm ,
    
    process=Process.hierarchical,

)

result = crew.kickoff(inputs={'input' :"i want y  to  translate this : hello world" })
print(result)
