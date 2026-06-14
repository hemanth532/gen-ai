from crewai import Agent, Process,Task,Crew
from langchain_openai import ChatOpenAI
from openai import OpenAI
import os
from dotenv import load_dotenv
load_dotenv()


print("API_KEY:" , os.environ["OPENAI_API_KEY"]) 

# Ensure the API key is loaded correctly
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")  # Replace with your actual API key




llm  = "gpt-4o-mini"

# Agents

researcher = Agent(
    role="Market Researcher",
    goal="Find out the top competitors for Tesla",
    backstory="Expert in EV market analysis with 10 years of experience.",
    llm="gpt-4o-mini",
    verbose=True
)

analyst = Agent(
    role="Business Analyst",
    goal="Analyze compititors' strengths and weaknesses",
    backstory="Strategy consusltant with a focus on competitive analysis in the automotive industry.",
    llm="gpt-4o-mini",
    verbose=True
)

summarizer = Agent(
    role="Report Writer",
    goal="Create a comprehensive report on the competitive landscape",
    backstory="Professiona; business writer with experience in synthesizing complex information into clear reports.",
    llm="gpt-4o-mini",
    verbose=True
)


# Tasks
task = Task(
    description="Conduct a competitive analysis for Tesla in the EV market.",
    expected_output="A detailed report outlining Tesla's main competitors, their strengths and weaknesses, and market trends.",
    agent=researcher
)

# Crew
crew=Crew(
    agents=[researcher,analyst,summarizer],
    tasks=[task],
    process=Process.sequential,
    verbose=True
)

# Execute the crew
result=crew.kickoff()

print("\n\n Final Result: \n")
print("Result:", result)