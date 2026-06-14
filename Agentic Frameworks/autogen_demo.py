from autogen import AssistantAgent, UserProxyAgent
import os
from dotenv import load_dotenv

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

config_list = [
    {
        "model": "gpt-4o-mini",
        "api_key": os.environ["OPENAI_API_KEY"],
        "temperature": 0.1
    }
]

assistant =AssistantAgent(
    name="assistant",
    llm_config={"config_list": config_list}
)

user_proxy = UserProxyAgent(
    name="user",
    llm_config={"config_list": config_list},
    code_execution_config=False
)

user_proxy.initiate_chat(assistant, message="Analyze an idea and bring up an execution plan . The Idea is to start an electric fleet managemnt system in India")