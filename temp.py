import autogen
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure the AI models
config_list = [
    {
        "model": "gpt-4-mini",
        "api_key": os.getenv("OPENAI_API_KEY")
    }
]

llm_config = {
    "request_timeout": 600,
    "seed": 42,
    "config_list": config_list,
    "temperature": 0
}

# Create assistant agents
assistant = autogen.AssistantAgent(
    name="Assistant",
    llm_config=llm_config,
    system_message="You are a helpful AI assistant."
)

coder = autogen.AssistantAgent(
    name="Coder",
    llm_config=llm_config,
    system_message="You are an AI specialized in writing Python code."
)

# Create a human proxy agent
human_proxy = autogen.UserProxyAgent(
    name="Human",
    human_input_mode="TERMINATE",
    max_consecutive_auto_reply=10,
    is_termination_msg=lambda x: x.get("content", "").rstrip().endswith("TERMINATE"),
    code_execution_config={"work_dir": "coding"},
    llm_config=llm_config,
    system_message="You are a human user interacting with AI agents."
)

# Create a group chat
groupchat = autogen.GroupChat(
    agents=[human_proxy, assistant, coder],
    messages=[],
    max_round=50
)

# Create a group chat manager
manager = autogen.GroupChatManager(
    groupchat=groupchat,
    llm_config=llm_config
)

# Start the conversation
human_proxy.initiate_chat(
    manager,
    message="Let's work on a Python project to create a simple web scraper."
)