from autogen import AssistantAgent, UserProxyAgent
import os
import re
from utils import calculate_tokens

def create_one_on_one_group(num_agents, roles, prompts, decision_prompt):
    """
    Creates a one-on-one chat architecture with multiple agents and a moderator

    Args:
        num_agents (int): The number of agents to create.
        roles (list): A list of role names for the agents.
        prompts (list): A list of prompts for each agent.
        decision_prompt (str): The prompt or instructions for the moderator agent.

    Returns:
        tuple: (list of agents, moderator)

    Raises:
        ValueError: If the number of roles or prompts doesn't match the number of agents.
    """
    if len(roles) != num_agents or len(prompts) != num_agents:
        raise ValueError(f"Number of roles ({len(roles)}) and prompts ({len(prompts)}) must match the number of agents ({num_agents})")

    # Configure the AI models
    config_list = [
        {
            "model": os.getenv("LLM_MODEL"),
            "api_key": os.getenv("OPENAI_API_KEY")
        }
    ]

    # Create a configuration for the agents
    agent_config = {
        "config_list": config_list,
        "cache_seed": None,
    }

    # Create the agents
    agents = []
    for i in range(num_agents):
        agents.append(AssistantAgent(
            name=re.sub(r'[^a-zA-Z0-9_-]', '_', roles[i]),
            system_message=f"You are {roles[i]}. {prompts[i]}",
            llm_config=agent_config
        ))

    # Create the moderator
    moderator = AssistantAgent(
        name="Moderator",
        system_message=f"You are the moderator. {decision_prompt}",
        llm_config=agent_config
    )

    return agents, moderator
