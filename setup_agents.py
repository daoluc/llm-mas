from autogen import AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager, runtime_logging
import os
import prompt_store
import role_store
import re
import pandas as pd
import json

def get_log(dbname="logs.db", table="chat_completions"):
    import sqlite3

    con = sqlite3.connect(dbname)
    query = f"SELECT * from {table}"
    cursor = con.execute(query)
    rows = cursor.fetchall()
    column_names = [description[0] for description in cursor.description]
    data = [dict(zip(column_names, row)) for row in rows]
    con.close()
    return data

def str_to_dict(s):
    return json.loads(s)


def create_group_chat(num_debaters, roles, prompt, decision_prompt, debate_rounds=2):
    """
    Creates a debate group with multiple agents

    Args:
        num_debaters (int): The total number of agents in the group chat, excluding the moderator.
        roles (list): A list of role names for the agents. If None, all agents will be named as debaters.
        prompt (str): The main prompt or instructions for the debater agents.
        decision_prompt (str): The prompt or instructions for the moderator agent.
        debate_rounds (int, optional): The number of rounds each agent speaks before the moderator concludes. Defaults to 2.

    Returns:
        GroupChatManager

    Raises:
        ValueError: If the number of roles provided doesn't match the number of agents.
    """    

    # Configure the AI models
    config_list = [
        {
            "model": "gpt-4o-mini",
            "api_key": os.getenv("OPENAI_API_KEY")
        }
    ]


    # Create a configuration for the agents
    agent_config = {
    	"config_list": config_list,
    	"cache_seed": None,
    }
    total_expect_messages = num_debaters * debate_rounds + 2 # Each agent speaks debate_rounds before moderator concludes, the first message is the user input
    if roles is None:
       roles = [ f"Debater {i}" for i in range(num_debaters)]
    elif len(roles) != num_debaters:
        raise ValueError(f"Number of roles ({len(roles)}) does not match the number of agents ({num_debaters})")   

    # Create the agents
    agents = []
    for i in range(num_debaters):  # Create debaters
        agents.append(AssistantAgent(
            name=re.sub(r'[^a-zA-Z0-9_-]', '_', roles[i]),
            system_message=f"You are {roles[i]}. {prompt}",
            llm_config=agent_config
        ))

    # Add moderator as the last agent
    moderator = AssistantAgent(
        name="Moderator",
        system_message=f"You are the moderator of the group. {decision_prompt}",
        llm_config=agent_config
    )
    agents.append(moderator)

    # Round robin
    def custom_speaker_selection(last_speaker, groupchat):         
        if len(groupchat.messages) == total_expect_messages - 1:
            return moderator # the moderator only speaks last
        else:
            return agents[(len(groupchat.messages) - 1) % num_debaters]

    # Create a group chat for the debate
    groupchat = GroupChat(
        agents=agents,
        messages=[],
        max_round=total_expect_messages,  
        speaker_selection_method=custom_speaker_selection
    )

    # Create and return the group chat manager
    return GroupChatManager(groupchat=groupchat, llm_config=agent_config)

# Create a user proxy for interaction
user_proxy = UserProxyAgent("user_proxy", code_execution_config=False)
manager = create_group_chat(4, None, prompt_store.chain_of_thought, prompt_store.vote_based)

# Example of initiating a debate
# user_proxy.initiate_chat(
#     manager,
#     message="""Let p = (1, 2, 5, 4)(2, 3) in S_5 . Find the index of <p> in S_5.

# Options:
# A. 8
# B. 2
# C. 24
# D. 120

# Please select the correct answer (A, B, C, or D)."""
# )
# Correct answer: C

logging_session_id = runtime_logging.start(config={"dbname": "logs.db"})
print("Logging session ID: " + str(logging_session_id))

import time

start_time = time.time()

user_proxy.initiate_chat(
    manager,
    message="""Statement 1 | A factor group of a non-Abelian group is non-Abelian. Statement 2 | If K is a normal subgroup of H and H is a normal subgroup of G, then K is a normal subgroup of G.

Options:
A. True, True
B. False, False
C. True, False
D. False, True

Please select the correct answer (A, B, C, or D)."""
)
# Correct answer: B

end_time = time.time()
runtime = end_time - start_time
print(f"Runtime of initiate_chat: {runtime:.2f} seconds")

runtime_logging.stop()