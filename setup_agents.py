from autogen import AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager
import os

# Configure the AI models
config_list = [
    {
        "model": "gpt-4o-mini",
        "api_key": os.getenv("OPENAI_API_KEY")
    }
]

# Create a configuration for the agents
agent_config = {"config_list": config_list}

# Create the three debate agents
moderator = AssistantAgent(
    name="Moderator",
    system_message="You are a neutral moderator in a debate. You provide final decision when it is your turn. Finish your turn with \"the answer is X\" where X is the correct letter choice.",
    llm_config=agent_config
)

debater_1 = AssistantAgent(
    name="Debater1",
    system_message="You are a debater who takes a position on topics and argues persuasively. You should consider counterarguments and respond to them effectively.",
    llm_config=agent_config
)

debater_2 = AssistantAgent(
    name="Debater2",
    system_message="You are a debater who takes an opposing position to Debater 1. You should present strong arguments and challenge the other debater's points.",
    llm_config=agent_config
)

# Create a user proxy for interaction
user_proxy = UserProxyAgent("user_proxy", code_execution_config=False)

# Custom speaker selection method
def custom_speaker_selection(last_speaker, groupchat):
    order = [debater_1, debater_2, debater_1, debater_2, moderator]
    print("Message count:",len(groupchat.messages)-1)
    return order[(len(groupchat.messages)-1) % len(order)]

# Create a group chat for the debate
groupchat = GroupChat(
    agents=[debater_1, debater_2, moderator],
    messages=[],
    max_round=6,
    speaker_selection_method=custom_speaker_selection
)

# Create the group chat manager
manager = GroupChatManager(groupchat=groupchat, llm_config=agent_config)

# Example of initiating a debate
user_proxy.initiate_chat(
    manager,
    message="""Let p = (1, 2, 5, 4)(2, 3) in S_5 . Find the index of <p> in S_5.

Options:
A. 8
B. 2
C. 24
D. 120

Please select the correct answer (A, B, C, or D)."""
)

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
