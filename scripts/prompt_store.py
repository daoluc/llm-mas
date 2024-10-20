from typing import List
from group_architecture import PromptType

# Agent Prompt

chain_of_thought = """You are debating in a group of agents to find the correct answer for a multiple choice question. Use your expertise to analyze and respond to other agents answers, addressing any errors you notice. Break down the problem into smaller steps. Explain your thought process step by step. Conclude with your final answer in the format: "Based on this analysis, the answer is X" where X is the letter of your chosen option."""

step_back_abstract = """You are debating in a group of agents to find the correct answer for a multiple choice question. Use your expertise to analyze and respond to other agents answers, addressing any errors you notice. Consider the broader context and nature of the question. Identify relevant principles, theories, or domain knowledge that apply. End with your final answer: "Based on this analysis, the answer is X" where X is your chosen letter."""

# Decision Prompt

vote_based = """Do not provide any further opinion. Only count the vote for a group solving a multiple choice question. Considering only their last stated choice. Tally the votes for each option. Determine the most popular choice. If there's a tie, use your judgment to break it. Conclude with: "The final answer is X" where X is the most popular choice."""

moderator_decide = """You are the lead of a group solving a multiple choice question. Carefully review and summarize the key points from each agent's contribution. Identify areas of consensus and disagreement among the agents. Consider any potential biases or errors in the agents' reasoning. Make an informed decision based on the collective input. Conclude with: "The final answer is X" where X is your chosen letter."""

def get_agent_prompt(prompt_type: PromptType, n_agents: int) -> List[str]:
    """
    Returns a list of appropriate agent prompts based on the given Prompt type.

    Args:
        prompt_type (PromptType): The type of prompt to retrieve.
        n_agents (int): The number of agents.

    Returns:
        List[str]: A list of corresponding prompt texts.

    Raises:
        ValueError: If an unsupported Prompt type is provided.
    """
    if prompt_type == PromptType.CHAIN_OF_THOUGHT:
        return [chain_of_thought] * n_agents
    elif prompt_type == PromptType.STEP_BACK_ABSTRACTION:
        return [step_back_abstract] * n_agents
    elif prompt_type == PromptType.MIXED:
        mixed_prompts = [chain_of_thought, step_back_abstract]
        return [mixed_prompts[i % 2] for i in range(n_agents)]
    else:
        raise ValueError(f"Unsupported prompt type: {prompt_type}")
    
def get_decision_prompt() -> str:
    """
    Returns the appropriate decision prompt. Only support mederator_decide prompt for now

    Returns:
        str: The corresponding decision prompt text.

    Raises:
        ValueError: If an unsupported prompt type is provided.
    """
    return vote_based