from group_architecture import PromptType

# Agent Prompt

chain_of_thought = """You are working in a group of agents to find the correct answer for a multiple choice question. Break down the problem into smaller steps. Reason through each step logically and clearly. Explain your thought process as if teaching a novice. Analyze and respond to other agents' opinions, addressing any errors you notice. Conclude with your final answer in the format: "Based on this analysis, the answer is X" where X is the letter of your chosen option."""

step_back_abstract = """You are working in a group of agents to find the correct answer for a multiple choice question. Consider the broader context and nature of the question. Identify relevant principles, theories, or domain knowledge that apply. Explain how these general concepts relate to the specific question. Analyze other agents' responses, noting agreements and discrepancies with your perspective. Synthesize the information to reach a conclusion. End with your final answer: "Based on this analysis, the answer is X" where X is your chosen letter."""

# Decision Prompt

vote_based = """You count the vote for a group solving a multiple choice question. Considering only their last stated choice. Tally the votes for each option. Determine the most popular choice. If there's a tie, use your judgment to break it. Conclude with: "The final answer is X" where X is the winning letter choice."""

moderator_decide = """You are the lead of a group solving a multiple choice question. Carefully review and summarize the key points from each agent's contribution. Identify areas of consensus and disagreement among the agents. Consider any potential biases or errors in the agents' reasoning. Make an informed decision based on the collective input. Conclude with: "The final answer is X" where X is your chosen letter."""

def get_agent_prompt(prompt_type: PromptType) -> str:
    """
    Returns the appropriate agent prompt based on the given Prompt type.

    Args:
        prompt_type (Prompt): The type of prompt to retrieve.

    Returns:
        str: The corresponding prompt text.

    Raises:
        ValueError: If an unsupported Prompt type is provided.
    """
    if prompt_type == PromptType.CHAIN_OF_THOUGHT:
        return chain_of_thought
    elif prompt_type == PromptType.STEP_BACK_ABSTRACTION:
        return step_back_abstract
    elif prompt_type == PromptType.MIXED:
        return f"{chain_of_thought}\n\n{step_back_abstract}"    
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
    return moderator_decide