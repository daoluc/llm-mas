import os
from openai import OpenAI
from group_architecture import GroupArchitecture, Topology
import prompt_store
import role_store
import concurrent.futures
import threading

def run_single_agent(message, prompt):
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    llm_model = os.getenv("LLM_MODEL")
    temperature = float(os.getenv("TEMPERATURE"))

    messages = [        
        {'role': 'user', 'content': message},
        {'role': 'user', 'content': prompt}
    ]

    response = client.chat.completions.create(
        model=llm_model,
        messages=messages,
        temperature=temperature
    )
    
    completion_tokens = response.usage.completion_tokens
    prompt_tokens = response.usage.prompt_tokens

    return [{'role': 'user', 'content': message},{'role': 'user', 'content': response.choices[0].message.content}], completion_tokens, prompt_tokens

def run_groupchat(message, roles, prompts, decision_prompt, debate_rounds=2):    
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    llm_model = os.getenv("LLM_MODEL")
    temperature = float(os.getenv("TEMPERATURE"))

    if len(roles) != len(prompts):
        raise ValueError(f"Number of roles ({len(roles)}) does not match the number of prompts ({len(prompts)})")

    # Create the messages list
    messages = [
        {"role": "user", "content": message}        
    ]

    completion_tokens = 0
    prompt_tokens = 0

    # debate rounds        
    for i in range(len(roles)*debate_rounds):
        cur = i % len(roles)                        
        instruction = f"You are {roles[cur]}. {prompts[cur]}"

        # Send the request to the OpenAI API        
        response = client.chat.completions.create(
            model=llm_model,
            messages=messages + [{"role": "user", "content": instruction}],
            temperature=temperature
        )
        # Extract and return the response
        new_message = response.choices[0].message.content
        messages.append({"role": "user", "content": f"{roles[cur]}: {new_message}"})    
        completion_tokens += response.usage.completion_tokens
        prompt_tokens += response.usage.prompt_tokens

    # moderator
    instruction = f"You are moderator of the group. {decision_prompt}"    
    response = client.chat.completions.create(
        model=llm_model,
        messages=messages + [{"role": "user", "content": instruction}],
        temperature=temperature
    )
    new_message = response.choices[0].message.content
    messages.append({"role": "user", "content": f"Moderator: {new_message}"})
    completion_tokens += response.usage.completion_tokens
    prompt_tokens += response.usage.prompt_tokens


    return messages, completion_tokens, prompt_tokens

def run_one_on_one(message, roles, prompts, decision_prompt):
    """
    Run a one-on-one conversation where the input message is sent to each role independently.
    
    Args:
    message (str): The input message to be sent to each agent.
    roles (list): List of role names for the agents.
    prompts (list): List of prompts corresponding to each role.
    decision_prompt (str): The prompt for the final decision.
    
    Returns:
    dict: A dictionary containing the results of the conversation.
    """
    client = OpenAI()
    llm_model = os.getenv("LLM_MODEL")
    temperature = float(os.getenv("TEMPERATURE"))
    
    if len(roles) != len(prompts):
        raise ValueError("Number of roles must match the number of prompts.")
    
    all_messages = [{"role": "user", "content": message}]
    completion_tokens = 0
    prompt_tokens = 0

    # Function to process a single role
    def process_role(role, prompt):
        instruction = f"You are {role}. {prompt}"
        response = client.chat.completions.create(
            model=llm_model,
            messages=[
                {"role": "user", "content": message},
                {"role": "user", "content": instruction}
            ],
            temperature=temperature
        )
        new_message = response.choices[0].message.content
        return {
            "role": role,
            "message": new_message,
            "completion_tokens": response.usage.completion_tokens,
            "prompt_tokens": response.usage.prompt_tokens
        }

    # Use ThreadPoolExecutor to process roles in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(roles)) as executor:
        future_to_role = {executor.submit(process_role, role, prompt): role for role, prompt in zip(roles, prompts)}
        for future in concurrent.futures.as_completed(future_to_role):
            result = future.result()
            all_messages.append({"role": "user", "content": f"{result['role']}: {result['message']}"})
            completion_tokens += result['completion_tokens']
            prompt_tokens += result['prompt_tokens']
    
    # Final decision
    instruction = f"You are the moderator. {decision_prompt}"
    response = client.chat.completions.create(
        model=llm_model,
        messages=all_messages + [{"role": "user", "content": instruction}],
        temperature=temperature
    )
    final_decision = response.choices[0].message.content
    all_messages.append({"role": "user", "content": f"Moderator: {final_decision}"})
    completion_tokens += response.usage.completion_tokens
    prompt_tokens += response.usage.prompt_tokens
    
    return all_messages, completion_tokens, prompt_tokens


def run_reflection(message, roles, prompts, decision_prompt, debate_rounds=2):
    client = OpenAI()
    llm_model = os.getenv("LLM_MODEL")
    temperature = float(os.getenv("TEMPERATURE"))
    
    if len(roles) != len(prompts) or len(roles) % 2 != 0:
        raise ValueError("Number of roles must match the number of prompts and be even.")
    
    all_messages = [{"role": "user", "content": message}]
    completion_tokens = 0
    prompt_tokens = 0
    
    # Group roles and prompts into pairs
    pairs = list(zip(roles[::2], roles[1::2], prompts[::2], prompts[1::2]))
    
    # Function to run a single pair debate
    def run_pair_debate(role1, role2, prompt1, prompt2):
        pair_messages = [{"role": "user", "content": message}]
        pair_completion_tokens = 0
        pair_prompt_tokens = 0
        for _ in range(debate_rounds):
            for role, prompt in [(role1, prompt1), (role2, prompt2)]:
                instruction = f"You are {role}. {prompt}"
                response = client.chat.completions.create(
                    model=llm_model,
                    messages=pair_messages + [{"role": "user", "content": instruction}],
                    temperature=temperature
                )
                new_message = response.choices[0].message.content
                pair_messages.append({"role": "user", "content": f"{role}: {new_message}"})
                pair_completion_tokens += response.usage.completion_tokens
                pair_prompt_tokens += response.usage.prompt_tokens
        return pair_messages[1:], pair_completion_tokens, pair_prompt_tokens

    # Use ThreadPoolExecutor to run pair debates in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(pairs)) as executor:
        futures = [executor.submit(run_pair_debate, *pair) for pair in pairs]
        results = [future.result() for future in concurrent.futures.as_completed(futures)]

    # Aggregate results
    for pair_messages, pair_completion_tokens, pair_prompt_tokens in results:
        all_messages.extend(pair_messages)
        completion_tokens += pair_completion_tokens
        prompt_tokens += pair_prompt_tokens

    # Final decision
    instruction = f"You are the moderator. {decision_prompt}"
    response = client.chat.completions.create(
        model=llm_model,
        messages=all_messages + [{"role": "user", "content": instruction}],
        temperature=temperature
    )
    final_decision = response.choices[0].message.content
    all_messages.append({"role": "user", "content": f"Moderator: {final_decision}"})
    completion_tokens += response.usage.completion_tokens
    prompt_tokens += response.usage.prompt_tokens
    
    return all_messages, completion_tokens, prompt_tokens



def run_group_architecture(message: str, ga:GroupArchitecture):
    if ga.topology == Topology.SINGLE:
        roles = None
    elif ga.assign_role:
        roles = role_store.get_roles(ga.group_size)
    else:
        roles = [ f"Debater {i}" for i in range(ga.group_size)]
        
    prompts = prompt_store.get_agent_prompt(ga)
    decision_prompt = prompt_store.get_decision_prompt()
    
    if ga.topology == Topology.SINGLE:
        all_messages, completion_token, prompt_tokens = run_single_agent(message, prompts[0])
    elif ga.topology == Topology.GROUP_CHAT:
        all_messages, completion_token, prompt_tokens = run_groupchat(message, roles, prompts, decision_prompt)
    elif ga.topology == Topology.ONE_ON_ONE:
        all_messages, completion_token, prompt_tokens = run_one_on_one(message, roles, prompts, decision_prompt)
    elif ga.topology == Topology.REFLECTION:
        all_messages, completion_token, prompt_tokens = run_reflection(message, roles, prompts, decision_prompt)
    else:
        raise ValueError(f"Unsupported topology: {ga.topology}")

    return {
        "messages": all_messages,
        "completion_tokens": completion_token,
        "prompt_tokens": prompt_tokens
    }
    
# if __name__ == "__main__":
#     from prompt_store import get_agent_prompt, get_decision_prompt
#     from role_store import get_roles
#     from group_architecture import PromptType
#     from dotenv import load_dotenv
#     import time

#     # Load environment variables from .env file
#     load_dotenv()
    
#     # Example usage of run_group_architecture function
#     # Set up the parameters
#     message = "What is the capital of France? Options: A) London B) Berlin C) Paris D) Rome"
    
#     # Create a GroupArchitecture instance
#     ga = GroupArchitecture(
#         topology=Topology.GROUP_CHAT,
#         group_size=2,
#         prompt_type=PromptType.MIXED,