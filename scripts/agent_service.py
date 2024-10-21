import os
from openai import OpenAI
from group_architecture import GroupArchitecture, Topology
import prompt_store
import role_store

def run_single_agent(message, prompt):
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    llm_model = os.getenv("LLM_MODEL")

    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": message}
    ]

    response = client.chat.completions.create(
        model=llm_model,
        messages=messages,
        temperature=1.0
    )
    
    completion_tokens = response.usage.completion_tokens
    prompt_tokens = response.usage.prompt_tokens

    return [{'content': response.choices[0].message.content}], completion_tokens, prompt_tokens

def run_groupchat(message, roles, prompts, decision_prompt, debate_rounds=2):    
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # Set up OpenAI API key
    llm_model = os.getenv("LLM_MODEL")

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
            temperature=1.0
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
        temperature=1.0
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
    
    if len(roles) != len(prompts):
        raise ValueError("Number of roles must match the number of prompts.")
    
    all_messages = [{"role": "user", "content": message}]
    completion_tokens = 0
    prompt_tokens = 0

    # Send message to each role independently
    for role, prompt in zip(roles, prompts):
        instruction = f"You are {role}. {prompt}"
        response = client.chat.completions.create(
            model=llm_model,
            messages=[
                {"role": "user", "content": message},
                {"role": "user", "content": instruction}
            ],
            temperature=1.0
        )
        new_message = response.choices[0].message.content
        all_messages.append({"role": "user", "content": f"{role}: {new_message}"})
        completion_tokens += response.usage.completion_tokens
        prompt_tokens += response.usage.prompt_tokens
    
    # Final decision
    instruction = f"You are the moderator. {decision_prompt}"
    response = client.chat.completions.create(
        model=llm_model,
        messages=all_messages + [{"role": "user", "content": instruction}],
        temperature=1.0
    )
    final_decision = response.choices[0].message.content
    all_messages.append({"role": "user", "content": f"Moderator: {final_decision}"})
    completion_tokens += response.usage.completion_tokens
    prompt_tokens += response.usage.prompt_tokens
    
    return all_messages, completion_tokens, prompt_tokens


def run_reflection(message, roles, prompts, decision_prompt, debate_rounds=2):
    client = OpenAI()
    llm_model = os.getenv("LLM_MODEL")
    
    if len(roles) != len(prompts) or len(roles) % 2 != 0:
        raise ValueError("Number of roles must match the number of prompts and be even.")
    
    all_messages = [{"role": "user", "content": message}]
    completion_tokens = 0
    prompt_tokens = 0
    

    # Group roles and prompts into pairs
    pairs = list(zip(roles[::2], roles[1::2], prompts[::2], prompts[1::2]))
    
    for role1, role2, prompt1, prompt2 in pairs:
        pair_messages = [{"role": "user", "content": message}]
        for _ in range(debate_rounds):                            
            # First agent in the pair
            instruction1 = f"You are {role1}. {prompt1}"
            response1 = client.chat.completions.create(
                model=llm_model,
                messages=pair_messages + [{"role": "user", "content": instruction1}],
                temperature=1.0
            )
            new_message1 = response1.choices[0].message.content
            pair_messages.append({"role": "user", "content": f"{role1}: {new_message1}"})
            completion_tokens += response1.usage.completion_tokens
            prompt_tokens += response1.usage.prompt_tokens

            # Second agent in the pair
            instruction2 = f"You are {role2}. {prompt2}"
            response2 = client.chat.completions.create(
                model=llm_model,
                messages=pair_messages + [{"role": "user", "content": instruction2}],
                temperature=1.0
            )
            new_message2 = response2.choices[0].message.content
            pair_messages.append({"role": "user", "content": f"{role2}: {new_message2}"})
            completion_tokens += response2.usage.completion_tokens
            prompt_tokens += response2.usage.prompt_tokens
        all_messages += pair_messages[1:]

    # Final decision
    instruction = f"You are the moderator. {decision_prompt}"
    response = client.chat.completions.create(
        model=llm_model,
        messages=all_messages + [{"role": "user", "content": instruction}],
        temperature=1.0
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
#         assign_role=True,
#         malicious_target='A'
#     )

#     # Run the group architecture
#     start_time = time.time()
#     result = run_group_architecture(message, ga)
#     end_time = time.time()
#     runtime = end_time - start_time

#     print(f">>>{ga}")
#     print(f"Runtime: {runtime:.2f} seconds")
#     print(f"Completion tokens: {result['completion_tokens']}")
#     print(f"Prompt tokens: {result['prompt_tokens']}")

#     print("Group Architecture Results:")
#     for msg in result['messages']:
#         print(f"{msg['role']}: {msg['content']}")
    
    # # Set up the parameters
    # num_debaters = 2
    # roles = get_roles(num_debaters)
    # prompts = get_agent_prompt(PromptType.CHAIN_OF_THOUGHT, num_debaters)
    # decision_prompt = get_decision_prompt()
    # message = "What is the capital of France? A) London B) Berlin C) Paris D) Rome"

    # # Run the group chat
    # start_time = time.time()
    # result, completion_tokens, prompt_tokens = run_groupchat(num_debaters, roles, prompts, decision_prompt, message)
    # end_time = time.time()
    # runtime = end_time - start_time
    # print(f"Runtime: {runtime:.2f} seconds")

    # # Print completion and prompt tokens
    # print(f"Completion tokens: {completion_tokens}")
    # print(f"Prompt tokens: {prompt_tokens}")

    # # Print the results
    # if result:
    #     print("Group Chat Results:")
    #     for msg in result:
    #         print(f"{msg['content']}")
    # else:
    #     print("An error occurred during the group chat.")
    
    # # Example usage of run_single_agent function    
    # # Set up the parameters
    # message = "What is the capital of France? A) London B) Berlin C) Paris D) Rome"
    # prompt = get_agent_prompt(PromptType.CHAIN_OF_THOUGHT, 1)[0]

    # # Run the single agent
    # start_time = time.time()
    # result, completion_tokens, prompt_tokens  = run_single_agent(message, prompt)
    # end_time = time.time()
    # runtime = end_time - start_time
    
    # print(result)
    # print(completion_tokens)
    # print(prompt_tokens)
    
    # # Example usage of run_one_on_one function
    # # Set up the parameters
    # message = "What is the capital of France? A) London B) Berlin C) Paris D) Rome"
    # prompt = get_agent_prompt(PromptType.CHAIN_OF_THOUGHT, 2)

    # # Run the single agent
    # start_time = time.time()
    # result, completion_tokens, prompt_tokens  = run_one_on_one(message, get_roles(2), get_agent_prompt(PromptType.CHAIN_OF_THOUGHT, 2), get_decision_prompt())
    # end_time = time.time()
    # runtime = end_time - start_time
    
    # print(result)
    # print(completion_tokens)
    # print(prompt_tokens)
    
    # # Example usage of run_reflection function
    # # Set up the parameters
    # message = "What is the capital of France? A) London B) Berlin C) Paris D) Rome"
    # initial_prompt = get_agent_prompt(PromptType.CHAIN_OF_THOUGHT, 4)   

    # # Run the reflection
    # start_time = time.time()
    # result, completion_tokens, prompt_tokens = run_reflection(message, get_roles(4), get_agent_prompt(PromptType.MIXED, 4), get_decision_prompt())
    # end_time = time.time()
    # runtime = end_time - start_time

    # print("Reflection Results:")
    # print(result)
    # print(f"Completion tokens: {completion_tokens}")
    # print(f"Prompt tokens: {prompt_tokens}")
    # print(f"Runtime: {runtime:.2f} seconds")