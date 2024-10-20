import os
import time
from autogen import AssistantAgent, UserProxyAgent
from setup_groupchat import create_group_chat
from utils import extract_answer_letter

def run_single_agent(message, prompt):
    try:
        start_time = time.time()
        llm_model = os.getenv("LLM_MODEL")

        # Configure the AI model
        config_list = [
            {
                "model": llm_model,
                "api_key": os.getenv("OPENAI_API_KEY")
            }
        ]

        # Create a configuration for the agent
        agent_config = {
            "config_list": config_list,
            "cache_seed": None,
        }

        # Create the single agent
        agent = AssistantAgent(
            name="SingleAgent",
            system_message=prompt,
            llm_config=agent_config
        )

        # Create a user proxy for interaction
        user_proxy = UserProxyAgent("user_proxy", code_execution_config=False)

        # Send the message to the agent and get the response
        chat_result = user_proxy.initiate_chat(
            agent,
            max_turns=1,
            message=message,
            silent=True
        )

        # Calculate runtime
        end_time = time.time()
        runtime = end_time - start_time

        messages = user_proxy.chat_messages[agent]
        answer = extract_answer_letter(messages[-1]['content'])

        # Extract total token and cost from ChatResult
        total_tokens = chat_result.cost['usage_including_cached_inference'][llm_model]['total_tokens']
        total_cost = chat_result.cost['usage_including_cached_inference']['total_cost']

        return {
            "runtime": f"{runtime:.2f}",
            "messages": messages,
            'answer': answer,
            'total_tokens': total_tokens,
            'total_cost': total_cost,
        }
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return None


def run_group_chat(
    message,
    num_debaters,
    prompts,
    decision_prompt,
    roles = None,
):
    try:
        start_time = time.time()
        llm_model = os.getenv("LLM_MODEL")

        # Create the group chat
        chat_manager = create_group_chat(
            num_debaters,
            roles,
            prompts,
            decision_prompt
        )
        
        # Create a user proxy for interaction
        user_proxy = UserProxyAgent("user_proxy", code_execution_config=False)
        
        # Send the message to the group and get the last message
        user_proxy.initiate_chat(
            chat_manager,
            message=message,
            silent=True
        )                
        
        # Calculate runtime
        end_time = time.time()
        runtime = end_time - start_time
        
        messages = chat_manager.groupchat.messages
        answer = extract_answer_letter(messages[-1]['content'])
        
        # Extract cost and token usage from each agent        
        total_cost = 0
        total_tokens = 0
        for agent in chat_manager.groupchat.agents:
            if agent.client and agent.client.total_usage_summary:
                usage = agent.client.total_usage_summary
                total_cost += usage['total_cost']
                total_tokens += usage[llm_model]['total_tokens']                
              
        return {            
            "runtime": f"{runtime:.2f}",
            "messages": messages,
            'answer': answer,
            'total_cost': total_cost,
            'total_tokens': total_tokens
        }
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return None

# Example usage:
# from dotenv import load_dotenv
# if __name__ == "__main__":
#     load_dotenv()
    
    # result = run_group_chat(
    #     num_debaters=2,
    #     roles=["Scientist", "Philosopher"],
    #     prompt="Discuss the given topic from your perspective.",
    #     decision_prompt="Summarize the discussion and provide a conclusion.",
    #     message="What is the nature of consciousness?"
    # )
    
    # if result:
    #     print(f"Runtime: {result['runtime']} seconds")
    #     print(f"Final answer: {result['answer']}")
    #     print(f"Total cost: ${result['total_cost']:.6f}")
    #     print(f"Total tokens: {result['total_tokens']}")
    #     print("Messages:")
    #     for msg in result['messages']:
    #         print(f"{msg['name']}: {msg['content']}")
    # else:
    #     print("The group chat encountered an error.")
    
    # result = run_single_agent(
    #     message="What is the capital of France?",
    #     prompt="You are a knowledgeable assistant. Please answer the following question concisely."
    # )
    
    # if result:
    #     print(f"Runtime: {result['runtime']} seconds")
    #     print(f"Final answer: {result['answer']}")
    #     print(f"Total cost: ${result['total_cost']:.6f}")
    #     print(f"Total tokens: {result['total_tokens']}")
    #     print("Message:")
    #     for msg in result['messages']:
    #         print(f"{msg['name']}: {msg['content']}")
    # else:
    #     print("The single agent encountered an error.")

