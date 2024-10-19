from typing import List, Optional
from setup_groupchat import create_group_chat
from autogen import UserProxyAgent
from utils import extract_answer_letter
import time

def run_group_chat(
    message,
    num_debaters,
    prompt,
    decision_prompt,
    roles = None,
):
    try:
        start_time = time.time()

        # Create the group chat
        chat_manager = create_group_chat(
            num_debaters,
            roles,
            prompt,
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
              
        return {            
            "runtime": f"{runtime:.2f}",
            "messages": messages,
            'answer': answer,
        }
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return None

# Example usage:
# if __name__ == "__main__":
#     result = run_group_chat(
#         num_debaters=2,
#         roles=["Scientist", "Philosopher"],
#         prompt="Discuss the given topic from your perspective.",
#         decision_prompt="Summarize the discussion and provide a conclusion.",
#         debate_rounds=2,
#         message="What is the nature of consciousness?"
#     )
    
#     if result:
#         print(f"Runtime: {result['runtime']} seconds")
#         print(f"Final answer: {result['answer']}")
#         print("Messages:")
#         for msg in result['messages']:
#             print(f"{msg['name']}: {msg['content']}")
#     else:
#         print("The group chat encountered an error.")

