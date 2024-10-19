from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from setup_groupchat import create_group_chat
import time
from autogen import UserProxyAgent
from utils import extract_answer_letter

app = FastAPI()

class ChatRequest(BaseModel):
    num_debaters: int
    roles: Optional[List[str]] = None
    prompt: str
    decision_prompt: str
    debate_rounds: int = 2
    message: str

@app.post("/request_groupchat")
async def request_groupchat(request: ChatRequest):
    try:
        start_time = time.time()
        
        print("STARTING " + str(start_time))

        # Create the group chat
        chat_manager = create_group_chat(
            request.num_debaters,
            request.roles,
            request.prompt,
            request.decision_prompt,
            request.debate_rounds
        )
        
        # Create a user proxy for interaction
        user_proxy = UserProxyAgent("user_proxy", code_execution_config=False)
        
        # Send the message to the group and get the last message
        user_proxy.initiate_chat(
            chat_manager,
            message=request.message
        )
        
        # Calculate runtime
        end_time = time.time()
        runtime = end_time - start_time
        
        messages = chat_manager.groupchat.messages
        answer = extract_answer_letter(messages[-1]['content'])
        
        print("FINISHING " + str(start_time))
              
        return {            
            "runtime": f"{runtime:.2f}",
            # "completion_tokens": completion_tokens,
            # "prompt_tokens": prompt_tokens,
            "messages": messages,
            'answer': answer,
            # "session_id": logging_session_id,
            # "logname": logname
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
