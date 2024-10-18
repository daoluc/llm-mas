from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from setup_groupchat import create_group_chat
import time
import uuid
import re
from autogen import runtime_logging, UserProxyAgent

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
        
        # Start logging        
        logname = f"runtime_{uuid.uuid4()}.log"
        logging_session_id = runtime_logging.start(logger_type="file", config={"filename": logname})
        
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
        
        # Stop logging
        runtime_logging.stop()
        
        # Calculate runtime
        end_time = time.time()
        runtime = end_time - start_time
        
        conversation_history = chat_manager.groupchat.messages
        
        # Get token usage and session ID from the log
        # Read the log file
        with open("autogen_logs/"+logname, 'r') as log_file:
            log_content = log_file.read()
        
        # Extract all total_tokens values using regex        
        # token_matches = re.findall(r'total_tokens=(\d+)', log_content)
        completion_tokens_matches = re.findall(r'completion_tokens=(\d+)', log_content)
        prompt_tokens_matches = re.findall(r'prompt_tokens=(\d+)', log_content)
        
        # Convert matches to integers and sum them
        # total_tokens = sum(map(int, token_matches))
        completion_tokens = sum(map(int, completion_tokens_matches))
        prompt_tokens = sum(map(int, prompt_tokens_matches))
        
        # Clean up the log file
        # import os
        # os.remove(logname)
        
        return {            
            "runtime": f"{runtime:.2f} seconds",
            "completion_tokens": completion_tokens,
            "prompt_tokens": prompt_tokens,
            "messages": conversation_history,
            "session_id": logging_session_id,
            "logname": logname
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
