from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from setup_groupchat import create_group_chat, UserProxyAgent
import prompt_store

app = FastAPI()

class ChatRequest(BaseModel):
    num_debaters: int
    roles: Optional[List[str]] = None
    prompt: str
    decision_prompt: str
    debate_rounds: int = 2
    message: str

@app.post("/create_chat_and_send_message")
async def create_chat_and_send_message(request: ChatRequest):
    try:
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
        
        # Send the message to the group
        user_proxy.initiate_chat(
            chat_manager,
            message=request.message
        )
        
        return {"message": "Chat group created and message sent successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

