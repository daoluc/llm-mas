import tiktoken
import re

def extract_answer_letter(agent_response):
    # Convert the response to lowercase for case-insensitive matching
    response_lower = agent_response.lower()
    
    # Find the index of "answer is "
    index = response_lower.rfind("answer is ")
    
    if index != -1:
        # Get the first character after "answer is "
        answer_letter = agent_response[index + 10].upper()
        
        # Check if the extracted letter is a valid option (A, B, C, or D)
        if answer_letter in ['A', 'B', 'C', 'D']:
            return answer_letter
    
    # Return None if no valid answer letter is found
    return None

def calculate_tokens(text):
    """
    Calculate the number of tokens in the given text.
    
    Args:
    text (str): The input text to tokenize.
    
    Returns:
    int: The number of tokens in the text.
    """
    # Initialize the tokenizer
    enc = tiktoken.encoding_for_model("gpt-4o-mini")
    
    # Tokenize the text
    tokens = enc.encode(text)
    
    # Return the number of tokens
    return len(tokens)

def calculate_groupchat_tokens(messages):
    """
    Calculate the total number of tokens in a list of messages.
    
    Args:
    messages (list): A list of message dictionaries, each containing 'content' and 'role' keys.
    
    Returns:
    int: The total number of tokens in all messages.
    """
    total_tokens = 0
    
    for i in range(len(messages)):
        message = messages[i]
        
        # Calculate tokens for the message content
        content_tokens = calculate_tokens(message['content']) + calculate_tokens(message['name'])
        print(content_tokens)
        total_tokens += content_tokens * (len(messages)-1-i)  
    
    return total_tokens

def extract_answer_letter(agent_response):    
    # Remove special characters from the response
    response_clean = re.sub(r'[^a-zA-Z0-9\s]', '', agent_response).lower()
    
    # Find the index of "answer is "
    index = response_clean.rfind("answer is ")
    
    if index != -1:
        # Get the first character after "answer is "
        answer_letter = response_clean[index + 10].upper()                
        return answer_letter
    
    # Return None if no valid answer letter is found
    return None
