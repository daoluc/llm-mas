import requests
import prompt_store
import role_store

# Define the API endpoint
url = "http://localhost:8000/request_groupchat"  # Adjust the URL if needed

# Prepare the request payload
payload = {
    "num_debaters": 2,
    "roles": None,
    "prompt": prompt_store.chain_of_thought,
    "decision_prompt": prompt_store.vote_based,
    "debate_rounds": 2,
    "message": """Statement 1 | A factor group of a non-Abelian group is non-Abelian. Statement 2 | If K is a normal subgroup of H and H is a normal subgroup of G, then K is a normal subgroup of G.

Options:
A. True, True
B. False, False
C. True, False
D. False, True

Please select the correct answer (A, B, C, or D)."""
}

# Send the POST request
response = requests.post(url, json=payload)

# Check if the request was successful
if response.status_code == 200:
    result = response.json()
    print(f"Messages: {result['messages']}")
    print(f"Runtime: {result['runtime']}")
    print(f"Answer: {result['answer']}")
else:
    print(f"Error: {response.status_code}")
    print(response.text)
