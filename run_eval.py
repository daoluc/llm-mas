import json
import requests
from tqdm import tqdm
from datasets import load_dataset
import prompt_store
import role_store

def load_truthfulqa_mc1():
    """Load the TruthfulQA MC1 dataset."""
    return load_dataset("truthful_qa", "multiple_choice")["validation"]

def run_evaluation(dataset):
    """Run evaluation on the TruthfulQA MC1 dataset."""
    url = "http://localhost:8000/request_groupchat"
    correct_count = 0
    total_count = 0

    for item in tqdm(dataset):
        print(item)
        question = item['question']
        options = item['mc1_targets']['choices']
        correct_answer = chr(65 + item['mc1_targets']['labels'].index(1))
        print(correct_answer)

        # Prepare the payload
        payload = {
            "num_debaters": 2,
            "roles": None,
            "prompt": prompt_store.chain_of_thought,
            "decision_prompt": prompt_store.moderator_decide,
            "debate_rounds": 2,
            "message": f"{question}\n\nOptions:\n" + "\n".join([f"{chr(65+i)}. {opt}" for i, opt in enumerate(options)]) + "\n\nPlease select the correct option letter (A, B, C, etc.)."
        }

        # Send the request
        response = requests.post(url, json=payload)        

        if response.status_code == 200:
            result = response.json()
            print(result)
            agent_answer = result['answer']
            if agent_answer and agent_answer == correct_answer:                                
                correct_count += 1
            else:
                print("INCORRECT!")
            total_count += 1
        else:
            print(f"Error: {response.status_code}")
            print(response.text)

    accuracy = correct_count / total_count if total_count > 0 else 0
    return accuracy

def main():
    dataset = load_truthfulqa_mc1()
    # Select the first 2 questions from the dataset
    dataset = dataset.select(range(10))
    accuracy = run_evaluation(dataset)  # You can adjust the number of samples
    print(f"Accuracy: {accuracy:.2%}")

if __name__ == "__main__":
    main()
