import json
import requests
from tqdm import tqdm
from datasets import load_dataset
import prompt_store
import role_store
import concurrent.futures
from functools import partial
import threading

def load_truthfulqa_mc1():
    """Load the TruthfulQA MC1 dataset."""
    return load_dataset("truthful_qa", "multiple_choice")["validation"]

def process_item(item, url, prompt_store):    
    question = item['question']
    options = item['mc1_targets']['choices']
    correct_answer = chr(65 + item['mc1_targets']['labels'].index(1))

    payload = {
        "num_debaters": 2,
        "roles": None,
        "prompt": prompt_store.chain_of_thought,
        "decision_prompt": prompt_store.moderator_decide,
        "debate_rounds": 2,
        "message": f"{question}\n\nOptions:\n" + "\n".join([f"{chr(65+i)}. {opt}" for i, opt in enumerate(options)]) + "\n\nPlease select the correct option letter (A, B, C, etc.)."
    }

    try:
        response = requests.post(url, json=payload)
        print(f"PROCESSING: {item['question']}")
        response.raise_for_status()
        result = response.json()
        agent_answer = result['answer']
        is_correct = agent_answer == correct_answer
        return is_correct
    except requests.RequestException as e:
        print(f"Error processing item: {e}")
        return None

def run_evaluation(dataset):
    """Run evaluation on the TruthfulQA MC1 dataset using parallel processing."""
    import time
    start_time = time.time()

    url = "http://0.0.0.0:8000/request_groupchat"
    correct_count = 0
    total_count = 0
    
    # Create a thread-safe counter
    thread_lock = threading.Lock()
    
    # Create a partial function with fixed arguments
    process_item_partial = partial(process_item, url=url, prompt_store=prompt_store)
    
    # Use ThreadPoolExecutor to process items in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        # Submit all tasks and get future objects
        future_to_item = {executor.submit(process_item_partial, item): item for item in dataset}
        
        # Process completed tasks as they finish
        for future in tqdm(concurrent.futures.as_completed(future_to_item), total=len(dataset)):
            item = future_to_item[future]
            try:
                is_correct = future.result()
                if is_correct is not None:
                    with thread_lock:
                        if is_correct:
                            correct_count += 1
                        total_count += 1
                    
                    if not is_correct:
                        print(f"INCORRECT: {item['question']}")
                
            except Exception as e:
                print(f"Error processing item: {e}")

    accuracy = correct_count / total_count if total_count > 0 else 0
    
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Execution time: {execution_time:.2f} seconds")
    
    return accuracy

def main():
    dataset = load_truthfulqa_mc1()
    # Select the first 10 questions from the dataset
    dataset = dataset.select(range(11,21))
    accuracy = run_evaluation(dataset)  # You can adjust the number of samples
    print(f"Accuracy: {accuracy:.2%}")

if __name__ == "__main__":
    main()
