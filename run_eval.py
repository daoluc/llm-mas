from tqdm import tqdm
from datasets import load_dataset
from prompt_store import get_agent_prompt, get_decision_prompt
import concurrent.futures
from functools import partial
import threading
from agent_service import run_group_chat
from group_architecture import GroupArchitecture, Topology, PromptType

def load_truthfulqa_mc1():
    """Load the TruthfulQA MC1 dataset."""
    return load_dataset("truthful_qa", "multiple_choice")["validation"]

def process_item(ga: GroupArchitecture, item):
    question = item['question']
    options = item['mc1_targets']['choices']
    correct_answer = chr(65 + item['mc1_targets']['labels'].index(1))

    message = f"{question}\n\nOptions:\n" + "\n".join([f"{chr(65+i)}. {opt}" for i, opt in enumerate(options)]) + "\n\nPlease select the correct option letter (A, B, C, etc.)."

    try:
        print(f"PROCESSING: {str(ga)} {item['question']}")
        if ga.topology == Topology.GROUP_CHAT:
            result = run_group_chat(
                message=message,
                num_debaters=ga.group_size,
                roles=None,
                prompt=get_agent_prompt(ga.prompt_type),
                decision_prompt=get_decision_prompt()
            )
        else:
            raise NotImplementedError(f"Group Architecture {ga.topology} is not supported.")
        agent_answer = result['answer']
        is_correct = agent_answer == correct_answer
        return is_correct
    except Exception as e:
        print(f"Error processing item: {e}")
        return None

def run_evaluation(ga:GroupArchitecture, dataset, n_threads=10):
    """Run evaluation on the TruthfulQA MC1 dataset using parallel processing."""
    import time
    start_time = time.time()

    correct_count = 0
    total_count = 0
    
    # Create a thread-safe counter
    thread_lock = threading.Lock()
    
    # Create a partial function with fixed arguments
    process_item_partial = partial(process_item, ga)
    
    # Use ThreadPoolExecutor to process items in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_threads) as executor:
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
    ga = GroupArchitecture(Topology.GROUP_CHAT, 2, PromptType.STEP_BACK_ABSTRACTION)    
    
    dataset = load_truthfulqa_mc1()
    dataset = dataset.select(range(4))    
    accuracy = run_evaluation(ga, dataset, n_threads=2)  # You can adjust the number of samples
    print(f"Accuracy: {accuracy:.2%}")

if __name__ == "__main__":
    main()
