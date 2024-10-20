import csv
import os
import threading
import concurrent
from functools import partial
import logging
import json

from datasets import load_dataset
from dotenv import load_dotenv
from tqdm import tqdm

from agent_service import run_group_chat, run_single_agent
from group_architecture import GroupArchitecture, Topology, PromptType
from prompt_store import get_agent_prompt, get_decision_prompt
from role_store import get_roles
import traceback
from datetime import datetime
import random
import time

def load_truthfulqa_mc1():
    """Load the TruthfulQA MC1 dataset."""
    dataset = load_dataset("truthful_qa", "multiple_choice")["validation"]
    return dataset.add_column('id', range(len(dataset)))

def process_item(ga: GroupArchitecture, item):
    question = item['question']
    options = item['mc1_targets']['choices']
    correct_answer = chr(65 + item['mc1_targets']['labels'].index(1))

    message = f"{question}\n\nOptions:\n" + "\n".join([f"{chr(65+i)}. {opt}" for i, opt in enumerate(options)]) + "\n\nPlease select the correct option letter (A, B, C, etc.)."

    try:
        print(f"PROCESSING: {str(ga)} {item['question']}")
        if ga.topology == Topology.SINGLE:
            result = run_single_agent(message, get_agent_prompt(ga.prompt_type))
        elif ga.topology == Topology.GROUP_CHAT:
            result = run_group_chat(
                message=message,
                num_debaters=ga.group_size,
                roles=get_roles(ga.group_size),
                prompts=get_agent_prompt(ga.prompt_type, ga.group_size),
                decision_prompt=get_decision_prompt()
            )
        else:            
            raise NotImplementedError(f"Group Architecture {ga.topology} is not supported.")
        agent_answer = result['answer']
        runtime = result['runtime']        
        is_correct = agent_answer == correct_answer

        return {
            'architecture': str(ga),
            'question_id': item['id'],
            'question': question,
            'correct_answer': correct_answer,
            'agent_answer': agent_answer,
            'runtime': runtime,
            'tokens': result['total_tokens'],
            'cost': result['total_cost'],
            'is_correct': is_correct,
            'messages': result['messages']
        }
    except Exception as e:        
        logging.error(f"Error processing item: {e}")
        logging.error("Exception stack trace:")
        logging.error(traceback.format_exc())
        return None

def run_evaluation(ga:GroupArchitecture, dataset, current_datetime, n_threads=10):
    """Run evaluation on the TruthfulQA MC1 dataset using parallel processing."""
    start_time = time.time()

    correct_count = 0
    total_count = 0
    total_cost = 0
    total_tokens = 0
    
    # Create a thread-safe counter and CSV writer
    thread_lock = threading.Lock()
    
    # Prepare CSV file
    fieldnames = ['architecture', 'question_id', 'question', 'correct_answer', 'agent_answer', 'is_correct', 'runtime', 'tokens', 'cost', "current_datetime"]        
    os.makedirs('results', exist_ok=True)
    file_exists = os.path.isfile('results/result_per_question.csv') and os.path.getsize('results/result_per_question.csv') > 0
    csv_file = open('results/result_per_question.csv', 'a', newline='')
    csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    if not file_exists:
        csv_writer.writeheader()
    
    # Prepare messages file
    messages_file = open('results/messages.txt', 'a')
    
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
                result = future.result()
                if result is not None:
                    with thread_lock:
                        if result['is_correct']:
                            correct_count += 1
                        total_count += 1
                        total_cost += result['cost']
                        total_tokens += result['tokens']
                    
                        # Save messages to a common text file
                        messages_file.write(f">>>>> {current_datetime} {ga} Question ID:{result['question_id']}\n")
                        messages_file.write(json.dumps(result['messages'], indent=2))
                        messages_file.write("\n\n")
                        
                        # Write result to CSV file immediately
                        del result['messages']
                        result['current_datetime'] = current_datetime
                        csv_writer.writerow(result)
                    
                    if not result['is_correct']:
                        logging.warning(f"INCORRECT: {item['question']}")
                
            except Exception as e:
                logging.error(f"Error processing item: {e}")
                logging.error("Exception stack trace:")
                logging.error(traceback.format_exc())

    csv_file.close()
    messages_file.close()

    accuracy = correct_count / total_count if total_count > 0 else 0
    
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Execution time: {execution_time:.2f} seconds")    
    save_result(ga, accuracy, len(dataset), current_datetime, total_cost, total_tokens, execution_time)
    
    return accuracy

def save_result(ga: GroupArchitecture, accuracy, dataset_size, current_datetime, total_cost, total_tokens, execution_time):
    os.makedirs('results', exist_ok=True)
    with open('results/results.csv', 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if not os.path.isfile('results/results.csv') or os.path.getsize('results/results.csv') == 0:
            writer.writerow(['architecture', 'accuracy', 'dataset_size', 'datetime', 'total_cost', 'total_tokens', 'execution_time'])
        writer.writerow([str(ga), accuracy, dataset_size, current_datetime, total_cost, total_tokens, execution_time])

def main():
    load_dotenv()
    
    current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"RUN ROUND {current_datetime}")
        
    dataset = load_truthfulqa_mc1()
    dataset = dataset.select(random.sample(range(len(dataset)), 100))
    
    ga = GroupArchitecture(Topology.SINGLE, 1, PromptType.CHAIN_OF_THOUGHT)
    accuracy = run_evaluation(ga, dataset, current_datetime, n_threads=10)
    print(f"Accuracy: {accuracy:.2%}")    
    print(f"Results saved to results/results.csv and results/result_per_question.csv")
    
    ga = GroupArchitecture(Topology.GROUP_CHAT, 2, PromptType.CHAIN_OF_THOUGHT)
    accuracy = run_evaluation(ga, dataset, current_datetime, n_threads=10)
    print(f"Accuracy: {accuracy:.2%}")    
    print(f"Results saved to results/results.csv and results/result_per_question.csv")
    
    ga = GroupArchitecture(Topology.GROUP_CHAT, 2, PromptType.MIXED)
    accuracy = run_evaluation(ga, dataset, current_datetime, n_threads=10)
    print(f"Accuracy: {accuracy:.2%}")    
    print(f"Results saved to results/results.csv and results/result_per_question.csv")

if __name__ == "__main__":
    main()
