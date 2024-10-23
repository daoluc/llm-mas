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

from agent_service import run_group_architecture
from group_architecture import GroupArchitecture, Topology, PromptType
import traceback
from datetime import datetime
import random
import time
import utils

def load_truthfulqa_mc1():
    """Load the TruthfulQA MC1 dataset."""
    dataset = load_dataset("truthful_qa", "multiple_choice")["validation"]
    return dataset.add_column('id', range(len(dataset)))

def process_item(ga: GroupArchitecture, item):
    question = item['question']
    options = item['mc1_targets']['choices']
    correct_index = item['mc1_targets']['labels'].index(1)
    
    # Randomize the order of options
    shuffled_indices = list(range(len(options)))
    random.shuffle(shuffled_indices)
    
    shuffled_options = [options[i] for i in shuffled_indices]
    new_correct_index = shuffled_indices.index(correct_index)
    correct_answer = chr(65 + new_correct_index)
    
    # If malicious_target is set, assign it to a random incorrect choice
    if ga.malicious_target:
        incorrect_indices = [i for i in range(len(shuffled_options)) if i != new_correct_index]
        if incorrect_indices:
            malicious_index = random.choice(incorrect_indices)
            ga.malicious_target = chr(65 + malicious_index)

    message = f"{question}\n\nOptions:\n" + "\n".join([f"{chr(65+i)}. {opt}" for i, opt in enumerate(shuffled_options)]) + "\n\nPlease select the correct option letter (A, B, C, etc.)."

    try:
        print(f"PROCESSING: {str(ga)} {item['question']}")
        start_time = time.time()
        result = run_group_architecture(message, ga)
        end_time = time.time()
        
        runtime = end_time - start_time                        
        agent_answer = utils.extract_answer_letter(result['messages'][-1]['content'])            
        is_correct = agent_answer == correct_answer

        return {
            'architecture': str(ga),
            'question_id': item['id'],
            'question': question,
            'correct_answer': correct_answer,
            'agent_answer': agent_answer,
            'runtime': runtime,
            'completion_tokens': result['completion_tokens'],
            'prompt_tokens': result['prompt_tokens'],
            'is_correct': is_correct,
            'messages': result['messages']
        }
    except Exception as e:        
        logging.error(f"Error processing item: {e}")
        logging.error("Exception stack trace:")
        logging.error(traceback.format_exc())
        return None

def run_evaluation(ga:GroupArchitecture, dataset, datetime, n_threads=10):
    """Run evaluation on the TruthfulQA MC1 dataset using parallel processing."""
    start_time = time.time()

    correct_count = 0
    total_count = 0
    total_completion_tokens = 0
    total_prompt_tokens = 0
    total_runtime = 0
    
    # Create a thread-safe counter and CSV writer
    thread_lock = threading.Lock()
    
    # Prepare CSV file
    fieldnames = ['architecture', 'question_id', 'question', 'correct_answer', 'agent_answer', 'is_correct', 'runtime', 'completion_tokens', 'prompt_tokens', "datetime"]        
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
                        total_completion_tokens += result['completion_tokens']
                        total_prompt_tokens += result['prompt_tokens']
                        total_runtime += result['runtime']
                    
                        # Save messages to a common text file
                        messages_file.write(f">>>>> {datetime} {ga} Question ID:{result['question_id']}\n")
                        messages_file.write(json.dumps(result['messages'], indent=2))
                        messages_file.write("\n\n")
                        
                        # Write result to CSV file immediately
                        del result['messages']
                        result['datetime'] = datetime
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
    average_runtime = total_runtime / total_count if total_count > 0 else 0
    
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Execution time: {execution_time:.2f} seconds")    
    save_result(ga, accuracy, len(dataset), datetime, total_completion_tokens, total_prompt_tokens, execution_time, average_runtime)
    
    return accuracy

def save_result(ga: GroupArchitecture, accuracy, dataset_size, datetime, total_completion_tokens, total_tokens, execution_time, average_runtime):
    os.makedirs('results', exist_ok=True)
    with open('results/results.csv', 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if not os.path.isfile('results/results.csv') or os.path.getsize('results/results.csv') == 0:
            writer.writerow(['architecture', 'accuracy', 'dataset_size', 'datetime', 'total_completion_tokens', 'total_prompt_tokens', 'execution_time', 'average_runtime'])
        writer.writerow([str(ga), accuracy, dataset_size, datetime, total_completion_tokens, total_tokens, execution_time, average_runtime])

def main():
    load_dotenv()
    
    current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"RUN ROUND {current_datetime}")
        
    dataset = load_truthfulqa_mc1()
    dataset = dataset.select(random.sample(range(len(dataset)), 50))
    n_threads = 12    
    
    # ga = GroupArchitecture(Topology.SINGLE, 1, PromptType.CHAIN_OF_THOUGHT)
    # accuracy = run_evaluation(ga, dataset, current_datetime, n_threads=n_threads)
    # print(f"Accuracy: {accuracy:.2%}")    
    # print(f"Results saved to results/results.csv and results/result_per_question.csv")
    
    # ga = GroupArchitecture(Topology.GROUP_CHAT, 2, PromptType.MIXED)
    # accuracy = run_evaluation(ga, dataset, current_datetime, n_threads=n_threads)
    # print(f"Accuracy: {accuracy:.2%}")    
    # print(f"Results saved to results/results.csv and results/result_per_question.csv")
    
    # ga = GroupArchitecture(Topology.ONE_ON_ONE, 2, PromptType.MIXED)
    # accuracy = run_evaluation(ga, dataset, current_datetime, n_threads=n_threads//ga.group_size)
    # print(f"Accuracy: {accuracy:.2%}")    
    # print(f"Results saved to results/results.csv and results/result_per_question.csv")
    
    # ga = GroupArchitecture(Topology.REFLECTION, 4, PromptType.MIXED)
    # accuracy = run_evaluation(ga, dataset, current_datetime, n_threads=n_threads//ga.group_size*2)
    # print(f"Accuracy: {accuracy:.2%}")    
    # print(f"Results saved to results/results.csv and results/result_per_question.csv")
    
    # ga = GroupArchitecture(Topology.GROUP_CHAT, 2, PromptType.MIXED, malicious_target='Y')
    # accuracy = run_evaluation(ga, dataset, current_datetime, n_threads=n_threads)
    # print(f"Accuracy: {accuracy:.2%}")    
    # print(f"Results saved to results/results.csv and results/result_per_question.csv")
    
    ga = GroupArchitecture(Topology.ONE_ON_ONE, 2, PromptType.MIXED, malicious_target='Y')
    accuracy = run_evaluation(ga, dataset, current_datetime, n_threads=n_threads//ga.group_size)
    print(f"Accuracy: {accuracy:.2%}")    
    print(f"Results saved to results/results.csv and results/result_per_question.csv")
    
    ga = GroupArchitecture(Topology.REFLECTION, 4, PromptType.MIXED, malicious_target='Y')
    accuracy = run_evaluation(ga, dataset, current_datetime, n_threads=n_threads//ga.group_size*2)
    print(f"Accuracy: {accuracy:.2%}")    
    print(f"Results saved to results/results.csv and results/result_per_question.csv")

if __name__ == "__main__":
    main()
