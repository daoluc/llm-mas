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

def prepare_dataset(dataset):
    new_dataset = []
    for item in dataset:
        # Create a copy of the item that we can modify
        new_item = dict(item)
        new_item['mc1_targets'] = dict(item['mc1_targets'])
        
        # Create a list of tuples containing (choice, label)
        choice_label_pairs = list(zip(item['mc1_targets']['choices'], item['mc1_targets']['labels']))
        
        # Shuffle the pairs
        random.shuffle(choice_label_pairs)
        
        # Unzip the shuffled pairs back into separate lists
        shuffled_choices, shuffled_labels = zip(*choice_label_pairs)
        
        # Convert shuffled_labels to a list for mutability
        shuffled_labels = list(shuffled_labels)
        
        # Randomly pick a label 0 and set it to -1, which would be the malicious target
        zero_indices = [i for i, label in enumerate(shuffled_labels) if label == 0]
        if zero_indices:
            random_zero_index = random.choice(zero_indices)
            shuffled_labels[random_zero_index] = -1
        
        # Update the new_item with shuffled choices and labels
        new_item['mc1_targets']['choices'] = list(shuffled_choices)
        new_item['mc1_targets']['labels'] = shuffled_labels
        
        new_dataset.append(new_item)

    return new_dataset                    

def process_item(ga: GroupArchitecture, item):
    question = item['question']
    options = item['mc1_targets']['choices']
    correct_index = item['mc1_targets']['labels'].index(1)
    correct_answer = chr(65 + correct_index)
    
    # If malicious_target is set, assign it to a random incorrect choice
    if ga.malicious_target:
        malicious_index = item['mc1_targets']['labels'].index(-1)
        ga.malicious_target = chr(65 + malicious_index)

    message = f"{question}\n\nOptions:\n" + "\n".join([f"{chr(65+i)}. {opt}" for i, opt in enumerate(options)]) + "\n\nPlease select the correct option letter (A, B, C, etc.)."

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
    print(type(dataset))
    dataset = dataset.select(random.sample(range(len(dataset)), 10))
    dataset = prepare_dataset(dataset)
    n_threads = 8    
    
    # return True
    
    ga = GroupArchitecture(Topology.SINGLE, 1, PromptType.CHAIN_OF_THOUGHT)
    accuracy = run_evaluation(ga, dataset, current_datetime, n_threads=n_threads)
    print(f"Accuracy: {accuracy:.2%}")    
    print(f"Results saved to results/results.csv and results/result_per_question.csv")
    
    # ga = GroupArchitecture(Topology.GROUP_CHAT, 2, PromptType.CHAIN_OF_THOUGHT)
    # accuracy = run_evaluation(ga, dataset, current_datetime, n_threads=n_threads)
    # print(f"Accuracy: {accuracy:.2%}")    
    # print(f"Results saved to results/results.csv and results/result_per_question.csv")

    # ga = GroupArchitecture(Topology.GROUP_CHAT, 2, PromptType.STEP_BACK_ABSTRACTION)
    # accuracy = run_evaluation(ga, dataset, current_datetime, n_threads=n_threads)
    # print(f"Accuracy: {accuracy:.2%}")    
    # print(f"Results saved to results/results.csv and results/result_per_question.csv")
    
    # ga = GroupArchitecture(Topology.GROUP_CHAT, 2, PromptType.MIXED)
    # accuracy = run_evaluation(ga, dataset, current_datetime, n_threads=n_threads)
    # print(f"Accuracy: {accuracy:.2%}")    
    # print(f"Results saved to results/results.csv and results/result_per_question.csv")

    
    # ga = GroupArchitecture(Topology.GROUP_CHAT, 2, PromptType.MIXED)
    # accuracy = run_evaluation(ga, dataset, current_datetime, n_threads=n_threads)
    # print(f"Accuracy: {accuracy:.2%}")    
    # print(f"Results saved to results/results.csv and results/result_per_question.csv")
    
    ga = GroupArchitecture(Topology.CROWDSOURCING, 4, PromptType.CHAIN_OF_THOUGHT)
    accuracy = run_evaluation(ga, dataset, current_datetime, n_threads=n_threads//ga.group_size)
    print(f"Accuracy: {accuracy:.2%}")    
    print(f"Results saved to results/results.csv and results/result_per_question.csv")
    
    ga = GroupArchitecture(Topology.GROUP_CHAT, 4, PromptType.CHAIN_OF_THOUGHT)
    accuracy = run_evaluation(ga, dataset, current_datetime, n_threads=n_threads)
    print(f"Accuracy: {accuracy:.2%}")    
    print(f"Results saved to results/results.csv and results/result_per_question.csv")
    
    ga = GroupArchitecture(Topology.REFLECTION, 4, PromptType.CHAIN_OF_THOUGHT)
    accuracy = run_evaluation(ga, dataset, current_datetime, n_threads=n_threads//ga.group_size*2)
    print(f"Accuracy: {accuracy:.2%}")    
    print(f"Results saved to results/results.csv and results/result_per_question.csv")
    
    ga = GroupArchitecture(Topology.BLACKBOARD, 4, PromptType.CHAIN_OF_THOUGHT)
    accuracy = run_evaluation(ga, dataset, current_datetime, n_threads=n_threads//ga.group_size)
    print(f"Accuracy: {accuracy:.2%}")    
    print(f"Results saved to results/results.csv and results/result_per_question.csv")
    
    # ga = GroupArchitecture(Topology.GROUP_CHAT, 2, PromptType.CHAIN_OF_THOUGHT, malicious_target='Y')
    # accuracy = run_evaluation(ga, dataset, current_datetime, n_threads=n_threads)
    # print(f"Accuracy: {accuracy:.2%}")    
    # print(f"Results saved to results/results.csv and results/result_per_question.csv")
    
    # ga = GroupArchitecture(Topology.GROUP_CHAT, 2, PromptType.MIXED, malicious_target='Y')
    # accuracy = run_evaluation(ga, dataset, current_datetime, n_threads=n_threads)
    # print(f"Accuracy: {accuracy:.2%}")    
    # print(f"Results saved to results/results.csv and results/result_per_question.csv")
    
    # ga = GroupArchitecture(Topology.CROWDSOURCING, 4, PromptType.MIXED, malicious_target='Y')
    # accuracy = run_evaluation(ga, dataset, current_datetime, n_threads=n_threads//ga.group_size)
    # print(f"Accuracy: {accuracy:.2%}")    
    # print(f"Results saved to results/results.csv and results/result_per_question.csv")
    
    # ga = GroupArchitecture(Topology.GROUP_CHAT, 4, PromptType.MIXED, malicious_target='Y')
    # accuracy = run_evaluation(ga, dataset, current_datetime, n_threads=n_threads)
    # print(f"Accuracy: {accuracy:.2%}")    
    # print(f"Results saved to results/results.csv and results/result_per_question.csv")
    
    # ga = GroupArchitecture(Topology.REFLECTION, 4, PromptType.MIXED, malicious_target='Y')
    # accuracy = run_evaluation(ga, dataset, current_datetime, n_threads=n_threads//ga.group_size*2)
    # print(f"Accuracy: {accuracy:.2%}")    
    # print(f"Results saved to results/results.csv and results/result_per_question.csv")
    
    # ga = GroupArchitecture(Topology.BLACKBOARD, 4, PromptType.MIXED, malicious_target='Y')
    # accuracy = run_evaluation(ga, dataset, current_datetime, n_threads=n_threads//ga.group_size)
    # print(f"Accuracy: {accuracy:.2%}")    
    # print(f"Results saved to results/results.csv and results/result_per_question.csv")

if __name__ == "__main__":
    main()
