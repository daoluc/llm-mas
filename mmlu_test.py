import os
from dotenv import load_dotenv
import json
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
from tqdm import tqdm
from datasets import load_dataset

# Load environment variables from .env file
load_dotenv()

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


def test_agent_with_mmlu(model="gpt-4o-mini", num_questions=10):
    # Set up OpenAI API

    # Load MMLU dataset from Hugging Face
    dataset = load_dataset("cais/mmlu", "all", cache_dir=".cache")
    mmlu_data = dataset['test']

    # Get all unique question categories from MMLU dataset
    # categories = set(mmlu_data['subject'])
    # for c in categories:
    #     print(c)
    # print(f"Total number of categories: {len(categories)}")
    # print("---")
    
    correct_answers = 0
    total_questions = min(num_questions, len(mmlu_data))

    for i in tqdm(range(total_questions), desc="Testing"):
        question = mmlu_data[i]
        prompt = f"Question: {question['question']}\n\nOptions:\nA. {question['choices'][0]}\nB. {question['choices'][1]}\nC. {question['choices'][2]}\nD. {question['choices'][3]}\n\nPlease select the correct answer (A, B, C, or D)."

        response = client.chat.completions.create(model=model,
        messages=[
            {"role": "system", "content": "The following are multiple choice questions (with answers). Think step by step and then finish your answer with \"the answer is X\" where X is the correct letter choice."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=10000,
        n=1,
        stop=None,
        temperature=0.5)

        agent_answer = response.choices[0].message.content
        correct_answer = chr(65 + question['answer'])  # Convert 0-3 to A-D
        
        print(prompt)
        print(agent_answer)
        print(f"Correct answer: {correct_answer}")        

        if extract_answer_letter(agent_answer) == correct_answer:
            correct_answers += 1
        else:
            print("INCORRECT!")

        print("---")

    accuracy = correct_answers / total_questions
    print(f"MMLU Test Results:")
    print(f"Total questions: {total_questions}")
    print(f"Correct answers: {correct_answers}")
    print(f"Accuracy: {accuracy:.2%}")

    return accuracy

if __name__ == "__main__":
    test_agent_with_mmlu()
