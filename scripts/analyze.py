import pandas as pd
from typing import List, Tuple
from datetime import datetime

def read_results(file_path: str) -> pd.DataFrame:
    """
    Read the result_per_question.csv file and return it as a DataFrame.

    Args:
        file_path (str): Path to the result_per_question.csv file.

    Returns:
        pd.DataFrame: DataFrame containing the results.
    """
    return pd.read_csv(file_path)

def compare_architectures(result_filepath: str, arch1: str, arch2: str, datetime_filter: str = None) -> List[Tuple[str, str]]:
    """
    Compare two architectures and find questions that arch1 answered correctly but arch2 didn't.

    Args:
        result_filepath (str): Path to the result_per_question.csv file.
        arch1 (str): Name of the first architecture.
        arch2 (str): Name of the second architecture.
        datetime_filter (str, optional): Filter results by this datetime (format: 'YYYY-MM-DD HH:MM:SS').

    Returns:
        List[Tuple[str, str]]: List of tuples containing (question_id, question) that arch1 got right but arch2 didn't.
    """
    
    df = read_results(file_path)
    
    # Apply datetime filter if provided
    if datetime_filter:
        df = df[df['current_datetime'] == datetime_filter]
    
    # Filter the DataFrame for each architecture
    df1 = df[df['architecture'] == arch1]
    df2 = df[df['architecture'] == arch2]

    # Merge the two DataFrames on question_id
    merged = pd.merge(df1, df2, on='question_id', suffixes=('_1', '_2'))

    # Find questions where arch1 is correct and arch2 is incorrect
    diff_questions = merged[(merged['is_correct_1'] == True) & (merged['is_correct_2'] == False)]

    # Return a list of tuples (question_id, question)
    return list(zip(diff_questions['question_id'], diff_questions['question_1']))

# Example usage:
file_path = 'results/result_per_question.csv'
arch1 = 'A(gc_2_cot)'
arch2 = 'A(one_1_cot)'
datetime_filter = '2024-10-20 01:54:35'
different_questions = compare_architectures(file_path, arch1, arch2, datetime_filter)
for question_id, question in different_questions:
    print(f"Question ID: {question_id}")
    print(f"Question: {question}")
    print("---")
