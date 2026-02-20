import re
import random
import string
import json
import os
from typing import List, Dict, Any, Union, Optional

def clean_text(text: str) -> str:
    """
    Clean text by removing extra whitespace and normalizing.
    
    Args:
        text: Input text to clean
        
    Returns:
        Cleaned text
    """
    # Replace multiple spaces with a single space
    text = re.sub(r'\s+', ' ', text)
    
    # Remove leading/trailing whitespace
    text = text.strip()
    
    return text

def format_question_for_display(question: Dict[str, Any], question_type: str) -> str:
    """
    Format a question for display.
    
    Args:
        question: Question dictionary
        question_type: Type of question (short_answer, mcq, fill_blank)
        
    Returns:
        Formatted question string
    """
    if question_type == "short_answer":
        return f"Q: {question['question']}\nA: {question['answer']}"
    
    elif question_type == "mcq":
        formatted = f"Q: {question['question']}\n"
        for option in question['options']:
            formatted += f"{option['letter']}) {option['text']}\n"
        formatted += f"Correct Answer: {question['correct_answer']}"
        
        if 'explanation' in question:
            formatted += f"\nExplanation: {question['explanation']}"
        
        return formatted
    
    elif question_type == "fill_blank":
        return f"Q: {question['question']}\nA: {question['answer']}"
    
    else:
        return str(question)

def save_questions_to_file(questions: List[Dict[str, Any]], file_path: str, question_type: str) -> None:
    """
    Save generated questions to a file.
    
    Args:
        questions: List of question dictionaries
        file_path: Path to save the file
        question_type: Type of questions (short_answer, mcq, fill_blank)
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    # Save as JSON
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump({
            'question_type': question_type,
            'questions': questions
        }, f, indent=2)
    
    print(f"Saved {len(questions)} {question_type} questions to {file_path}")

def load_questions_from_file(file_path: str) -> Dict[str, Any]:
    """
    Load questions from a file.
    
    Args:
        file_path: Path to the file
        
    Returns:
        Dictionary containing question type and questions
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"Loaded {len(data['questions'])} {data['question_type']} questions from {file_path}")
        return data
    except Exception as e:
        print(f"Error loading questions from {file_path}: {e}")
        return {'question_type': 'unknown', 'questions': []}

def shuffle_mcq_options(question: Dict[str, Any]) -> Dict[str, Any]:
    """
    Shuffle the options of an MCQ while preserving the correct answer.
    
    Args:
        question: MCQ question dictionary
        
    Returns:
        Question with shuffled options
    """
    if 'options' not in question or 'correct_answer' not in question:
        return question
    
    # Get the correct option
    correct_option = None
    for option in question['options']:
        if option['letter'] == question['correct_answer']:
            correct_option = option
            break
    
    if not correct_option:
        return question
    
    # Shuffle the options
    options = question['options'].copy()
    random.shuffle(options)
    
    # Reassign letters
    letters = ['A', 'B', 'C', 'D', 'E', 'F'][:len(options)]
    new_correct_letter = None
    
    for i, option in enumerate(options):
        if option == correct_option:
            new_correct_letter = letters[i]
        option['letter'] = letters[i]
    
    # Update the question
    shuffled_question = question.copy()
    shuffled_question['options'] = options
    shuffled_question['correct_answer'] = new_correct_letter
    
    return shuffled_question

def generate_unique_id() -> str:
    """
    Generate a unique ID for questions.
    
    Returns:
        Unique ID string
    """
    # Generate a random string of letters and digits
    letters_and_digits = string.ascii_letters + string.digits
    return ''.join(random.choice(letters_and_digits) for _ in range(8))

def add_metadata_to_question(question: Dict[str, Any], metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    Add metadata to a question.
    
    Args:
        question: Question dictionary
        metadata: Metadata to add
        
    Returns:
        Question with added metadata
    """
    question_with_metadata = question.copy()
    
    # Add a unique ID if not present
    if 'id' not in question_with_metadata:
        question_with_metadata['id'] = generate_unique_id()
    
    # Add metadata
    if 'metadata' not in question_with_metadata:
        question_with_metadata['metadata'] = {}
    
    question_with_metadata['metadata'].update(metadata)
    
    return question_with_metadata

def filter_questions_by_difficulty(questions: List[Dict[str, Any]], difficulty: str) -> List[Dict[str, Any]]:
    """
    Filter questions by difficulty level.
    
    Args:
        questions: List of question dictionaries
        difficulty: Difficulty level to filter by
        
    Returns:
        Filtered list of questions
    """
    return [
        q for q in questions 
        if 'metadata' in q and 'difficulty' in q['metadata'] and q['metadata']['difficulty'] == difficulty
    ]

def get_question_statistics(questions: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Get statistics about a list of questions.
    
    Args:
        questions: List of question dictionaries
        
    Returns:
        Dictionary of statistics
    """
    # Count questions by type
    question_types = {}
    for q in questions:
        if 'metadata' in q and 'type' in q['metadata']:
            q_type = q['metadata']['type']
            question_types[q_type] = question_types.get(q_type, 0) + 1
    
    # Count questions by difficulty
    difficulties = {}
    for q in questions:
        if 'metadata' in q and 'difficulty' in q['metadata']:
            difficulty = q['metadata']['difficulty']
            difficulties[difficulty] = difficulties.get(difficulty, 0) + 1
    
    return {
        'total': len(questions),
        'by_type': question_types,
        'by_difficulty': difficulties
    }

# Example usage
if __name__ == "__main__":
    # Example MCQ
    mcq = {
        'question': 'What is the value of x in the equation 2x + 5 = 15?',
        'options': [
            {'letter': 'A', 'text': 'x = 5'},
            {'letter': 'B', 'text': 'x = 10'},
            {'letter': 'C', 'text': 'x = 15'},
            {'letter': 'D', 'text': 'x = 20'}
        ],
        'correct_answer': 'A'
    }
    
    # Format for display
    formatted = format_question_for_display(mcq, 'mcq')
    print("Formatted MCQ:")
    print(formatted)
    print()
    
    # Shuffle options
    shuffled = shuffle_mcq_options(mcq)
    formatted_shuffled = format_question_for_display(shuffled, 'mcq')
    print("Shuffled MCQ:")
    print(formatted_shuffled)
    print()
    
    # Add metadata
    with_metadata = add_metadata_to_question(mcq, {
        'type': 'mcq',
        'difficulty': 'easy',
        'topic': 'algebra',
        'subtopic': 'linear equations'
    })
    
    print("MCQ with metadata:")
    print(json.dumps(with_metadata, indent=2))