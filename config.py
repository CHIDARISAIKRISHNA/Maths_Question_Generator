"""
Configuration settings for the Maths Question Generator.
"""

import os
from typing import Dict, Any

# Model configuration
MODEL_CONFIG = {
    "llama_version": "llama-3.1",
    "max_tokens": 800,
    "temperature": 0.7,
    "top_p": 0.9,
    "frequency_penalty": 0.0,
    "presence_penalty": 0.0
}

# Question generation settings
QUESTION_CONFIG = {
    "short_answer": {
        "default_num_questions": 3,
        "max_questions": 10,
        "difficulty_levels": ["easy", "medium", "hard"]
    },
    "mcq": {
        "default_num_questions": 3,
        "max_questions": 10,
        "default_num_options": 4,
        "max_options": 6,
        "difficulty_levels": ["easy", "medium", "hard"]
    },
    "fill_blank": {
        "default_num_questions": 3,
        "max_questions": 10,
        "difficulty_levels": ["easy", "medium", "hard"]
    }
}

# Paths for saving and loading data
DATA_PATHS = {
    "sample_contexts": "data/sample_contexts.json",
    "generated_questions": "data/generated_questions",
    "evaluation_results": "data/evaluation_results"
}

# Ensure directories exist
for path in DATA_PATHS.values():
    directory = os.path.dirname(path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)

# UI configuration
UI_CONFIG = {
    "theme": {
        "primaryColor": "#1E88E5",
        "backgroundColor": "#FFFFFF",
        "secondaryBackgroundColor": "#F0F2F6",
        "textColor": "#262730",
        "font": "sans serif"
    },
    "page_title": "Maths Question Generator",
    "page_icon": "➗",
    "layout": "wide",
    "initial_sidebar_state": "expanded"
}

# Default sample contexts
DEFAULT_CONTEXTS = {
    "algebra": "In algebra, a quadratic equation is an equation of the form ax² + bx + c = 0, where a, b, and c are constants and a ≠ 0. The solutions to this equation are given by the quadratic formula: x = (-b ± √(b² - 4ac)) / (2a). The discriminant b² - 4ac determines the nature of the solutions.",
    
    "calculus": "The fundamental theorem of calculus establishes the relationship between differentiation and integration. It states that if f is a continuous function on the closed interval [a, b] and F is an antiderivative of f on [a, b], then ∫(a to b) f(x) dx = F(b) - F(a). This theorem forms the foundation of integral calculus.",
    
    "geometry": "In Euclidean geometry, the Pythagorean theorem states that in a right triangle, the square of the length of the hypotenuse equals the sum of the squares of the lengths of the other two sides. If a and b are the legs and c is the hypotenuse, then a² + b² = c². This theorem is fundamental to trigonometry and coordinate geometry.",
    
    "probability": "In probability theory, Bayes' theorem describes the probability of an event based on prior knowledge of conditions that might be related to the event. It is stated mathematically as P(A|B) = P(B|A) × P(A) / P(B), where P(A|B) is the probability of event A occurring given that B has occurred."
}

def get_config() -> Dict[str, Any]:
    """
    Get the complete configuration.
    
    Returns:
        Dictionary containing all configuration settings
    """
    return {
        "model": MODEL_CONFIG,
        "questions": QUESTION_CONFIG,
        "data_paths": DATA_PATHS,
        "ui": UI_CONFIG,
        "default_contexts": DEFAULT_CONTEXTS
    }