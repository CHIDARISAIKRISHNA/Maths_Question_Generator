import streamlit as st
import random
import re
import json
import nltk
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any, Optional, Union
import time
import io
from PIL import Image
import pytesseract
import cv2
import numpy as np

# Import our modules
from question_generators.short_answer import ShortAnswerGenerator
from question_generators.mcq import MCQGenerator
from question_generators.fill_blank import FillBlankGenerator

# Download necessary NLTK resources
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Set page configuration
st.set_page_config(
    page_title="Maths Question Generator",
    page_icon="➗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'generated_questions' not in st.session_state:
    st.session_state.generated_questions = {
        'short_answer': [],
        'mcq': [],
        'fill_blank': []
    }
if 'current_page' not in st.session_state:
    st.session_state.current_page = 'home'

if 'current_page' not in st.session_state:
    st.session_state.current_page = 'select_context'

if 'selected_question_types' not in st.session_state:
    st.session_state.selected_question_types = []

if 'context_source' not in st.session_state:
    st.session_state.context_source = 'sample'

if 'custom_context' not in st.session_state:
    st.session_state.custom_context = ""

if 'dataset_problems' not in st.session_state:
    st.session_state.dataset_problems = []

if 'dataset_loaded' not in st.session_state:
    st.session_state.dataset_loaded = False

if 'generation_warnings' not in st.session_state:
    st.session_state.generation_warnings = {}

if 'extracted_text' not in st.session_state:
    st.session_state.extracted_text = ""

# Sample mathematical contexts
sample_contexts = {
    "algebra": (
        "In algebra, a quadratic equation is an equation of the form ax² + bx + c = 0, "
        "where a, b, and c are real constants and a ≠ 0. The solutions to this equation are given by "
        "the quadratic formula: x = (-b ± √(b² - 4ac)) / (2a). The discriminant b² - 4ac determines "
        "the nature of the solutions: if b² - 4ac > 0 there are two distinct real roots, if b² - 4ac = 0 "
        "there is one repeated real root, and if b² - 4ac < 0 the roots are complex. Quadratic equations "
        "can be solved by factoring, completing the square, using the quadratic formula, or graphing the "
        "parabola y = ax² + bx + c and finding the x-intercepts. The axis of symmetry of the parabola is "
        "given by x = -b / (2a), and the vertex represents the maximum or minimum value of the quadratic function."
    ),
    
    "calculus": (
        "The fundamental theorem of calculus establishes the relationship between differentiation and integration. "
        "It states that if f is a continuous function on the closed interval [a, b] and F is an antiderivative of f "
        "on [a, b], then ∫ₐᵇ f(x) dx = F(b) - F(a). The derivative of the integral function G(x) = ∫ₐˣ f(t) dt "
        "satisfies G'(x) = f(x). Limits are used to define both the derivative and the definite integral. "
        "The derivative f'(x) represents the instantaneous rate of change of f at x and is defined as "
        "f'(x) = limₕ→0 (f(x + h) - f(x)) / h when this limit exists. Techniques such as the chain rule, product rule, "
        "and quotient rule are used to differentiate more complicated functions, while substitution and integration by parts "
        "are key methods for evaluating integrals. Applications include computing velocities and accelerations, areas under curves, "
        "and accumulated quantities in physics and engineering."
    ),
    
    "geometry": (
        "In Euclidean geometry, the Pythagorean theorem states that in a right triangle with legs of length a and b "
        "and hypotenuse of length c, the relationship a² + b² = c² holds. This theorem allows us to compute distances "
        "in the Cartesian plane using the distance formula d = √[(x₂ - x₁)² + (y₂ - y₁)²]. The midpoint of a segment "
        "with endpoints (x₁, y₁) and (x₂, y₂) is given by ((x₁ + x₂)/2, (y₁ + y₂)/2). The slope of a line passing through "
        "two distinct points is m = (y₂ - y₁) / (x₂ - x₁), and the equation of a line with slope m and y-intercept b is "
        "y = mx + b. Angles in a triangle sum to 180°, and similar triangles have proportional corresponding sides and equal "
        "corresponding angles. The area of a circle of radius r is πr² and its circumference is 2πr; these formulas are "
        "fundamental in plane geometry and appear frequently in coordinate geometry problems."
    ),
    
    "probability": (
        "In probability theory, an experiment has a sample space S consisting of all possible outcomes. An event A is a subset "
        "of S, and its probability P(A) satisfies 0 ≤ P(A) ≤ 1 and P(S) = 1. For equally likely outcomes, P(A) = |A| / |S|. "
        "If events A and B are independent, then P(A ∩ B) = P(A)P(B). The conditional probability of A given B with P(B) > 0 "
        "is defined as P(A|B) = P(A ∩ B) / P(B). Bayes' theorem describes how to update probabilities using new information and "
        "is given by P(A|B) = P(B|A)P(A) / P(B). Random variables assign numerical values to outcomes, and their distributions "
        "can be discrete, such as the binomial distribution, or continuous, such as the normal distribution. The expected value "
        "E[X] represents the long-run average of X, while the variance Var(X) measures how spread out the values of X are around "
        "the mean. These concepts are central to statistical reasoning and risk analysis."
    )
}

# Function to extract text from uploaded image
def extract_text_from_image(image_file):
    """Extract text from an uploaded image file using OCR"""
    try:
        # Read the image file
        image_bytes = image_file.read()
        image = Image.open(io.BytesIO(image_bytes))
        
        # Convert PIL Image to OpenCV format
        image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Preprocess the image to improve OCR accuracy
        # Convert to grayscale
        gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
        
        # Apply thresholding to handle different lighting conditions
        _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Perform OCR on the preprocessed image
        extracted_text = pytesseract.image_to_string(thresh)
        
        # Clean up the extracted text
        extracted_text = extracted_text.strip()
        
        return extracted_text
    except Exception as e:
        st.error(f"Error extracting text from image: {e}")
        return ""

# Function to load dataset
def load_jsonl_dataset(file_path):
    """Load problems and solutions from the JSONL dataset"""
    try:
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line))
        
        # Process the data into a standard format
        processed_data = []
        for item in data:
            if "question" in item:
                # Extract the question
                question_text = item.get("question", "")
                
                # Extract the ground truth answer
                ground_truth = item.get("ground_truth", "")
                
                # Extract any solution information
                solution = ""
                for key in ["solution", "answer", "ground_truth"]:
                    if key in item and item[key]:
                        solution = item[key]
                        break
                
                # Extract the category (if available)
                category = "Algebra"  # Default category
                
                # Create a standardized problem entry
                problem_entry = {
                    "problem": question_text,
                    "solution": solution,
                    "category": category
                }
                
                processed_data.append(problem_entry)
        
        return processed_data
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        return []

# Custom context processing function
def process_custom_context(context):
    """Process a custom context to extract mathematical concepts"""
    # Extract key mathematical concepts, definitions, formulas
    concepts = []
    
    # Look for equations (contains = sign)
    equations = re.findall(r'[^.!?]*=\s*[^.!?]*[.!?]', context)
    if equations:
        for eq in equations:
            concepts.append({
                "type": "equation", 
                "text": eq.strip()
            })
    
    # Look for definitions (contains "is defined as" or "refers to" or "is a")
    definitions = re.findall(r'[^.!?]*(?:is defined as|refers to|is a)[^.!?]*[.!?]', context)
    if definitions:
        for defn in definitions:
            concepts.append({
                "type": "definition", 
                "text": defn.strip()
            })
    
    # Look for formulas (contains formula names)
    formula_patterns = ["theorem", "formula", "rule", "law", "principle"]
    for pattern in formula_patterns:
        formulas = re.findall(r'[^.!?]*' + pattern + r'[^.!?]*[.!?]', context, re.IGNORECASE)
        for formula in formulas:
            concepts.append({
                "type": "formula", 
                "text": formula.strip()
            })
    
    # Extract all sentences if we didn't find specific concepts
    if not concepts:
        sentences = re.split(r'[.!?]', context)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
        for sentence in sentences:
            concepts.append({
                "type": "statement",
                "text": sentence
            })
    
    return concepts

def context_has_math_content(context: str) -> bool:
    """Check if context contains mathematical content (equations, numbers, formulas) for math-only question generation."""
    if not context or len(context.strip()) < 10:
        return False
    # Equations (e.g. x = 5, a² + b² = c²)
    if re.search(r'[=≠≈≤≥]', context):
        return True
    # Numbers (including decimals and negative)
    if re.search(r'[-+]?\d*\.?\d+', context):
        return True
    # Math keywords (formulas, theorems, definitions)
    math_keywords = ["formula", "theorem", "equation", "function", "derivative", "integral", "probability", "sum", "product", "variable", "solve", "root", "factor", "quadratic", "linear", "geometry", "angle", "triangle"]
    context_lower = context.lower()
    if any(kw in context_lower for kw in math_keywords):
        return True
    return False

def is_duplicate_question(question: Dict[str, Any], existing_questions: List[Dict[str, Any]]) -> bool:
    """Check if a newly generated question already exists in the list (exact text match)."""
    if not isinstance(question, dict):
        return False
    new_text = str(question.get("question", "")).strip()
    if not new_text:
        return False
    for q in existing_questions:
        if str(q.get("question", "")).strip() == new_text:
            return True
    return False

# Initialize question generators
short_answer_generator = ShortAnswerGenerator()
mcq_generator = MCQGenerator()
fill_blank_generator = FillBlankGenerator()

# Helper functions for the UI
def navigate_to(page: str) -> None:
    """Navigate to a specific page in the app."""
    st.session_state.current_page = page
def display_home_page() -> None:
    """Display the home page."""
    st.title("Maths Question Generator")
    st.markdown("""
    Welcome to the Maths Question Generator! This tool helps you generate various types of mathematics questions from a given context.
    
    ### Features:
    - Generate short answer questions
    - Generate multiple-choice questions (MCQs)
    - Generate fill-in-the-blank questions
    - Use sample mathematical contexts, provide your own, or load from a dataset
    
    ### Get Started:
    Click the button below to start generating questions.
    """)
    
    if st.button("Get Started", key="get_started"):
        # Start the flow at step 1 in the left sidebar: Select Context
        navigate_to('select_context')


def display_select_context_page() -> None:
    """Display the page for selecting the context."""
    st.title("Select Context")
    st.markdown("Choose a context source:")
    
    # Radio buttons for context source
    context_source = st.radio(
        "Context Source",
        ["Sample Contexts", "Custom Input", "Load from Dataset"],
        index=0 if st.session_state.context_source == 'sample' else 
              1 if st.session_state.context_source == 'custom' else 2
    )
    
    if context_source == "Sample Contexts":
        st.session_state.context_source = 'sample'
        # Display sample contexts
        sample_context = st.selectbox(
            "Select a Sample Context",
            list(sample_contexts.keys()),
            format_func=lambda x: x.capitalize()
        )
        
        st.markdown("### Preview:")
        st.info(sample_contexts[sample_context])
        
        st.session_state.custom_context = sample_contexts[sample_context]
        
    elif context_source == "Custom Input":
        st.session_state.context_source = 'custom'
        # Custom context input with image upload option
        st.markdown("### Enter Your Mathematical Context")
        
        # Add image upload option
        uploaded_file = st.file_uploader("Upload an image containing mathematical content (optional)", 
                                         type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            # Display the uploaded image
            st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)
            
            # Extract text from the image
            if st.button("Extract Text from Image"):
                with st.spinner("Extracting text from image..."):
                    extracted_text = extract_text_from_image(uploaded_file)
                    st.session_state.extracted_text = extracted_text
                    st.session_state.custom_context = extracted_text
            
            # Display extracted text
            if st.session_state.extracted_text:
                st.markdown("### Extracted Text:")
                st.info(st.session_state.extracted_text)
        
        # Text area for manual input or editing extracted text
        st.session_state.custom_context = st.text_area(
            "Enter or edit mathematical text",
            value=st.session_state.custom_context,
            height=200,
            placeholder="Enter mathematical text, formulas, or problems here..."
        )
    
    else:  # Load from Dataset
        st.session_state.context_source = 'dataset'
        
        # Option to use built-in datasets
        st.markdown("### Select Dataset:")
        dataset_option = st.selectbox(
            "Select Dataset",
            ["None", "example_model_solutions.jsonl", "train_socratic.jsonl"]
        )
        
        if dataset_option != "None":
            dataset_path = os.path.join("data", dataset_option)
            
            with st.spinner(f"Loading {dataset_option}..."):
                if os.path.exists(dataset_path):
                    problems_data = load_jsonl_dataset(dataset_path)
                    if problems_data:
                        st.session_state.dataset_problems = problems_data
                        st.session_state.dataset_loaded = True
                        st.success(f"Successfully loaded {len(problems_data)} problems from {dataset_option}.")
                    else:
                        st.error(f"No problems found in {dataset_option}.")
                else:
                    st.error(f"File not found: {dataset_path}")
        
        # If dataset is loaded, show a message and proceed to next step
        if st.session_state.dataset_loaded:
            st.info(f"Dataset loaded with {len(st.session_state.dataset_problems)} problems. Click 'Next' to specify the number of questions to generate.")
            
            # Set a placeholder context (will be replaced with random problems during generation)
            st.session_state.custom_context = "Dataset problems will be used for generation"
    
    # Navigation buttons (Back to Home, Next to Question Types)
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Back", key="back_to_home_from_context"):
            navigate_to('home')
    with col2:
        if st.button("Next", key="next_to_question_types"):
            if st.session_state.context_source == 'dataset' and not st.session_state.dataset_loaded:
                st.error("Dataset could not be loaded. Please try another option.")
            elif st.session_state.custom_context.strip():
                navigate_to('select_question_types')
            else:
                st.error("Please provide a context.")

def display_select_question_types_page() -> None:
    """Display the page for selecting question types."""
    st.title("Select Question Types")
    
    # Different question type options based on context source
    if st.session_state.context_source == 'dataset':
        st.markdown("Choose the types of questions you want to generate from the dataset:")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            short_answer = st.checkbox("Short Answer Questions", value=False)
        
        with col2:
            mcq = st.checkbox("Multiple Choice Questions (MCQs)", value=False)
        
        with col3:
            fill_blank = st.checkbox("Fill-in-the-Blank Questions", value=False)
        
        # Store selected question types
        selected_types = []
        if short_answer:
            selected_types.append('short_answer')
        if mcq:
            selected_types.append('mcq')
        if fill_blank:
            selected_types.append('fill_blank')
    elif st.session_state.context_source == 'custom':
        # For custom input, disable Short Answer type completely; only MCQ and Fill-in-the-Blank are available
        st.markdown("For custom input, Short Answer questions are not available. Choose from the following types:")
        
        col1, col2 = st.columns(2)
        
        with col1:
            mcq = st.checkbox("Multiple Choice Questions (MCQs)", value=False)
        
        with col2:
            fill_blank = st.checkbox("Fill-in-the-Blank Questions", value=False)
        
        selected_types = []
        if mcq:
            selected_types.append('mcq')
        if fill_blank:
            selected_types.append('fill_blank')
    else:  # Sample contexts
        # For sample contexts, only MCQ and Fill-in-the-Blank are available
        st.markdown("For sample contexts, Short Answer questions are not available. Choose from the following types:")
        
        col1, col2 = st.columns(2)
        
        with col1:
            mcq = st.checkbox("Multiple Choice Questions (MCQs)", value=False)
        
        with col2:
            fill_blank = st.checkbox("Fill-in-the-Blank Questions", value=False)
        
        # Store selected question types
        selected_types = []
        if mcq:
            selected_types.append('mcq')
        if fill_blank:
            selected_types.append('fill_blank')
    
    st.session_state.selected_question_types = selected_types
    
    # Navigation buttons
    col1, col2 = st.columns(2)
    
    with col1:
        # Go back one step in the main flow: Select Question Types <- Select Context
        if st.button("Back", key="back_to_context_from_types"):
            navigate_to('select_context')
    
    with col2:
        if st.button("Next", key="next_to_generate"):
            if selected_types:
                navigate_to('generate_questions')
            else:
                st.error("Please select at least one question type.")

def display_generate_questions_page() -> None:
    """Display the page for generating questions."""
    st.title("Generate Questions")
    
    # Reset generation warnings
    st.session_state.generation_warnings = {}
    
    # Different UI based on context source
    if st.session_state.context_source == 'dataset':
        st.markdown("### Generate Questions from Dataset")
        
        # Number of questions per type
        num_questions_per_type = {}
        
        # Create columns for each question type
        if st.session_state.selected_question_types:
            cols = st.columns(len(st.session_state.selected_question_types))
            
            for i, question_type in enumerate(st.session_state.selected_question_types):
                with cols[i]:
                    display_name = {
                        'short_answer': 'Short Answer',
                        'mcq': 'Multiple Choice',
                        'fill_blank': 'Fill-in-the-Blank'
                    }[question_type]
                    
                    num_questions_per_type[question_type] = st.number_input(
                        f"{display_name} Questions",
                        min_value=1,
                        max_value=100,  # allow larger requests; we'll warn if not all can be generated
                        value=1,
                        step=1,
                        key=f"dataset_{question_type}_count"
                    )
        
        # Generate questions button
        if st.button("Generate Questions", key="generate_questions_dataset"):
            with st.spinner("Generating questions..."):
                # Randomly select problems from the dataset
                if len(st.session_state.dataset_problems) > 0:
                    # Reset generated questions
                    st.session_state.generated_questions = {
                        'short_answer': [],
                        'mcq': [],
                        'fill_blank': []
                    }
                    
                    # Get a pool of problems to work with
                    total_questions_needed = sum(num_questions_per_type.values())
                    pool_size = min(len(st.session_state.dataset_problems), total_questions_needed * 3)
                    
                    selected_problems = random.sample(
                        st.session_state.dataset_problems, 
                        pool_size
                    )
                    
                    # Generate questions for each selected type
                    for question_type in st.session_state.selected_question_types:
                        target_count = num_questions_per_type[question_type]
                        attempts = 0
                        max_attempts = min(50, len(selected_problems) * 2)
                        
                        # Try to generate the requested number of UNIQUE questions
                        while (len(st.session_state.generated_questions[question_type]) < target_count and 
                               attempts < max_attempts):
                            
                            # Select a random problem
                            problem_data = random.choice(selected_problems)
                            
                            try:
                                # Generate a question based on type
                                if question_type == 'short_answer':
                                    question = short_answer_generator.generate_from_problem(problem_data)
                                elif question_type == 'mcq':
                                    question = mcq_generator.generate_from_problem(problem_data)
                                elif question_type == 'fill_blank':
                                    question = fill_blank_generator.generate_from_problem(problem_data)
                                
                                # Only add if this question text has not been seen before
                                if not is_duplicate_question(question, st.session_state.generated_questions[question_type]):
                                    st.session_state.generated_questions[question_type].append(question)
                            except Exception as e:
                                st.error(f"Error generating {question_type} question: {e}")
                            
                            attempts += 1
                        
                        # Record warning if we couldn't generate enough questions
                        if len(st.session_state.generated_questions[question_type]) < target_count:
                            display_name = {
                                'short_answer': 'Short Answer',
                                'mcq': 'Multiple Choice',
                                'fill_blank': 'Fill-in-the-Blank'
                            }[question_type]
                            
                            st.session_state.generation_warnings[question_type] = {
                                'requested': target_count,
                                'generated': len(st.session_state.generated_questions[question_type]),
                                'display_name': display_name
                            }
                    
                    # Navigate to the results page
                    navigate_to('display_results')
                else:
                    st.error("No problems available in the dataset. Please select a different dataset.")
    
    else:  # Sample or Custom context
        st.markdown("### Generate Questions from Context")
        
        # Number of questions per type
        num_questions_per_type = {}
        
        # Create columns for each question type
        if st.session_state.selected_question_types:
            cols = st.columns(len(st.session_state.selected_question_types))
            
            for i, question_type in enumerate(st.session_state.selected_question_types):
                with cols[i]:
                    display_name = {
                        'short_answer': 'Short Answer',
                        'mcq': 'Multiple Choice',
                        'fill_blank': 'Fill-in-the-Blank'
                    }[question_type]
                    
                    num_questions_per_type[question_type] = st.number_input(
                        f"{display_name} Questions",
                        min_value=1,
                        max_value=100,  # allow larger requests; we'll warn if not all can be generated
                        value=1,
                        step=1,
                        key=f"context_{question_type}_count"
                    )
        
        # Display the context
        st.markdown("### Context:")
        st.info(st.session_state.custom_context)
        st.caption("Questions are generated from mathematical content (formulas, equations, definitions).")
        
        # Generate questions button
        if st.button("Generate Questions", key="generate_questions_context"):
            # Validate: context must contain mathematical content for math-only generation
            if not context_has_math_content(st.session_state.custom_context):
                st.error("The context does not appear to contain mathematical content (e.g. formulas, equations, numbers). Please paste a maths-related paragraph to generate mathematics questions.")
            else:
                with st.spinner("Generating questions..."):
                    # Reset generated questions
                    st.session_state.generated_questions = {
                        'short_answer': [],
                        'mcq': [],
                        'fill_blank': []
                    }
                    
                    # Process the context to extract concepts
                    context_concepts = process_custom_context(st.session_state.custom_context)
                    
                    # If no concepts found, use the entire context
                    if not context_concepts:
                        context_concepts = [{
                            "type": "full_context",
                            "text": st.session_state.custom_context
                        }]
                    
                    # Maximum attempts per question
                    max_attempts_per_question = 15
                    
                    # For each selected question type
                    for question_type in st.session_state.selected_question_types:
                        target_count = num_questions_per_type[question_type]
                        attempts = 0
                        max_attempts = max_attempts_per_question * target_count * 2
                        
                        # Try to generate the requested number of UNIQUE questions
                        while (len(st.session_state.generated_questions[question_type]) < target_count and 
                               attempts < max_attempts):
                            
                            # Get a random concept to use
                            concept_data = random.choice(context_concepts)
                            concept_text = concept_data["text"]
                            
                            try:
                                # Generate a question based on type (math-focused)
                                if question_type == 'short_answer':
                                    question = short_answer_generator.generate_from_context(concept_text)
                                elif question_type == 'mcq':
                                    question = mcq_generator.generate_from_context(concept_text)
                                elif question_type == 'fill_blank':
                                    question = fill_blank_generator.generate_from_context(concept_text)
                                
                                # Only add if this question text has not been seen before
                                if not is_duplicate_question(question, st.session_state.generated_questions[question_type]):
                                    st.session_state.generated_questions[question_type].append(question)
                            except Exception as e:
                                pass
                            
                            attempts += 1
                        
                        # Record warning when we cannot generate the requested number
                        if len(st.session_state.generated_questions[question_type]) < target_count:
                            display_name = {
                                'short_answer': 'Short Answer',
                                'mcq': 'Multiple Choice',
                                'fill_blank': 'Fill-in-the-Blank'
                            }[question_type]
                            
                            st.session_state.generation_warnings[question_type] = {
                                'requested': target_count,
                                'generated': len(st.session_state.generated_questions[question_type]),
                                'display_name': display_name
                            }
                    
                    # Check if we generated any questions at all
                    if all(len(q) == 0 for q in st.session_state.generated_questions.values()):
                        st.error("Unable to generate questions from the provided context. Please try a different context or provide more mathematical content.")
                    else:
                        # Navigate to the results page
                        navigate_to('display_results')

    # Back navigation: go to previous step in main flow (Select Question Types)
    st.markdown("---")
    if st.button("Back", key="back_to_types_from_generate"):
        navigate_to('select_question_types')

def display_results_page() -> None:
    """Display the results page with generated questions."""
    st.title("Generated Questions")
    
    # Display any generation warnings at the top (requested more questions than could be generated)
    if st.session_state.generation_warnings:
        warning_container = st.container()
        with warning_container:
            for question_type, warning in st.session_state.generation_warnings.items():
                st.warning(
                    f"⚠️ Based on the given context, it can generate up to "
                    f"{warning['generated']} {warning['display_name']} question(s) only "
                    f"(you requested {warning['requested']})."
                )
    
    # Create tabs for each question type
    tabs = []
    for question_type in st.session_state.selected_question_types:
        display_name = {
            'short_answer': 'Short Answer',
            'mcq': 'Multiple Choice',
            'fill_blank': 'Fill-in-the-Blank'
        }[question_type]
        tabs.append(display_name)
    
    if tabs:
        tab1, *other_tabs = st.tabs(tabs)
        
        # Display questions in each tab
        for i, question_type in enumerate(st.session_state.selected_question_types):
            tab = tab1 if i == 0 else other_tabs[i-1]
            
            with tab:
                questions = st.session_state.generated_questions[question_type]
                
                if not questions:
                    st.warning(f"No {question_type.replace('_', ' ')} questions could be generated from the given context.")
                    continue
                
                for j, question in enumerate(questions, 1):
                    st.markdown(f"### Question {j}")
                    
                    if question_type == 'short_answer':
                        st.markdown(f"**Q:** {question['question']}")
                        
                        # Toggle for showing/hiding the answer
                        if st.checkbox(f"Show Answer {j}", key=f"show_answer_{question_type}_{j}"):
                            st.success(f"**A:** {question['answer']}")
                    
                    elif question_type == 'mcq':
                        st.markdown(f"**Q:** {question['question']}")
                        
                        # Display options
                        for option in question['options']:
                            st.markdown(f"{option['letter']}) {option['text']}")
                        
                        # Toggle for showing/hiding the correct answer
                        if st.checkbox(f"Show Correct Answer {j}", key=f"show_answer_{question_type}_{j}"):
                            st.success(f"**Correct Answer:** {question['correct_answer']}")
                    
                    elif question_type == 'fill_blank':
                        st.markdown(f"**Q:** {question['question']}")
                        
                        # Toggle for showing/hiding the answer
                        if st.checkbox(f"Show Answer {j}", key=f"show_answer_{question_type}_{j}"):
                            st.success(f"**A:** {question['answer']}")
                    
                    st.markdown("---")
                
                # Export button
                if st.button(f"Export {tabs[i]} Questions", key=f"export_{question_type}"):
                    # Create a JSON string for download
                    json_str = json.dumps({
                        'question_type': question_type,
                        'questions': questions
                    }, indent=2)
                    
                    # Create a download button
                    st.download_button(
                        label=f"Download {tabs[i]} Questions as JSON",
                        data=json_str,
                        file_name=f"{question_type}_questions.json",
                        mime="application/json",
                        key=f"download_{question_type}"
                    )
    
    # Navigation buttons
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Back to Generate", key="back_to_generate"):
            navigate_to('generate_questions')
    
    with col2:
        if st.button("Start Over", key="start_over"):
            # Reset session state
            st.session_state.generated_questions = {
                'short_answer': [],
                'mcq': [],
                'fill_blank': []
            }
            st.session_state.selected_question_types = []
            st.session_state.context_source = 'sample'
            st.session_state.custom_context = ""
            st.session_state.generation_warnings = {}
            st.session_state.extracted_text = ""
            # Keep dataset loaded for faster reuse
            
            navigate_to('home')

# Main app logic
def main():
    # Sidebar
    with st.sidebar:
        st.title("Navigation")
        if st.button("Home", key="nav_home"):
            navigate_to('home')
        
        if st.button("Select Context", key="nav_context"):
            navigate_to('select_context')
        
        if st.button("Select Question Types", key="nav_question_types"):
            if st.session_state.custom_context.strip():
                navigate_to('select_question_types')
            else:
                st.error("Please select a context first.")
        
        if st.button("Generate Questions", key="nav_generate"):
            if st.session_state.custom_context.strip() and st.session_state.selected_question_types:
                navigate_to('generate_questions')
            else:
                st.error("Please select context and question types first.")
        
        if st.button("View Results", key="nav_results"):
            if any(st.session_state.generated_questions.values()):
                navigate_to('display_results')
            else:
                st.error("No questions have been generated yet.")
        
        st.markdown("---")
        st.markdown("### About")
        st.markdown("""
        This application generates mathematics questions from various sources.
        
        It supports:
        - Short answer questions
        - Multiple-choice questions
        - Fill-in-the-blank questions
        - Loading problems from a dataset
        - Extracting text from images
        
        Version 3.0 - Enhanced with image processing and improved mathematical question generation
        """)
    
    if st.session_state.current_page == 'home':
        display_home_page()    
    elif st.session_state.current_page == 'select_context':
        display_select_context_page()
    elif st.session_state.current_page == 'select_question_types':
        display_select_question_types_page()
    elif st.session_state.current_page == 'generate_questions':
        display_generate_questions_page()
    elif st.session_state.current_page == 'display_results':
        display_results_page()

if __name__ == "__main__":
    main()