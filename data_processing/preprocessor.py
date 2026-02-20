import re
import nltk
from nltk.tokenize import sent_tokenize
from typing import List, Dict, Any, Tuple
import numpy as np
import sympy as sp
from sympy.parsing.sympy_parser import parse_expr
from sympy import symbols

# Download necessary NLTK resources
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

class MathPreprocessor:
    """
    Preprocesses mathematical text for question generation.
    """
    
    def __init__(self):
        """Initialize the preprocessor."""
        pass
    
    def clean_text(self, text: str) -> str:
        """
        Clean the text by removing extra whitespaces and normalizing.
        
        Args:
            text: Input text to clean
            
        Returns:
            Cleaned text
        """
        # Replace multiple spaces with a single space
        text = re.sub(r'\s+', ' ', text)
        
        # Normalize mathematical symbols
        text = text.replace('×', '*')
        text = text.replace('÷', '/')
        
        # Handle LaTeX-style math expressions
        text = self._normalize_latex(text)
        
        return text.strip()
    
    def _normalize_latex(self, text: str) -> str:
        """
        Normalize LaTeX-style mathematical expressions.
        
        Args:
            text: Input text with potential LaTeX expressions
            
        Returns:
            Text with normalized LaTeX expressions
        """
        # Replace common LaTeX math commands
        replacements = {
            r'\frac{([^{}]+)}{([^{}]+)}': r'(\1)/(\2)',
            r'\sqrt{([^{}]+)}': r'sqrt(\1)',
            r'\sin': 'sin',
            r'\cos': 'cos',
            r'\tan': 'tan',
            r'\pi': 'π',
            r'\infty': '∞'
        }
        
        for pattern, replacement in replacements.items():
            text = re.sub(pattern, replacement, text)
            
        return text
    
    def extract_math_expressions(self, text: str) -> List[str]:
        """
        Extract mathematical expressions from text.
        
        Args:
            text: Input text
            
        Returns:
            List of extracted mathematical expressions
        """
        # Pattern to match common mathematical expressions
        # This is a simplified pattern and may need refinement for complex expressions
        math_pattern = r'(\d+\s*[\+\-\*\/\^]\s*\d+|' \
                       r'$$[^()]+$$|' \
                       r'\d+\s*=\s*[^=]+|' \
                       r'[xyz]\s*=\s*[^=]+)'
        
        expressions = re.findall(math_pattern, text)
        return expressions
    
    def tokenize_sentences(self, text: str) -> List[str]:
        """
        Tokenize text into sentences.
        
        Args:
            text: Input text
            
        Returns:
            List of sentences
        """
        return sent_tokenize(text)
    
    def identify_key_concepts(self, text: str) -> List[str]:
        """
        Identify key mathematical concepts in the text.
        
        Args:
            text: Input text
            
        Returns:
            List of key mathematical concepts
        """
        # List of common mathematical concepts to look for
        math_concepts = [
            'theorem', 'lemma', 'corollary', 'proof', 'equation', 'inequality',
            'function', 'derivative', 'integral', 'limit', 'series', 'sequence',
            'vector', 'matrix', 'determinant', 'eigenvalue', 'eigenvector',
            'probability', 'statistics', 'distribution', 'hypothesis', 'test',
            'algebra', 'geometry', 'trigonometry', 'calculus', 'topology',
            'group', 'ring', 'field', 'module', 'category'
        ]
        
        found_concepts = []
        for concept in math_concepts:
            if re.search(r'\b' + concept + r'\b', text.lower()):
                found_concepts.append(concept)
                
        return found_concepts
    
    def extract_variables(self, text: str) -> List[str]:
        """
        Extract mathematical variables from text.
        
        Args:
            text: Input text
            
        Returns:
            List of variables
        """
        # Pattern to match common variable names in mathematics
        var_pattern = r'\b([a-zA-Z])\b'
        variables = re.findall(var_pattern, text)
        
        # Filter out common words that might be mistaken for variables
        common_words = ['a', 'A', 'I']
        variables = [var for var in variables if var not in common_words]
        
        return list(set(variables))
    
    def process_problem(self, problem: str) -> Dict[str, Any]:
        """
        Process a mathematical problem to extract useful information.
        
        Args:
            problem: Mathematical problem text
            
        Returns:
            Dictionary with processed information
        """
        cleaned_text = self.clean_text(problem)
        sentences = self.tokenize_sentences(cleaned_text)
        expressions = self.extract_math_expressions(cleaned_text)
        concepts = self.identify_key_concepts(cleaned_text)
        variables = self.extract_variables(cleaned_text)
        
        return {
            'cleaned_text': cleaned_text,
            'sentences': sentences,
            'expressions': expressions,
            'concepts': concepts,
            'variables': variables
        }
    
    def is_equation_solvable(self, equation_str: str) -> Tuple[bool, Any]:
        """
        Check if an equation is solvable and return the solution if possible.
        
        Args:
            equation_str: String representation of an equation
            
        Returns:
            Tuple of (is_solvable, solution)
        """
        try:
            # Try to parse the equation
            if '=' in equation_str:
                left, right = equation_str.split('=')
                left = parse_expr(left.strip())
                right = parse_expr(right.strip())
                equation = sp.Eq(left, right)
                
                # Extract variables
                vars_in_eq = list(equation.free_symbols)
                
                # If there's only one variable, try to solve
                if len(vars_in_eq) == 1:
                    solution = sp.solve(equation, vars_in_eq[0])
                    return True, solution
                else:
                    return True, None  # Equation with multiple variables
            else:
                # It's an expression, not an equation
                return False, None
        except Exception as e:
            print(f"Error analyzing equation: {e}")
            return False, None

# Example usage
if __name__ == "__main__":
    preprocessor = MathPreprocessor()
    
    # Example problem
    problem = "Find the value of x if 2x + 5 = 15. Also, calculate the value of y = x^2 - 3."
    
    processed = preprocessor.process_problem(problem)
    print("Cleaned text:", processed['cleaned_text'])
    print("Sentences:", processed['sentences'])
    print("Expressions:", processed['expressions'])
    print("Concepts:", processed['concepts'])
    print("Variables:", processed['variables'])
    
    # Check if an equation is solvable
    equation = "2x + 5 = 15"
    is_solvable, solution = preprocessor.is_equation_solvable(equation)
    if is_solvable:
        print(f"Equation {equation} is solvable. Solution: {solution}")
    else:
        print(f"Equation {equation} is not solvable or not an equation.")