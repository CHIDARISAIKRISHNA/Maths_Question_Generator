import re
import random
from typing import Dict, Any

class FillBlankGenerator:
    """Generates fill-in-the-blank questions"""
    
    def generate_from_problem(self, problem: Dict[str, str], difficulty: str = "medium") -> Dict[str, Any]:
        """Generate a fill-in-the-blank question from a problem"""
        question_text = problem.get("problem", "")
        solution_text = problem.get("solution", "")
        category = problem.get("category", "Algebra")
        
        # Clean solution text - remove any "A:" prefix
        if "A:" in solution_text:
            solution_text = solution_text.replace("A:", "").strip()
        
        # Extract numerical values from the solution
        numerical_matches = re.findall(r'([-+]?\d*\.?\d+)', solution_text)
        
        if numerical_matches and len(question_text) > 10:
            # Use the last numerical result as the answer
            answer = numerical_matches[-1]
            
            # Format the question to include only a blank (no extra English phrase)
            question_with_blank = question_text + " ________."
            
            return {
                "question": question_with_blank,
                "answer": answer,
                "difficulty": difficulty
            }
        elif "=" in question_text:
            # Try to create a fill in the blank for an equation
            parts = question_text.split("=")
            if len(parts) >= 2:
                equation_left = parts[0].strip()
                equation_right = parts[1].strip()
                
                # Replace a term in the equation with a blank
                if len(equation_left.split()) > 1:
                    terms = equation_left.split()
                    replaced_term = random.choice(terms)
                    blank_equation = equation_left.replace(replaced_term, "________", 1)
                    
                    return {
                        "question": f"{blank_equation} = {equation_right}",
                        "answer": replaced_term,
                        "difficulty": difficulty
                    }
                else:
                    return {
                        "question": f"{equation_left} = ________",
                        "answer": equation_right,
                        "difficulty": difficulty
                    }
            else:
                # If no good candidates found, create a generic fill-in-the-blank
                return {
                    "question": f"Complete this problem: {question_text} = ________",
                    "answer": "Cannot determine",
                    "difficulty": difficulty
                }
        else:
            # Handle word problems - try to find numerical values within the question
            numbers_in_question = re.findall(r'([-+]?\d*\.?\d+)', question_text)
            if numbers_in_question:
                num_to_replace = random.choice(numbers_in_question)
                blanked_question = question_text.replace(num_to_replace, "________", 1)
                
                return {
                    "question": blanked_question,
                    "answer": num_to_replace,
                    "difficulty": difficulty
                }
            else:
                # If no numerical values found, create a generic fill-in-the-blank
                words = question_text.split()
                if len(words) > 5:
                    word_to_replace = words[len(words) // 2]
                    blanked_question = question_text.replace(word_to_replace, "________", 1)
                    
                    return {
                        "question": blanked_question,
                        "answer": word_to_replace,
                        "difficulty": difficulty
                    }
                else:
                    return {
                        "question": f"{question_text} ________",
                        "answer": "Cannot determine",
                        "difficulty": difficulty
                    }
    
    def generate_from_context(self, context: str, difficulty: str = "medium") -> Dict[str, Any]:
        """Generate a fill-in-the-blank question from a context"""
        # Extract meaningful content from the context
        sentences = re.split(r'[.!?]', context)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
        
        if not sentences:
            # Default when no good sentences found - purely mathematical definition
            return {
                "question": "In mathematics, a ________ is a statement that has been proven based on previously established statements.",
                "answer": "theorem",
                "difficulty": difficulty
            }
            
        # Select a sentence with good content
        potential_sentences = []
        for sentence in sentences:
            # Look for sentences with mathematical terms or numbers
            if re.search(r'([-+]?\d*\.?\d+)', sentence) or re.search(r'(formula|equation|theorem|function|value)', sentence, re.IGNORECASE):
                potential_sentences.append(sentence)
        
        if potential_sentences:
            selected_sentence = random.choice(potential_sentences)
        else:
            selected_sentence = random.choice(sentences)
        
        # Try to find numbers first
        numbers = re.findall(r'([-+]?\d*\.?\d+)', selected_sentence)
        math_terms = re.findall(r'\b(formula|equation|theorem|function|value|sum|product|integral|derivative)\b', selected_sentence, re.IGNORECASE)
        variables = re.findall(r'\b([a-zA-Z])\b', selected_sentence)
        
        # Decide what to blank out
        if numbers:
            # Replace a number with a blank
            number_to_replace = random.choice(numbers)
            blanked_sentence = selected_sentence.replace(number_to_replace, "________", 1)
            answer = number_to_replace
        elif math_terms:
            # Replace a mathematical term with a blank
            term_to_replace = random.choice(math_terms)
            blanked_sentence = re.sub(r'\b' + term_to_replace + r'\b', "________", selected_sentence, count=1, flags=re.IGNORECASE)
            answer = term_to_replace
        elif variables:
            var_to_replace = random.choice(variables)
            blanked_sentence = re.sub(r'\b' + var_to_replace + r'\b', "________", selected_sentence, count=1)
            answer = var_to_replace
        else:
            # If nothing else works, replace a random word
            words = selected_sentence.split()
            if len(words) < 5:
                return {
                    "question": "In mathematics, a ________ is a statement that has been proven based on previously established statements.",
                    "answer": "theorem",
                    "difficulty": difficulty
                }
                
            word_index = random.randint(len(words) // 3, (len(words) * 2) // 3)  # Choose word from middle third
            word_to_replace = words[word_index]
            words[word_index] = "________"
            blanked_sentence = " ".join(words)
            answer = word_to_replace
        
        return {
            "question": blanked_sentence,
            "answer": answer,
            "difficulty": difficulty
        }