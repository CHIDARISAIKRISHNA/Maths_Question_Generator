import re
import random
from typing import Dict, Any

class ShortAnswerGenerator:
    """Generates short answer questions"""
    
    def generate_from_problem(self, problem: Dict[str, str], difficulty: str = "medium") -> Dict[str, Any]:
        """Generate a short answer question from a problem"""
        question_text = problem.get("problem", "")
        solution_text = problem.get("solution", "")
        category = problem.get("category", "Algebra")
        
        # Extract the final answer from the solution if possible
        answer = ""
        
        # Clean solution text - remove any "A:" prefix before extracting
        if "A:" in solution_text:
            clean_solution = solution_text.replace("A:", "").strip()
        else:
            clean_solution = solution_text
        
        # Try to find numerical answers
        numerical_matches = re.findall(r'([-+]?\d*\.?\d+)', clean_solution)
        if numerical_matches:
            answer = numerical_matches[-1]  # Use the last number as the answer
        
        # If no answer found, use the whole solution
        if not answer:
            answer = clean_solution if clean_solution else "Cannot determine"
        
        return {
            "question": f"{question_text}",
            "answer": answer,
            "difficulty": difficulty
        }
    
    def generate_from_context(self, context: str, difficulty: str = "medium") -> Dict[str, Any]:
        """Generate a short answer question from a context"""
        # Extract meaningful information from the context
        sentences = re.split(r'[.!?]', context)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
        
        if not sentences:
            # Default when no good sentences found - keep it purely mathematical
            return {
                "question": "In mathematics, what is a theorem?",
                "answer": "A theorem is a mathematical statement that has been proven to be true based on axioms and previously established theorems.",
                "difficulty": difficulty
            }
        
        # Look for specific mathematical concepts
        math_concepts = []
        
        # Look for equations (contains = sign)
        equations = [s for s in sentences if "=" in s]
        if equations:
            for eq in equations:
                # Provide a mathematical explanation rather than repeating the equation itself
                equation_explanations = [
                    "It describes a specific relationship between the variables in the equation.",
                    "It gives a rule that relates the quantities appearing in the equation.",
                    "It represents a mathematical condition that the variables must satisfy.",
                    "It expresses how the unknowns in the equation depend on the given constants."
                ]
                math_concepts.append({
                    "type": "equation",
                    "text": eq,
                    "question": f"In the context, what does the equation '{eq}' describe?",
                    "answer": random.choice(equation_explanations)
                })
        
        # Look for definitions
        definitions = [s for s in sentences if any(x in s.lower() for x in ["is defined as", "refers to", "is a", "is called", "is known as"])]
        if definitions:
            for defn in definitions:
                # Try to extract what's being defined
                match = re.search(r'([^,]+)(?:is defined as|refers to|is a|is called|is known as)', defn, re.IGNORECASE)
                if match:
                    term = match.group(1).strip()
                    math_concepts.append({
                        "type": "definition",
                        "text": defn,
                        "question": f"Define the term '{term}' as mentioned in the context.",
                        "answer": defn
                    })
                else:
                    math_concepts.append({
                        "type": "definition",
                        "text": defn,
                        "question": "Explain the definition provided in the context.",
                        "answer": defn
                    })
        
        # Look for theorems, formulas, etc.
        for keyword in ["theorem", "formula", "rule", "law", "principle"]:
            matches = [s for s in sentences if keyword in s.lower()]
            for match in matches:
                math_concepts.append({
                    "type": "concept",
                    "text": match,
                    "question": f"Explain the {keyword} mentioned in the context.",
                    "answer": match
                })
        
        # If we found specific concepts, use one of them
        if math_concepts:
            concept = random.choice(math_concepts)
            return {
                "question": concept["question"],
                "answer": concept["answer"],
                "difficulty": difficulty
            }
        
        # If no specific concepts found, create a question from a random sentence
        selected_sentence = random.choice(sentences)
        
        # Try to identify key mathematical terms
        math_terms = ["function", "variable", "equation", "expression", "value", 
                      "formula", "theorem", "proof", "calculation", "solution"]
        
        found_terms = []
        for term in math_terms:
            if term in selected_sentence.lower():
                found_terms.append(term)
        
        if found_terms:
            term = random.choice(found_terms)
            return {
                "question": f"Explain the concept of '{term}' as it appears in the context.",
                "answer": selected_sentence,
                "difficulty": difficulty
            }
        else:
            # Create a generic but still mathematics-focused question
            return {
                "question": "Explain the main mathematical idea described in the context.",
                "answer": selected_sentence,
                "difficulty": difficulty
            }