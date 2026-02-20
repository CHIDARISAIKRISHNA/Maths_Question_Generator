import random
import re
from typing import Dict, Any, List

class MCQGenerator:
    """Generates multiple-choice questions"""
    
    def generate_from_problem(self, problem: Dict[str, str], difficulty: str = "medium") -> Dict[str, Any]:
        """Generate a multiple-choice question from a problem"""
        question_text = problem.get("problem", "")
        solution_text = problem.get("solution", "")
        category = problem.get("category", "Algebra")
        
        # Clean solution text - remove any "A:" prefix
        if "A:" in solution_text:
            solution_text = solution_text.replace("A:", "").strip()
        
        # Extract the final answer from the solution if possible
        correct_answer = ""
        
        # Try to find the last calculation result
        matches = re.findall(r'=\s*([-+]?\d*\.?\d+)', solution_text)
        if matches:
            correct_answer = matches[-1]
        else:
            # Try to find any number as a last resort
            matches = re.findall(r'([-+]?\d*\.?\d+)', solution_text)
            if matches:
                correct_answer = matches[-1]
            else:
                correct_answer = solution_text if solution_text else "Cannot determine"
        
        # Generate distractors (wrong options)
        try:
            correct_num = float(correct_answer)
            # Generate different types of distractors
            options = [
                {"text": str(correct_num)},
                {"text": str(round(correct_num + 1, 2))},
                {"text": str(round(correct_num - 1, 2))},
                {"text": str(round(correct_num * 2, 2))}
            ]
        except:
            # If we can't convert to a number, create generic options
            options = [
                {"text": correct_answer},
                {"text": "Cannot be determined"},
                {"text": "None of the above"},
                {"text": "All of the above"}
            ]
        
        # Shuffle the options to randomize the correct answer position
        random.shuffle(options)
        
        # Assign letters to options and find the correct answer letter
        letters = ["A", "B", "C", "D"]
        correct_letter = ""
        
        for i, option in enumerate(options):
            option["letter"] = letters[i]
            if option["text"] == str(correct_num) if 'correct_num' in locals() else correct_answer:
                correct_letter = letters[i]
        
        return {
            "question": f"{question_text}",
            "options": options,
            "correct_answer": correct_letter,
            "difficulty": difficulty
        }
    
    def generate_from_context(self, context: str, difficulty: str = "medium") -> Dict[str, Any]:
        """Generate a multiple-choice question from a context"""
        # Clean the context to handle special characters and formatting
        context = self._clean_context(context)
        
        # Extract mathematical content from the context
        equations = self._extract_equations(context)
        formulas = self._extract_formulas(context)
        numerical_values = self._extract_numerical_values(context)
        variables = self._extract_variables(context)
        mathematical_terms = self._extract_mathematical_terms(context)
        
        # Determine the mathematical topic from the context
        topic = self._determine_topic(context)
        
        # Generate multiple question types based on the content
        question_generators = []
        
        # Add question generators based on available content
        if equations:
            question_generators.extend([
                self._generate_equation_application_question,
                self._generate_equation_solving_question,
                self._generate_equation_property_question
            ])
        
        if formulas:
            question_generators.extend([
                self._generate_formula_application_question,
                self._generate_formula_identification_question
            ])
        
        if variables:
            question_generators.append(self._generate_variable_question)
        
        if mathematical_terms:
            question_generators.extend([
                self._generate_definition_question,
                self._generate_concept_application_question
            ])
        
        # If we have valid question types, choose one randomly
        if question_generators:
            question_generator = random.choice(question_generators)
            return question_generator(context, topic, equations, formulas, numerical_values, variables, mathematical_terms, difficulty)
        
        # Fallback to the default question generation
        return self._generate_default_mcq(context, difficulty)
    
    def _clean_context(self, context: str) -> str:
        """Clean the context to handle special characters and formatting"""
        # Replace unicode characters with their ASCII equivalents
        context = context.replace('−', '-')  # Replace unicode minus with hyphen
        context = context.replace('²', '^2')  # Replace superscript 2 with ^2
        context = context.replace('₁', '1')  # Replace subscript 1 with 1
        context = context.replace('₂', '2')  # Replace subscript 2 with 2
        
        # Remove any other non-ASCII characters
        context = ''.join(c for c in context if ord(c) < 128)
        
        return context
    
    def _determine_topic(self, context: str) -> str:
        """Determine the mathematical topic from the context"""
        # Check for keywords related to different mathematical topics
        algebra_keywords = ["equation", "polynomial", "factor", "exponent", "inequality", "function", "linear", "quadratic"]
        calculus_keywords = ["derivative", "integral", "limit", "continuity", "differentiation", "integration", "theorem of calculus"]
        probability_keywords = ["probability", "random", "event", "outcome", "permutation", "combination", "distribution"]
        geometry_keywords = ["triangle", "circle", "angle", "polygon", "coordinate", "distance", "plane"]
        
        # Count occurrences of keywords for each topic
        algebra_count = sum(1 for keyword in algebra_keywords if keyword.lower() in context.lower())
        calculus_count = sum(1 for keyword in calculus_keywords if keyword.lower() in context.lower())
        probability_count = sum(1 for keyword in probability_keywords if keyword.lower() in context.lower())
        geometry_count = sum(1 for keyword in geometry_keywords if keyword.lower() in context.lower())
        
        # Determine the topic with the most keyword matches
        counts = {
            "Algebra": algebra_count,
            "Calculus": calculus_count,
            "Probability": probability_count,
            "Geometry": geometry_count
        }
        
        # Return the topic with the highest count, or "General Mathematics" if all counts are 0
        max_topic = max(counts, key=counts.get)
        return max_topic if counts[max_topic] > 0 else "General Mathematics"
    
    def _extract_equations(self, context: str) -> List[str]:
        """Extract equations from the context"""
        # Look for equations with = sign
        basic_equations = re.findall(r'([^.!?]*=\s*[^.!?]*)', context)
        
        # Look for equations with mathematical notation
        math_equations = re.findall(r'([^.!?]*$$[^)]*$$\s*=\s*[^.!?]*)', context)
        
        # Look for named equations like "quadratic equation", "linear equation", etc.
        named_equations = re.findall(r'([^.!?]*(equation|formula|expression)[^.!?]*)', context, re.IGNORECASE)
        named_equations = [eq[0] for eq in named_equations]
        
        # Combine all equations and remove duplicates
        all_equations = basic_equations + math_equations + named_equations
        return list(set([eq.strip() for eq in all_equations if len(eq.strip()) > 5]))
    
    def _extract_formulas(self, context: str) -> List[str]:
        """Extract formulas from the context"""
        # Look for sentences containing formula keywords
        formula_keywords = ["formula", "theorem", "law", "rule", "identity"]
        formulas = []
        
        for keyword in formula_keywords:
            matches = re.findall(r'([^.!?]*' + keyword + r'[^.!?]*)', context, re.IGNORECASE)
            formulas.extend(matches)
        
        return list(set([f.strip() for f in formulas if len(f.strip()) > 10]))
    
    def _extract_numerical_values(self, context: str) -> List[str]:
        """Extract numerical values from the context"""
        # Find all numbers, including those with decimal points
        numbers = re.findall(r'([-+]?\d*\.?\d+)', context)
        
        # Find numbers with units (e.g., "5 meters", "20 m/s")
        numbers_with_units = re.findall(r'([-+]?\d*\.?\d+\s*[a-zA-Z°/%]+(?:/[a-zA-Z]+)?)', context)
        
        # Combine all numerical values and remove duplicates
        all_numbers = numbers + numbers_with_units
        return list(set(all_numbers))
    
    def _extract_variables(self, context: str) -> List[str]:
        """Extract variables from the context"""
        # Find single-letter variables commonly used in math
        single_vars = re.findall(r'\b([a-zA-Z])\b', context)
        
        # Find multi-letter variables or functions
        multi_vars = re.findall(r'\b([a-zA-Z]+$$[a-zA-Z]+$$)\b', context)
        
        # Combine all variables and remove duplicates
        all_vars = single_vars + multi_vars
        
        # Filter out common English words that might be mistaken for variables
        common_words = ["a", "i", "in", "is", "it", "of", "or", "to", "the", "and", "for"]
        filtered_vars = [v for v in all_vars if v.lower() not in common_words]
        
        return filtered_vars
    
    def _extract_mathematical_terms(self, context: str) -> List[str]:
        """Extract mathematical terms from the context"""
        # List of common mathematical terms to look for
        math_terms = [
            "function", "equation", "polynomial", "derivative", "integral", "limit", 
            "continuity", "probability", "permutation", "combination", "distribution",
            "triangle", "circle", "angle", "polygon", "coordinate", "distance",
            "exponent", "factor", "quadratic", "linear", "inequality", "absolute value",
            "system", "rational", "irrational", "real number", "complex number",
            "vector", "matrix", "determinant", "eigenvalue", "eigenvector",
            "sequence", "series", "convergence", "divergence", "theorem"
        ]
        
        # Find occurrences of mathematical terms
        found_terms = []
        for term in math_terms:
            if term.lower() in context.lower():
                # Find the complete phrase containing the term
                term_matches = re.findall(r'([^.!?]*\b' + term + r'\b[^.!?]*)', context, re.IGNORECASE)
                if term_matches:
                    found_terms.extend(term_matches)
        
        return list(set([t.strip() for t in found_terms if len(t.strip()) > 10]))
    
    def _generate_equation_application_question(self, context, topic, equations, formulas, numerical_values, variables, mathematical_terms, difficulty):
        """Generate a question about applying an equation"""
        if not equations:
            return self._generate_default_mcq(context, difficulty)
        
        equation = random.choice(equations)
        
        # Generate question templates based on the topic
        if topic == "Algebra":
            question_templates = [
                f"What is the solution to the equation {equation}?",
                f"If you solve {equation}, what value of x do you get?",
                f"Which of the following is the correct solution to {equation}?",
                f"When solving {equation}, what is the value of the variable?"
            ]
        elif topic == "Calculus":
            question_templates = [
                f"What is the derivative of {equation}?",
                f"If f(x) = {equation}, what is f'(x)?",
                f"What is the integral of {equation}?",
                f"If you differentiate {equation}, what do you get?"
            ]
        else:
            question_templates = [
                f"How would you apply {equation} to solve a problem?",
                f"What does {equation} represent in mathematical terms?",
                f"Which of the following is a correct application of {equation}?",
                f"In what scenario would you use {equation}?"
            ]
        
        question = random.choice(question_templates)
        
        # Generate options based on the equation
        if "=" in equation:
            parts = equation.split("=")
            left_side = parts[0].strip()
            right_side = parts[1].strip()
            
            # Try to create options based on the equation structure
            if "x" in left_side or "x" in right_side:
                options = [
                    "x = 2",
                    "x = -2",
                    "x = 0",
                    "x = 1"
                ]
            elif any(var in left_side or var in right_side for var in variables):
                var = next((v for v in variables if v in left_side or v in right_side), "y")
                options = [
                    f"{var} = 2",
                    f"{var} = -2",
                    f"{var} = 0",
                    f"{var} = 1"
                ]
            else:
                options = [
                    "The solution is 2",
                    "The solution is -2",
                    "The solution is 0",
                    "The solution is 1"
                ]
        else:
            # If no equals sign, create generic options
            options = [
                "The result is a linear equation",
                "The result is a quadratic equation",
                "The result is a constant",
                "The result is a variable expression"
            ]
        
        # Shuffle options to randomize the correct answer position
        correct_option = options[0]  # Save the correct answer
        random.shuffle(options)  # Shuffle the options
        
        # Create option objects and find the correct answer letter
        option_objects = []
        letters = ["A", "B", "C", "D"]
        correct_letter = ""
        
        for i, option in enumerate(options):
            option_objects.append({"text": option, "letter": letters[i]})
            if option == correct_option:
                correct_letter = letters[i]
        
        return {
            "question": question,
            "options": option_objects,
            "correct_answer": correct_letter,
            "difficulty": difficulty
        }
    
    def _generate_equation_solving_question(self, context, topic, equations, formulas, numerical_values, variables, mathematical_terms, difficulty):
        """Generate a question about solving an equation"""
        if not equations:
            return self._generate_default_mcq(context, difficulty)
        
        equation = random.choice(equations)
        
        # Generate question templates
        question_templates = [
            f"What method would be most efficient to solve {equation}?",
            f"Which technique should be applied first when solving {equation}?",
            f"What is the first step in solving {equation}?",
            f"To solve {equation}, which mathematical property would you apply?"
        ]
        
        question = random.choice(question_templates)
        
        # Generate options based on the equation type
        if "x^2" in equation or "x²" in equation or "quadratic" in equation.lower():
            options = [
                "Factoring the quadratic expression",
                "Using the quadratic formula",
                "Completing the square",
                "Graphing the function"
            ]
        elif "=" in equation and ("|" in equation or "absolute" in equation.lower()):
            options = [
                "Split into two separate equations",
                "Square both sides",
                "Apply the definition of absolute value",
                "Use a number line to analyze cases"
            ]
        elif "=" in equation and any(op in equation for op in ["+", "-", "*", "/"]):
            options = [
                "Isolate the variable by performing inverse operations",
                "Combine like terms",
                "Distribute terms",
                "Simplify expressions on both sides"
            ]
        else:
            options = [
                "Apply algebraic manipulation",
                "Use substitution",
                "Apply a specific formula",
                "Convert to a different form"
            ]
        
        # Shuffle options to randomize the correct answer position
        correct_option = options[0]  # Save the correct answer
        random.shuffle(options)  # Shuffle the options
        
        # Create option objects and find the correct answer letter
        option_objects = []
        letters = ["A", "B", "C", "D"]
        correct_letter = ""
        
        for i, option in enumerate(options):
            option_objects.append({"text": option, "letter": letters[i]})
            if option == correct_option:
                correct_letter = letters[i]
        
        return {
            "question": question,
            "options": option_objects,
            "correct_answer": correct_letter,
            "difficulty": difficulty
        }
    
    def _generate_equation_property_question(self, context, topic, equations, formulas, numerical_values, variables, mathematical_terms, difficulty):
        """Generate a question about properties of an equation"""
        if not equations:
            return self._generate_default_mcq(context, difficulty)
        
        equation = random.choice(equations)
        
        # Generate question templates
        question_templates = [
            f"What property does the equation {equation} exhibit?",
            f"Which of the following is true about {equation}?",
            f"What can be concluded about {equation}?",
            f"What mathematical property is demonstrated by {equation}?"
        ]
        
        question = random.choice(question_templates)
        
        # Generate options based on the equation type
        if "x^2" in equation or "x²" in equation or "quadratic" in equation.lower():
            options = [
                "It has at most two real solutions",
                "It represents a parabola when graphed",
                "Its degree is 2",
                "It can be written in the form ax² + bx + c = 0"
            ]
        elif "=" in equation and "x" in equation and not any(higher in equation for higher in ["x^2", "x²", "x^3"]):
            options = [
                "It represents a linear relationship",
                "It has exactly one solution",
                "It represents a straight line when graphed",
                "Its degree is 1"
            ]
        elif "=" in equation and any(trig in equation.lower() for trig in ["sin", "cos", "tan"]):
            options = [
                "It has infinitely many solutions",
                "It is periodic",
                "It involves trigonometric functions",
                "Its solutions repeat in intervals of 2π"
            ]
        else:
            options = [
                "It can be solved algebraically",
                "It represents a mathematical relationship",
                "It contains at least one variable",
                "It can be graphed on a coordinate plane"
            ]
        
        # Shuffle options to randomize the correct answer position
        correct_option = options[0]  # Save the correct answer
        random.shuffle(options)  # Shuffle the options
        
        # Create option objects and find the correct answer letter
        option_objects = []
        letters = ["A", "B", "C", "D"]
        correct_letter = ""
        
        for i, option in enumerate(options):
            option_objects.append({"text": option, "letter": letters[i]})
            if option == correct_option:
                correct_letter = letters[i]
        
        return {
            "question": question,
            "options": option_objects,
            "correct_answer": correct_letter,
            "difficulty": difficulty
        }
    
    def _generate_formula_application_question(self, context, topic, equations, formulas, numerical_values, variables, mathematical_terms, difficulty):
        """Generate a question about applying a formula"""
        if not formulas and not equations:
            return self._generate_default_mcq(context, difficulty)
        
        formula = random.choice(formulas) if formulas else random.choice(equations)
        
        # Extract a formula name if present
        formula_name = ""
        formula_name_match = re.search(r'\b([A-Z][a-z]+(?:\'s)?\s+(?:formula|theorem|law|rule|identity))\b', formula, re.IGNORECASE)
        if formula_name_match:
            formula_name = formula_name_match.group(1)
        
        # Generate question templates
        if formula_name:
            question_templates = [
                f"How is {formula_name} applied in mathematics?",
                f"What is the primary use of {formula_name}?",
                f"In which scenario would you apply {formula_name}?",
                f"What problem can be solved using {formula_name}?"
            ]
        else:
            question_templates = [
                f"How would you apply the formula {formula}?",
                f"What is the primary use of the formula {formula}?",
                f"In which scenario would you use {formula}?",
                f"What type of problem can be solved using {formula}?"
            ]
        
        question = random.choice(question_templates)
        
        # Generate options based on the topic
        if topic == "Algebra":
            options = [
                "Solving equations with variables",
                "Factoring polynomial expressions",
                "Simplifying algebraic fractions",
                "Finding the vertex of a parabola"
            ]
        elif topic == "Calculus":
            options = [
                "Finding the rate of change of a function",
                "Calculating the area under a curve",
                "Determining the limit of a function",
                "Analyzing the behavior of a function"
            ]
        elif topic == "Probability":
            options = [
                "Calculating the likelihood of an event",
                "Determining the number of possible outcomes",
                "Finding the expected value of a random variable",
                "Analyzing the distribution of outcomes"
            ]
        elif topic == "Geometry":
            options = [
                "Calculating the area of a shape",
                "Finding the distance between points",
                "Determining the angle between lines",
                "Analyzing the properties of geometric figures"
            ]
        else:
            options = [
                "Solving mathematical problems",
                "Analyzing relationships between variables",
                "Making predictions based on patterns",
                "Simplifying complex calculations"
            ]
        
        # Shuffle options to randomize the correct answer position
        correct_option = options[0]  # Save the correct answer
        random.shuffle(options)  # Shuffle the options
        
        # Create option objects and find the correct answer letter
        option_objects = []
        letters = ["A", "B", "C", "D"]
        correct_letter = ""
        
        for i, option in enumerate(options):
            option_objects.append({"text": option, "letter": letters[i]})
            if option == correct_option:
                correct_letter = letters[i]
        
        return {
            "question": question,
            "options": option_objects,
            "correct_answer": correct_letter,
            "difficulty": difficulty
        }
    
    def _generate_formula_identification_question(self, context, topic, equations, formulas, numerical_values, variables, mathematical_terms, difficulty):
        """Generate a question about identifying a formula"""
        if not formulas and not equations:
            return self._generate_default_mcq(context, difficulty)
        
        formula = random.choice(formulas) if formulas else random.choice(equations)
        
        # Generate question templates
        question_templates = [
            f"Which formula is represented by {formula}?",
            f"What is the name of the formula {formula}?",
            f"In mathematics, {formula} is known as:",
            f"The expression {formula} is commonly referred to as:"
        ]
        
        question = random.choice(question_templates)
        
        # Generate options based on the topic and formula content
        if "a^2 + b^2 = c^2" in formula or "pythagorean" in formula.lower():
            options = [
                "The Pythagorean Theorem",
                "The Law of Cosines",
                "The Distance Formula",
                "The Quadratic Formula"
            ]
        elif "(-b ± √(b² - 4ac))/(2a)" in formula or "quadratic formula" in formula.lower():
            options = [
                "The Quadratic Formula",
                "The Discriminant Formula",
                "The Completing the Square Method",
                "The Polynomial Root Formula"
            ]
        elif "d/dx" in formula or "derivative" in formula.lower():
            options = [
                "The Power Rule of Differentiation",
                "The Chain Rule",
                "The Product Rule",
                "The Quotient Rule"
            ]
        elif "∫" in formula or "integral" in formula.lower():
            options = [
                "The Fundamental Theorem of Calculus",
                "The Integration by Parts Formula",
                "The Substitution Method",
                "The Partial Fractions Method"
            ]
        elif "P(A)" in formula or "probability" in formula.lower():
            options = [
                "The Probability Formula",
                "Bayes' Theorem",
                "The Law of Total Probability",
                "The Conditional Probability Formula"
            ]
        else:
            # Generate generic formula names based on the topic
            if topic == "Algebra":
                options = [
                    "The Binomial Theorem",
                    "The Factor Theorem",
                    "The Remainder Theorem",
                    "The Rational Root Theorem"
                ]
            elif topic == "Calculus":
                options = [
                    "The Mean Value Theorem",
                    "L'Hôpital's Rule",
                    "The Taylor Series Formula",
                    "The Intermediate Value Theorem"
                ]
            elif topic == "Probability":
                options = [
                    "The Binomial Probability Formula",
                    "The Expected Value Formula",
                    "The Variance Formula",
                    "The Normal Distribution Formula"
                ]
            else:
                options = [
                    "A Mathematical Identity",
                    "A Fundamental Theorem",
                    "A Mathematical Law",
                    "A Mathematical Rule"
                ]
        
        # Shuffle options to randomize the correct answer position
        correct_option = options[0]  # Save the correct answer
        random.shuffle(options)  # Shuffle the options
        
        # Create option objects and find the correct answer letter
        option_objects = []
        letters = ["A", "B", "C", "D"]
        correct_letter = ""
        
        for i, option in enumerate(options):
            option_objects.append({"text": option, "letter": letters[i]})
            if option == correct_option:
                correct_letter = letters[i]
        
        return {
            "question": question,
            "options": option_objects,
            "correct_answer": correct_letter,
            "difficulty": difficulty
        }
    
    def _generate_calculation_question(self, context, topic, equations, formulas, numerical_values, variables, mathematical_terms, difficulty):
        """Generate a calculation question using numerical values"""
        if len(numerical_values) < 2:
            return self._generate_default_mcq(context, difficulty)
        
        # Choose two numerical values
        try:
            # Try to find clean numerical values (just numbers)
            clean_values = []
            for val in numerical_values:
                match = re.search(r'([-+]?\d*\.?\d+)', val)
                if match:
                    clean_values.append(match.group(1))
            
            if len(clean_values) >= 2:
                num1, num2 = random.sample(clean_values, 2)
            else:
                # If we don't have enough clean values, use the first one and derive another
                num1 = clean_values[0] if clean_values else "5"
                num2 = str(int(float(num1)) + 2)  # Simple derivation
            
            # Convert to float for calculations
            float_num1 = float(num1)
            float_num2 = float(num2)
            
            # Generate a mathematical operation question
            operations = [
                (float_num1 + float_num2, f"Calculate {num1} + {num2}.", "+"),
                (float_num1 - float_num2, f"Calculate {num1} - {num2}.", "-"),
                (float_num1 * float_num2, f"Calculate {num1} × {num2}.", "×"),
                (float_num1 / float_num2 if float_num2 != 0 else float_num1 * 2, f"Calculate {num1} ÷ {num2}.", "÷")
            ]
            
            # Choose a random operation
            result, question, operation_symbol = random.choice(operations)
            
            # Generate options
            correct_answer = str(round(result, 2))
            options = [
                correct_answer,
                str(round(result * 0.8, 2)),
                str(round(result * 1.2, 2)),
                str(round(result * 0.5, 2))
            ]
            
            # Shuffle options to randomize the correct answer position
            correct_option = options[0]  # Save the correct answer
            random.shuffle(options)  # Shuffle the options
            
            # Create option objects and find the correct answer letter
            option_objects = []
            letters = ["A", "B", "C", "D"]
            correct_letter = ""
            
            for i, option in enumerate(options):
                option_objects.append({"text": option, "letter": letters[i]})
                if option == correct_option:
                    correct_letter = letters[i]
            
            return {
                "question": question,
                "options": option_objects,
                "correct_answer": correct_letter,
                "difficulty": difficulty
            }
        except Exception as e:
            # Fallback if any error occurs
            return self._generate_default_mcq(context, difficulty)
    
    def _generate_variable_question(self, context, topic, equations, formulas, numerical_values, variables, mathematical_terms, difficulty):
        """Generate a question about variables in the context"""
        if not variables:
            return self._generate_default_mcq(context, difficulty)
        
        # Filter out common English words that might be mistaken for variables
        common_words = ["a", "i", "in", "is", "it", "of", "or", "to", "the", "and", "for"]
        filtered_vars = [v for v in variables if v.lower() not in common_words and len(v) == 1]
        
        if not filtered_vars:
            return self._generate_default_mcq(context, difficulty)
        
        variable = random.choice(filtered_vars)
        
        # Generate question templates
        question_templates = [
            f"What does the variable {variable} represent in this mathematical context?",
            f"In the given context, what is the meaning of {variable}?",
            f"What mathematical quantity does {variable} symbolize?",
            f"What role does the variable {variable} play in this mathematical scenario?"
        ]
        
        question = random.choice(question_templates)
        
        # Generate options based on the variable and topic
        if topic == "Algebra":
            if variable.lower() == 'x':
                options = [
                    "The independent variable in the equation",
                    "A constant value",
                    "The coefficient of a term",
                    "The exponent of a term"
                ]
            elif variable.lower() == 'y':
                options = [
                    "The dependent variable in the equation",
                    "A parameter",
                    "A constant term",
                    "The slope of a line"
                ]
            else:
                options = [
                    "A variable in the equation",
                    "A coefficient",
                    "A constant",
                    "An exponent"
                ]
        elif topic == "Calculus":
            if variable.lower() == 'x':
                options = [
                    "The independent variable being differentiated",
                    "The upper limit of integration",
                    "The lower limit of integration",
                    "The constant of integration"
                ]
            elif variable.lower() == 'y':
                options = [
                    "The dependent variable or function",
                    "The derivative",
                    "The integral",
                    "The limit"
                ]
            else:
                options = [
                    "A variable in the function",
                    "A parameter",
                    "A constant",
                    "A limit"
                ]
        elif topic == "Probability":
            options = [
                "A random variable",
                "An event",
                "A probability value",
                "A sample space element"
            ]
        else:
            options = [
                "A variable in the mathematical expression",
                "A parameter that can be adjusted",
                "A constant value",
                "A coefficient"
            ]
        
        # Shuffle options to randomize the correct answer position
        correct_option = options[0]  # Save the correct answer
        random.shuffle(options)  # Shuffle the options
        
        # Create option objects and find the correct answer letter
        option_objects = []
        letters = ["A", "B", "C", "D"]
        correct_letter = ""
        
        for i, option in enumerate(options):
            option_objects.append({"text": option, "letter": letters[i]})
            if option == correct_option:
                correct_letter = letters[i]
        
        return {
            "question": question,
            "options": option_objects,
            "correct_answer": correct_letter,
            "difficulty": difficulty
        }
    
    def _generate_definition_question(self, context, topic, equations, formulas, numerical_values, variables, mathematical_terms, difficulty):
        """Generate a question about defining a mathematical term"""
        if not mathematical_terms:
            return self._generate_default_mcq(context, difficulty)
        
        term_sentence = random.choice(mathematical_terms)
        
        # Extract the key mathematical term from the sentence
        math_term_keywords = [
            "function", "equation", "polynomial", "derivative", "integral", "limit", 
            "continuity", "probability", "permutation", "combination", "distribution",
            "triangle", "circle", "angle", "polygon", "coordinate", "distance",
            "exponent", "factor", "quadratic", "linear", "inequality", "absolute value",
            "system", "rational", "irrational", "real number", "complex number",
            "vector", "matrix", "determinant", "eigenvalue", "eigenvector",
            "sequence", "series", "convergence", "divergence", "theorem"
        ]
        
        # Find the first occurrence of a math term in the sentence
        term = next((keyword for keyword in math_term_keywords if keyword.lower() in term_sentence.lower()), "term")
        
        # Generate question templates
        question_templates = [
            f"What is the definition of a {term}?",
            f"Which of the following best defines a {term}?",
            f"In mathematics, what is a {term}?",
            f"How would you define a {term} in mathematical terms?"
        ]
        
        question = random.choice(question_templates)
        
        # Generate options based on the term
        if term == "function":
            options = [
                "A relation that assigns exactly one output to each input",
                "An equation with variables on both sides",
                "A mathematical operation that combines two values",
                "A set of ordered pairs"
            ]
        elif term == "equation":
            options = [
                "A mathematical statement that asserts the equality of two expressions",
                "A formula used to calculate a specific value",
                "A relationship between variables",
                "A mathematical expression containing variables"
            ]
        elif term == "derivative":
            options = [
                "The rate of change of a function with respect to a variable",
                "The area under a curve",
                "The limit of a function as x approaches infinity",
                "The inverse of a function"
            ]
        elif term == "integral":
            options = [
                "The accumulation of quantities represented as the area under a curve",
                "The slope of a tangent line to a curve",
                "The rate of change of a function",
                "The limit of a function at a point"
            ]
        elif term == "probability":
            options = [
                "A measure of the likelihood of an event occurring",
                "The number of possible outcomes in an experiment",
                "The ratio of favorable outcomes to total outcomes",
                "A statistical measure of central tendency"
            ]
        else:
            # Generate generic options for other terms
            options = [
                f"A mathematical concept related to {term}s",
                f"A formula used to calculate {term}s",
                f"A property of mathematical {term}s",
                f"A relationship involving {term}s"
            ]
        
        # Shuffle options to randomize the correct answer position
        correct_option = options[0]  # Save the correct answer
        random.shuffle(options)  # Shuffle the options
        
        # Create option objects and find the correct answer letter
        option_objects = []
        letters = ["A", "B", "C", "D"]
        correct_letter = ""
        
        for i, option in enumerate(options):
            option_objects.append({"text": option, "letter": letters[i]})
            if option == correct_option:
                correct_letter = letters[i]
        
        return {
            "question": question,
            "options": option_objects,
            "correct_answer": correct_letter,
            "difficulty": difficulty
        }
    
    def _generate_concept_application_question(self, context, topic, equations, formulas, numerical_values, variables, mathematical_terms, difficulty):
        """Generate a question about applying a mathematical concept"""
        if not mathematical_terms:
            return self._generate_default_mcq(context, difficulty)
        
        term_sentence = random.choice(mathematical_terms)
        
        # Extract the key mathematical term from the sentence
        math_term_keywords = [
            "function", "equation", "polynomial", "derivative", "integral", "limit", 
            "continuity", "probability", "permutation", "combination", "distribution",
            "triangle", "circle", "angle", "polygon", "coordinate", "distance",
            "exponent", "factor", "quadratic", "linear", "inequality", "absolute value",
            "system", "rational", "irrational", "real number", "complex number",
            "vector", "matrix", "determinant", "eigenvalue", "eigenvector",
            "sequence", "series", "convergence", "divergence", "theorem"
        ]
        
        # Find the first occurrence of a math term in the sentence
        term = next((keyword for keyword in math_term_keywords if keyword.lower() in term_sentence.lower()), "concept")
        
        # Generate question templates
        question_templates = [
            f"Which of the following is an application of {term}s in real life?",
            f"How are {term}s applied in practical situations?",
            f"In what scenario would you use {term}s?",
            f"What is a practical application of {term}s?"
        ]
        
        question = random.choice(question_templates)
        
        # Generate options based on the term
        if term == "function":
            options = [
                "Modeling the relationship between time and distance in physics",
                "Calculating the area of a rectangle",
                "Finding the sum of a series",
                "Determining the probability of an event"
            ]
        elif term == "derivative":
            options = [
                "Finding the velocity of an object given its position function",
                "Calculating the area under a curve",
                "Determining the volume of a solid",
                "Finding the sum of an infinite series"
            ]
        elif term == "integral":
            options = [
                "Calculating the total distance traveled from a velocity function",
                "Finding the slope of a tangent line",
                "Determining the maximum value of a function",
                "Solving a differential equation"
            ]
        elif term == "probability":
            options = [
                "Predicting the likelihood of rain tomorrow",
                "Calculating the area of a circle",
                "Finding the derivative of a function",
                "Solving a system of linear equations"
            ]
        elif term == "equation":
            options = [
                "Determining when two cars traveling at different speeds will meet",
                "Finding the area under a curve",
                "Calculating the probability of an event",
                "Determining the limit of a sequence"
            ]
        else:
            # Generate generic options for other terms
            options = [
                f"Applying {term}s to solve real-world problems",
                f"Using {term}s in mathematical calculations",
                f"Implementing {term}s in computational algorithms",
                f"Utilizing {term}s in theoretical mathematics"
            ]
        
        # Shuffle options to randomize the correct answer position
        correct_option = options[0]  # Save the correct answer
        random.shuffle(options)  # Shuffle the options
        
        # Create option objects and find the correct answer letter
        option_objects = []
        letters = ["A", "B", "C", "D"]
        correct_letter = ""
        
        for i, option in enumerate(options):
            option_objects.append({"text": option, "letter": letters[i]})
            if option == correct_option:
                correct_letter = letters[i]
        
        return {
            "question": question,
            "options": option_objects,
            "correct_answer": correct_letter,
            "difficulty": difficulty
        }
    
    def _generate_default_mcq(self, context, difficulty):
        """Generate a default MCQ based on the context"""
        # Try to extract a sentence from the context
        sentences = re.split(r'[.!?]', context)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
        
        if sentences:
            selected_sentence = random.choice(sentences)
            
            # Generate a question based on the selected sentence
            question = f"Which of the following statements about {selected_sentence[:30]}... is correct?"
            
            # Generate options
            options = [
                selected_sentence,
                "This statement is incorrect",
                "This concept applies only in specific cases",
                "This is a common misconception"
            ]
            
            # Shuffle options to randomize the correct answer position
            correct_option = options[0]  # Save the correct answer
            random.shuffle(options)  # Shuffle the options
            
            # Create option objects and find the correct answer letter
            option_objects = []
            letters = ["A", "B", "C", "D"]
            correct_letter = ""
            
            for i, option in enumerate(options):
                option_objects.append({"text": option, "letter": letters[i]})
                if option == correct_option:
                    correct_letter = letters[i]
            
            return {
                "question": question,
                "options": option_objects,
                "correct_answer": correct_letter,
                "difficulty": difficulty
            }
        else:
            # If no good sentences found, use a topic-based question
            return self._generate_topic_based_mcq(context, difficulty)
    
    def _generate_topic_based_mcq(self, context, difficulty):
        """Generate a topic-based MCQ when no specific content can be extracted"""
        # Determine the topic from the context
        topic = self._determine_topic(context)
        
        # Generate a question based on the topic
        if topic == "Algebra":
            question = "Which of the following is a fundamental concept in algebra?"
            options = [
                "The manipulation of variables and constants using arithmetic operations",
                "The calculation of limits as x approaches infinity",
                "The measurement of angles in a triangle",
                "The computation of probabilities in random experiments"
            ]
        elif topic == "Calculus":
            question = "Which of the following best describes a key concept in calculus?"
            options = [
                "The study of rates of change and accumulation",
                "The solution of systems of linear equations",
                "The properties of geometric shapes",
                "The analysis of statistical data"
            ]
        elif topic == "Probability":
            question = "Which statement correctly describes a fundamental principle of probability?"
            options = [
                "The probability of any event is a number between 0 and 1",
                "The derivative of a function gives its rate of change",
                "Similar triangles have proportional sides",
                "Parallel lines never intersect"
            ]
        elif topic == "Geometry":
            question = "Which of the following is a core principle in geometry?"
            options = [
                "The study of properties and relationships of points, lines, surfaces, and solids",
                "The analysis of rates of change",
                "The manipulation of algebraic expressions",
                "The calculation of probabilities"
            ]
        else:
            question = "Which of the following is a fundamental mathematical concept?"
            options = [
                "Mathematics provides a language for describing patterns and relationships",
                "All mathematical problems have exactly one solution",
                "Mathematics is only applicable to theoretical scenarios",
                "Mathematical principles change frequently over time"
            ]
        
        # Shuffle options to randomize the correct answer position
        correct_option = options[0]  # Save the correct answer
        random.shuffle(options)  # Shuffle the options
        
        # Create option objects and find the correct answer letter
        option_objects = []
        letters = ["A", "B", "C", "D"]
        correct_letter = ""
        
        for i, option in enumerate(options):
            option_objects.append({"text": option, "letter": letters[i]})
            if option == correct_option:
                correct_letter = letters[i]
        
        return {
            "question": question,
            "options": option_objects,
            "correct_answer": correct_letter,
            "difficulty": difficulty
        }