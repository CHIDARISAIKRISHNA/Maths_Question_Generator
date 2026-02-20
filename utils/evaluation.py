import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from nltk.tokenize import word_tokenize
from rouge import Rouge
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from typing import List, Dict, Any, Union

# Download necessary NLTK resources
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

class QuestionEvaluator:
    """
    Evaluates the quality of generated questions using various metrics.
    """
    
    def __init__(self):
        """Initialize the evaluator."""
        self.rouge = Rouge()
        self.smoothing = SmoothingFunction().method1
    
    def calculate_bleu(self, reference: str, candidate: str) -> float:
        """
        Calculate BLEU score between reference and candidate strings.
        
        Args:
            reference: Reference text
            candidate: Candidate text to evaluate
            
        Returns:
            BLEU score
        """
        reference_tokens = word_tokenize(reference.lower())
        candidate_tokens = word_tokenize(candidate.lower())
        
        # BLEU requires a list of references
        references = [reference_tokens]
        
        # Calculate BLEU score with smoothing
        return sentence_bleu(references, candidate_tokens, smoothing_function=self.smoothing)
    
    def calculate_meteor(self, references, candidate):
        tokenized_references = [word_tokenize(ref.lower()) for ref in references]
        tokenized_candidate = word_tokenize(candidate.lower())  # Tokenize the candidate string

        return meteor_score(tokenized_references, tokenized_candidate)
    
    def calculate_rouge(self, reference: str, candidate: str) -> Dict[str, Dict[str, float]]:
        """
        Calculate ROUGE scores between reference and candidate strings.
        
        Args:
            reference: Reference text
            candidate: Candidate text to evaluate
            
        Returns:
            Dictionary of ROUGE scores
        """
        try:
            scores = self.rouge.get_scores(candidate, reference)[0]
            return scores
        except Exception as e:
            print(f"Error calculating ROUGE: {e}")
            return {
                'rouge-1': {'p': 0, 'r': 0, 'f': 0},
                'rouge-2': {'p': 0, 'r': 0, 'f': 0},
                'rouge-l': {'p': 0, 'r': 0, 'f': 0}
            }
    
    def calculate_f1(self, reference: str, candidate: str) -> float:
        """
        Calculate F1 score between reference and candidate strings.
        
        Args:
            reference: Reference text
            candidate: Candidate text to evaluate
            
        Returns:
            F1 score
        """
        reference_tokens = set(word_tokenize(reference.lower()))
        candidate_tokens = set(word_tokenize(candidate.lower()))
        
        # Calculate precision, recall, and F1
        common_tokens = reference_tokens.intersection(candidate_tokens)
        
        if not candidate_tokens:
            precision = 0
        else:
            precision = len(common_tokens) / len(candidate_tokens)
        
        if not reference_tokens:
            recall = 0
        else:
            recall = len(common_tokens) / len(reference_tokens)
        
        if precision + recall == 0:
            f1 = 0
        else:
            f1 = 2 * precision * recall / (precision + recall)
        
        return f1
    
    def evaluate_questions(self, references: List[str], candidates: List[str]) -> Dict[str, Union[float, Dict[str, float]]]:
        """
        Evaluate a list of candidate questions against reference questions.
        
        Args:
            references: List of reference questions
            candidates: List of candidate questions to evaluate
            
        Returns:
            Dictionary of average evaluation metrics
        """
        if len(references) != len(candidates):
            raise ValueError("Number of references and candidates must be the same")
        
        # Initialize metrics
        total_bleu = 0
        total_meteor = 0
        total_f1 = 0
        total_rouge1_p = 0
        total_rouge1_r = 0
        total_rouge1_f = 0
        total_rouge2_p = 0
        total_rouge2_r = 0
        total_rouge2_f = 0
        total_rougel_p = 0
        total_rougel_r = 0
        total_rougel_f = 0
        
        # Calculate metrics for each pair
        for ref, cand in zip(references, candidates):
            # Calculate BLEU
            bleu = self.calculate_bleu(ref, cand)
            total_bleu += bleu
            
            # Calculate METEOR
            meteor = self.calculate_meteor(ref, cand)
            total_meteor += meteor
            
            # Calculate F1
            f1 = self.calculate_f1(ref, cand)
            total_f1 += f1
            
            # Calculate ROUGE
            rouge = self.calculate_rouge(ref, cand)
            total_rouge1_p += rouge['rouge-1']['p']
            total_rouge1_r += rouge['rouge-1']['r']
            total_rouge1_f += rouge['rouge-1']['f']
            total_rouge2_p += rouge['rouge-2']['p']
            total_rouge2_r += rouge['rouge-2']['r']
            total_rouge2_f += rouge['rouge-2']['f']
            total_rougel_p += rouge['rouge-l']['p']
            total_rougel_r += rouge['rouge-l']['r']
            total_rougel_f += rouge['rouge-l']['f']
        
        # Calculate averages
        n = len(references)
        avg_bleu = total_bleu / n
        avg_meteor = total_meteor / n
        avg_f1 = total_f1 / n
        
        avg_rouge1_p = total_rouge1_p / n
        avg_rouge1_r = total_rouge1_r / n
        avg_rouge1_f = total_rouge1_f / n
        avg_rouge2_p = total_rouge2_p / n
        avg_rouge2_r = total_rouge2_r / n
        avg_rouge2_f = total_rouge2_f / n
        avg_rougel_p = total_rougel_p / n
        avg_rougel_r = total_rougel_r / n
        avg_rougel_f = total_rougel_f / n
        
        # Return average metrics
        return {
            'bleu': avg_bleu,
            'meteor': avg_meteor,
            'f1': avg_f1,
            'rouge-1': {
                'precision': avg_rouge1_p,
                'recall': avg_rouge1_r,
                'f1': avg_rouge1_f
            },
            'rouge-2': {
                'precision': avg_rouge2_p,
                'recall': avg_rouge2_r,
                'f1': avg_rouge2_f
            },
            'rouge-l': {
                'precision': avg_rougel_p,
                'recall': avg_rougel_r,
                'f1': avg_rougel_f
            }
        }
    
    def generate_latex_report(self, model_results: Dict[str, Dict[str, Union[float, Dict[str, float]]]], output_file: str = "evaluation_report.tex") -> None:
        """
        Generate a LaTeX report of the evaluation results.
        
        Args:
            model_results: Dictionary of model results
            output_file: Path to save the LaTeX report
        """
        # Start the LaTeX document
        latex_doc = """\\documentclass{article}
\\usepackage{graphicx}
\\usepackage{booktabs}
\\usepackage{amsmath}
\\usepackage{hyperref}
\\usepackage{xcolor}
\\usepackage{multirow}

\\title{Evaluation of Math Question Generation Models Using BLEU Score}
\\author{Math Question Generator Team}
\\date{\\today}

\\begin{document}

\\maketitle

\\section{Introduction}

In this report, we present the evaluation of our Math Question Generator model using various metrics, with a focus on the BLEU (Bilingual Evaluation Understudy) score. The evaluation compares the performance of our model against reference questions from standard datasets.

\\section{Evaluation Metrics}

\\subsection{BLEU Score}

BLEU (Bilingual Evaluation Understudy) is an algorithm for evaluating the quality of text that has been machine-translated from one natural language to another. In our context, we use BLEU to measure how similar the generated questions are to reference questions.

BLEU works by comparing n-grams of the candidate text with n-grams of the reference text and counting the number of matches. The more matches, the better the translation. The score ranges from 0 to 1, with 1 being a perfect match.

The formula for calculating BLEU score is:

\\begin{equation}
\\text{BLEU} = \\text{BP} \\cdot \\exp\\left(\\sum_{n=1}^{N} w_n \\log p_n\\right)
\\end{equation}

where:
\\begin{itemize}
    \\item $\\text{BP}$ is the brevity penalty, which penalizes short translations
    \\item $w_n$ are weights for each n-gram (typically uniform)
    \\item $p_n$ is the precision for n-grams of size n
    \\item $N$ is the maximum n-gram size (typically 4)
\\end{itemize}

\\subsection{ROUGE Score}

ROUGE (Recall-Oriented Understudy for Gisting Evaluation) is a set of metrics used for evaluating automatic summarization and machine translation. We use ROUGE-1, ROUGE-2, and ROUGE-L to evaluate our question generation model.

\\subsection{F1 Score}

The F1 score is the harmonic mean of precision and recall. It provides a balance between the two metrics and is particularly useful when the class distribution is uneven.

\\subsection{METEOR Score}

METEOR (Metric for Evaluation of Translation with Explicit ORdering) is a metric for machine translation evaluation that is based on the harmonic mean of unigram precision and recall, with recall weighted higher than precision. It also considers word order and synonyms, making it more comprehensive than BLEU.

\\section{Experimental Setup}

We evaluated our Math Question Generator using a dataset of mathematical problems and solutions. The evaluation was performed by comparing the generated questions against reference questions from the dataset.

\\section{Results}

Table \\ref{tab:evaluation_results} shows the evaluation metrics for our model.

\\begin{table}[h]
\\centering
\\caption{Evaluation metrics for our Math Question Generator model}
\\label{tab:evaluation_results}
\\begin{tabular}{lc}
\\toprule
\\textbf{Metric} & \\textbf{Score} \\\\
\\midrule
"""
        
        # Add BLEU Score
        model_name = list(model_results.keys())[0]
        metrics = model_results[model_name]
        
        latex_doc += f"BLEU Score & {metrics['bleu']:.4f} \\\\\n"
        
        # Add ROUGE Scores
        latex_doc += "\\midrule\n"
        latex_doc += "\\multirow{3}{*}{ROUGE-1} & "
        latex_doc += f"Precision: {metrics['rouge-1']['precision']:.4f} \\\\\n"
        latex_doc += f" & Recall: {metrics['rouge-1']['recall']:.4f} \\\\\n"
        latex_doc += f" & F1: {metrics['rouge-1']['f1']:.4f} \\\\\n"
        
        latex_doc += "\\midrule\n"
        latex_doc += "\\multirow{3}{*}{ROUGE-2} & "
        latex_doc += f"Precision: {metrics['rouge-2']['precision']:.4f} \\\\\n"
        latex_doc += f" & Recall: {metrics['rouge-2']['recall']:.4f} \\\\\n"
        latex_doc += f" & F1: {metrics['rouge-2']['f1']:.4f} \\\\\n"
        
        latex_doc += "\\midrule\n"
        latex_doc += "\\multirow{3}{*}{ROUGE-L} & "
        latex_doc += f"Precision: {metrics['rouge-l']['precision']:.4f} \\\\\n"
        latex_doc += f" & Recall: {metrics['rouge-l']['recall']:.4f} \\\\\n"
        latex_doc += f" & F1: {metrics['rouge-l']['f1']:.4f} \\\\\n"
        
        # Add F1 Score
        latex_doc += "\\midrule\n"
        latex_doc += f"F1 Score & {metrics['f1']:.4f} \\\\\n"
        
        # Add METEOR Score
        latex_doc += f"METEOR Score & {metrics['meteor']:.4f} \\\\\n"
        
        # End the table
        latex_doc += """\\bottomrule
\\end{tabular}
\\end{table}

\\section{Analysis}

Our Math Question Generator model achieves the following performance:

\\begin{itemize}
    \\item \\textbf{BLEU Score}: The model achieves a BLEU score of """
        
        latex_doc += f"{metrics['bleu']:.4f}, which indicates "
        
        if metrics['bleu'] < 0.2:
            latex_doc += "a relatively low similarity to reference questions. This suggests that our model generates questions that are quite different from the reference questions, which could be due to creative generation or a need for improvement."
        elif metrics['bleu'] < 0.4:
            latex_doc += "a moderate similarity to reference questions. This suggests that our model captures some aspects of the reference questions but still has room for improvement."
        else:
            latex_doc += "a good similarity to reference questions. This indicates that our model generates questions that are quite similar to the reference questions."
        
        latex_doc += """
    
    \\item \\textbf{ROUGE Scores}: The ROUGE scores provide insights into the recall and precision of our generated questions compared to reference questions. """
        
        latex_doc += f"The ROUGE-1 F1 score of {metrics['rouge-1']['f1']:.4f} indicates "
        
        if metrics['rouge-1']['f1'] < 0.3:
            latex_doc += "a relatively low overlap of unigrams between generated and reference questions."
        elif metrics['rouge-1']['f1'] < 0.6:
            latex_doc += "a moderate overlap of unigrams between generated and reference questions."
        else:
            latex_doc += "a good overlap of unigrams between generated and reference questions."
        
        latex_doc += f" The ROUGE-2 F1 score of {metrics['rouge-2']['f1']:.4f} indicates "
        
        if metrics['rouge-2']['f1'] < 0.2:
            latex_doc += "a relatively low overlap of bigrams."
        elif metrics['rouge-2']['f1'] < 0.4:
            latex_doc += "a moderate overlap of bigrams."
        else:
            latex_doc += "a good overlap of bigrams."
        
        latex_doc += f" The ROUGE-L F1 score of {metrics['rouge-l']['f1']:.4f} indicates "
        
        if metrics['rouge-l']['f1'] < 0.3:
            latex_doc += "a relatively low overlap of longest common subsequences."
        elif metrics['rouge-l']['f1'] < 0.6:
            latex_doc += "a moderate overlap of longest common subsequences."
        else:
            latex_doc += "a good overlap of longest common subsequences."
        
        latex_doc += """
    
    \\item \\textbf{F1 Score}: """
        
        latex_doc += f"The F1 score of {metrics['f1']:.4f} indicates "
        
        if metrics['f1'] < 0.3:
            latex_doc += "a relatively low balance between precision and recall."
        elif metrics['f1'] < 0.6:
            latex_doc += "a moderate balance between precision and recall."
        else:
            latex_doc += "a good balance between precision and recall."
        
        latex_doc += """
    
    \\item \\textbf{METEOR Score}: """
        
        latex_doc += f"The METEOR score of {metrics['meteor']:.4f} indicates "
        
        if metrics['meteor'] < 0.3:
            latex_doc += "a relatively low alignment between generated and reference questions, considering word order and synonyms."
        elif metrics['meteor'] < 0.6:
            latex_doc += "a moderate alignment between generated and reference questions, considering word order and synonyms."
        else:
            latex_doc += "a good alignment between generated and reference questions, considering word order and synonyms."
        
        latex_doc += """
\\end{itemize}

\\section{Conclusion}

Based on the evaluation results, our Math Question Generator model demonstrates """
        
        avg_score = (metrics['bleu'] + metrics['f1'] + metrics['meteor'] + metrics['rouge-1']['f1'] + metrics['rouge-2']['f1'] + metrics['rouge-l']['f1']) / 6
        
        if avg_score < 0.3:
            latex_doc += "room for improvement in generating questions that are similar to reference questions. Future work will focus on enhancing the model's ability to capture the structure and content of mathematical questions."
        elif avg_score < 0.6:
            latex_doc += "moderate performance in generating questions that are similar to reference questions. The model captures some aspects of the reference questions but could be improved further."
        else:
            latex_doc += "good performance in generating questions that are similar to reference questions. The model effectively captures the structure and content of mathematical questions."
        
        latex_doc += """

Future work will focus on:
\\begin{itemize}
    \\item Incorporating more diverse mathematical contexts
    \\item Enhancing the question deduplication mechanism
    \\item Fine-tuning the model on domain-specific mathematical content
\\end{itemize}

\\end{document}
"""
        
        # Save the LaTeX document
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(latex_doc)
        
        print(f"LaTeX report saved to {output_file}")

# Example usage
if __name__ == "__main__":
    evaluator = QuestionEvaluator()
    
    # Example references and candidates
    references = [
        "What is the quadratic formula used to solve quadratic equations?",
        "In a right triangle with legs of length 3 and 4, what is the length of the hypotenuse?"
    ]
    
    candidates = [
        "What formula is used to solve quadratic equations?",
        "If a right triangle has legs of length 3 and 4, what is the hypotenuse length?"
    ]
    
    # Evaluate
    results = evaluator.evaluate_questions(references, candidates)
    
    # Generate LaTeX report
    evaluator.generate_latex_report({"Our Model": results})
    
    print("Evaluation results:")
    print(f"BLEU Score: {results['bleu']:.4f}")
    print(f"METEOR Score: {results['meteor']:.4f}")
    print(f"F1 Score: {results['f1']:.4f}")
    print(f"ROUGE-1 F1: {results['rouge-1']['f1']:.4f}")
    print(f"ROUGE-2 F1: {results['rouge-2']['f1']:.4f}")
    print(f"ROUGE-L F1: {results['rouge-l']['f1']:.4f}")