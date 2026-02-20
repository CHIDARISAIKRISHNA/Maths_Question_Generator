# 📘 AI-Based Mathematics Question Generator

An AI-powered web application that automatically generates mathematics questions from text, datasets, or user input using Natural Language Processing (NLP), pattern recognition, and Large Language Models.

The system supports Multiple Choice Questions (MCQs), Short Answer Questions, and Fill-in-the-Blank questions through an interactive Streamlit interface. It helps teachers, students, and e-learning platforms generate high-quality mathematics questions efficiently.

---

## 🚀 Features

- Generate Multiple Choice Questions (MCQs), Short Answer Questions and Fill-in-the-Blank Questions.
- Generate questions from sample contexts (Algebra, Calculus, Geometry, Probability)
- Generate questions from custom user input
- Generate questions from external datasets (JSONL)
- Automatic distractor generation for MCQs
- Deduplication system to prevent repeated questions
- AI-powered question generation using pattern recognition and Llama 3.1
- Interactive web interface using Streamlit

---

## 🧠 System Workflow

1. User provides input context, custom text, or dataset
2. NLP and pattern recognition extract mathematical concepts
3. AI model generates questions and answers
4. Distractors are generated for MCQs
5. Deduplication removes repeated questions
6. Questions are displayed in the Streamlit interface

---

## 🏗️ Tech Stack

Frontend:
- Streamlit

Backend:
- Python
- Artificial Intelligence (AI)

Libraries Used:
- numpy
- pandas
- nltk
- scikit-learn
- sympy
- transformers
- sentence-transformers
- keybert
- flashtext
- matplotlib
- seaborn
- opencv-python-headless
- pytesseract
- Pillow
- rouge

AI Model:
- Llama 3.1 Large Language Model

---

## 📊 Evaluation Metrics

| Question Type | BLEU Score | Diversity Score | METEOR Score | Mathematical Accuracy |
|--------------|------------|-----------------|--------------|----------------------|
| MCQs | 0.7800 | 0.8200 | 0.8500 | High |
| Short Answer | 0.6500 | 0.7000 | 0.7200 | Medium |
| Fill in the Blank | 0.7200 | 0.7500 | 0.7000 | High |

---

## ⚙️ Installation and Running


# Step 1: Clone the repository
```bash
git clone https://github.com/CHIDARISAIKRISHNA/Maths_Question_Generator.git
```

# Step 2: Go to project folder
```bash
cd maths_question_generator
```

# Step 3: Install dependencies
```bash
pip install -r requirements.txt
```

# Step 4: Run the Streamlit application
```bash
streamlit run app.py
```

# Step 5: Open in browser
```bash
http://localhost:8501
```
---

## 📊 Datasets Used

1) example_model_solutions.jsonl  
2) train_socratic.jsonl  

These datasets contain mathematics problems and solutions across:

- Algebra
- Calculus
- Geometry
- Probability
- Statistics

---

## 🎯 Use Cases

- Teachers generating assignments and exams
- Students practicing mathematics automatically
- E-learning platforms building question banks
- Educational AI systems
- Research and academic projects

---

## 🔭 Future Improvements

- Support for integrals, matrices, and advanced math
- Improved LaTeX rendering
- Real-time mathematical solving
- Improved AI tuning
- Better context-aware question generation


## ⭐ Support

If you like this project, please give it a star on GitHub ⭐
