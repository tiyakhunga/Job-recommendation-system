# AI-Powered Job Recommendation System 🚀

An end-to-end **Job Recommendation System** that matches resumes with job descriptions using **NLP, Machine Learning, and Generative AI**.  
The system computes similarity scores, performs skill gap analysis, and generates **human-readable recruiter-style explanations** using a local LLM.

---

## 🔍 Features

- 📄 Resume parsing from PDF files  
- 🧾 Job description parsing  
- 🧹 Text cleaning & normalization  
- 📊 TF-IDF vectorization + cosine similarity  
- 🧠 Skill extraction & skill gap analysis  
- 🤖 Local LLM (GPT4All) explanations (“Why this match?”)  
- 🌐 Interactive Streamlit web app  
- 📥 Downloadable evaluation reports  

---

## 🧠 Tech Stack

- **Python**
- **Pandas, NumPy**
- **Scikit-learn**
- **NLTK / Regex**
- **PDFPlumber**
- **TF-IDF + Cosine Similarity**
- **GPT4All (Local LLM)**
- **Streamlit**

---

## 🏗 Project Architecture

job_rec_sys/
│
├── data/                       # Data storage and outputs
│   ├── Resumes/                # Raw resume files
│   ├── JD.xlsx                 # Job Description source
│   ├── jobs_raw.csv            # Raw job data
│   ├── resumes_raw.csv         # Raw resume data
│   ├── parsed_resumes.csv      # LLM-parsed structured data
│   ├── final_results_with_explanations.csv # Final match output
│   └── skill_gap_analysis.csv  # Identified missing skills
│
├── notebook/                   # Core logic and processing
│   ├── extract_jobs.ipynb      # Job data extraction
│   ├── extract_resume.ipynb    # Resume text extraction
│   ├── text_preprocessing_v2.ipynb # Text cleaning and normalization
│   ├── llm_local.ipynb         # Local LLM configuration
│   ├── parsing.ipynb           # Structured data parsing
│   ├── text_vectorization.ipynb# Text-to-vector transformation
│   ├── cosin_similarity.ipynb  # Matching and ranking logic
│   ├── skill_gap_analysis.ipynb# Comparison and gap identification
│   └── evaluation.ipynb        # System performance testing
│
├── app.py                      # User interface/Application
├── requirements.txt            # Dependency list
└── README.md                   # Project documentation

## 🚀 How It Works (High Level)

1. Extract text from resumes (PDF) and job descriptions  
2. Clean and normalize text  
3. Convert text into numerical vectors (TF-IDF)  
4. Compute similarity using cosine similarity  
5. Extract skills and identify missing skills  
6. Generate recruiter-style explanations using a local LLM  
7. Display results via Streamlit UI  

---

## 🖥 Streamlit App Modes

### 📊 Precomputed Job Matching
- View ranked resumes for each job
- Skill match percentage
- Missing skills
- LLM explanations
- Download results

### 📤 Upload Resume & JD
- Upload resume PDF
- Paste job description
- Get real-time match score
- Skill gap analysis
- “Why this match?” explanation

---

## ⚙️ Installation & Setup

```bash
pip install -r requirements.txt
```

Run the Streamlit app:

```bash
streamlit run app.py
```

> Note: Raw resume PDFs and extracted resume text are not included in this public version to avoid exposing personal data.
