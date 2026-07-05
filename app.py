import streamlit as st
import pandas as pd
import pdfplumber
import re
import ast
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from gpt4all import GPT4All

def extract_text_from_pdf(uploaded_file):
    text = ""
    with pdfplumber.open(uploaded_file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + " "
    return text

def clean_text(text):
        text = text.lower()
        text = re.sub(r'\S+@\S+', ' ', text)
        text = re.sub(r'http\S+|www\S+', ' ', text)
        text = re.sub(r'[^a-z\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

def extract_skills(text, skill_list):
    return sorted([
        skill for skill in skill_list
        if skill in text
    ])

#Create a Skill Dictionary
PROGRAMMING_LANGUAGES = [
    "python", "java", "c", "c++", "c#", "javascript",
    "typescript", "go", "rust", "r", "scala", "kotlin",
    "php", "swift", "dart", "bash"
]
DATA_SKILLS = [
    "sql", "mysql", "postgresql", "mongodb",
    "data analysis", "data analytics", "data visualization",
    "excel", "power bi", "tableau", "statistics",
    "pandas", "numpy", "matplotlib", "seaborn"
]
ML_AI_SKILLS = [
    "machine learning", "deep learning", "artificial intelligence",
    "supervised learning", "unsupervised learning",
    "nlp", "computer vision", "time series",
    "feature engineering", "model training", "model evaluation",
    "scikit-learn", "tensorflow", "pytorch", "keras"
]
GENAI_SKILLS = [
    "llm", "large language model", "generative ai",
    "prompt engineering", "openai api",
    "langchain", "llamaindex", "rag",
    "vector database", "embeddings", "faiss", "pinecone"
]
BACKEND_SKILLS = [
    "backend", "rest api", "api development",
    "django", "flask", "fastapi",
    "spring boot", "nodejs", "express",
    "microservices", "authentication", "authorization"
]
FRONTEND_SKILLS = [
    "html", "css", "javascript", "react",
    "angular", "vue", "nextjs",
    "bootstrap", "tailwind", "ui", "ux"
]
DATABASE_BIGDATA_SKILLS = [
    "database", "sql server", "oracle",
    "redis", "elasticsearch",
    "hadoop", "spark", "pyspark",
    "kafka", "airflow", "etl"
]
CLOUD_DEVOPS_SKILLS = [
    "aws", "azure", "gcp",
    "docker", "kubernetes",
    "ci/cd", "jenkins",
    "terraform", "ansible",
    "linux", "unix"
]
SOFTWARE_PRACTICES = [
    "git", "github", "gitlab",
    "version control",
    "unit testing", "integration testing",
    "agile", "scrum", "jira",
    "design patterns", "system design"
]
DATA_ENGINEERING_SKILLS = [
    "data pipeline", "data warehouse",
    "bigquery", "redshift", "snowflake",
    "airflow", "dbt", "etl", "elt"
]
SECURITY_MISC_SKILLS = [
    "cybersecurity", "encryption",
    "oauth", "jwt",
    "performance optimization",
    "scalability", "monitoring"
]

SKILLS = (
    PROGRAMMING_LANGUAGES
    + DATA_SKILLS
    + ML_AI_SKILLS
    + GENAI_SKILLS
    + BACKEND_SKILLS
    + FRONTEND_SKILLS
    + DATABASE_BIGDATA_SKILLS
    + CLOUD_DEVOPS_SKILLS
    + SOFTWARE_PRACTICES
    + DATA_ENGINEERING_SKILLS
    + SECURITY_MISC_SKILLS
)

#load llm model
@st.cache_resource
def load_llm_model():
    return GPT4All(
        "orca-mini-3b-gguf2-q4_0",
        allow_download=True
    )

llm_model = load_llm_model()

#llm model
def generate_llm_explanation(
    resume_skills,
    jd_skills,
    missing_skills,
    match_percent
):
    prompt = f"""
You are an HR recruiter.

Candidate Skills:
{', '.join(resume_skills) if resume_skills else 'None'}

Job Required Skills:
{', '.join(jd_skills) if jd_skills else 'None'}

Missing Skills:
{', '.join(missing_skills) if missing_skills else 'None'}

Match Score:
{match_percent}%

Explain in simple, non-technical language:
- Why this resume matches the job
- What is good about the profile
- What can be improved
Limit to 4–5 lines.
"""

    response = llm_model.generate(
        prompt,
        max_tokens=180,
        temp=0.4
    )

    return response.strip()
#llm model








st.title("Job Recommendation System")
#slider bar
st.sidebar.title("Mode Selection")


mode = st.sidebar.radio(
    "Choose Mode",
    ["📊 Precomputed Job Matching", "📤 Upload Resume & JD"]
)
if mode == "📊 Precomputed Job Matching":

    @st.cache_data
    def load_data():
        return pd.read_csv("data/final_results_with_explanations.csv")

    df = load_data()

    st.success("Evaluation results loaded successfully!")

    # Select Job ID
    st.subheader("Select Job Role")

    job_ids = sorted(df["job_id"].unique())
    selected_job = st.selectbox("Choose a Job ID", job_ids)

    # TOP-K SLIDER
    top_k = st.slider(
        "Select number of top candidates to display",
        min_value=1,
        max_value=5,
        value=3
    )

    # Filter results
    job_results = (
        df[df["job_id"] == selected_job]
        .sort_values("rank")
        .head(top_k)
    )

    # Table
    st.subheader("Top Matching Resumes")

    st.dataframe(
        job_results[[
            "resume_id",
            "rank",
            "similarity_score",
            "skill_match_percent",
            "missing_skills"
        ]]
    )

    # Detailed insights
    st.subheader("Detailed Match Insights")

    for _, row in job_results.iterrows():
        st.markdown(f"### 📄 Resume: {row['resume_id']}")

        st.write(f"**Rank:** {row['rank']}")
        st.write(f"**Text Similarity Score:** {round(row['similarity_score'], 3)}")
        st.write(f"**Skill Match:** {round(row['skill_match_percent'], 2)}%")

        missing = ast.literal_eval(row["missing_skills"])

        if len(missing) == 0:
            st.success("✅ Strong match — no missing skills")
        else:
            st.warning(f"⚠️ Missing Skills: {', '.join(missing)}")

        if pd.notna(row["llm_explanation"]) and row["llm_explanation"].strip():
            st.markdown("### 🧠 Why this match?")
            st.info(row["llm_explanation"])

    # Download
    st.download_button(
        label="📥 Download Results as CSV",
        data=job_results.to_csv(index=False),
        file_name=f"{selected_job}_top_matches.csv",
        mime="text/csv"
    )

if mode == "📤 Upload Resume & JD":

    st.header("📤 Upload Resume and Job Description")

    resume_file = st.file_uploader(
        "Upload Resume (PDF)",
        type=["pdf"]
    )

    jd_text = st.text_area(
        "Paste Job Description here",
        height=200
    )
    if resume_file and jd_text.strip():

        resume_text = extract_text_from_pdf(resume_file)

        st.success("Resume and Job Description received successfully!")
    
        clean_resume = clean_text(resume_text)
        clean_jd = clean_text(jd_text)

        vectorizer = TfidfVectorizer()
        vectors = vectorizer.fit_transform([clean_resume, clean_jd])

        similarity_score = cosine_similarity(vectors[0], vectors[1])[0][0]
        match_percent = round(similarity_score * 100, 2)

        st.subheader("📊 Match Result")

        st.metric(
            label="Resume–Job Match Score",
            value=f"{match_percent}%"
        )

        resume_skills = extract_skills(clean_resume, SKILLS)
        jd_skills = extract_skills(clean_jd, SKILLS)

        missing_skills = list(set(jd_skills) - set(resume_skills))

        st.write("✅ Resume Skills:", ", ".join(resume_skills) if resume_skills else "None")

        if missing_skills:
            st.warning(f"⚠️ Missing Skills: {', '.join(missing_skills)}")
        else:
            st.success("🎉 Excellent match — no missing skills")

        if st.button("🧠 Explain Match"):
            with st.spinner("Thinking like a recruiter... 🤔"):
                explanation = generate_llm_explanation(
                    resume_skills,
                    jd_skills,
                    missing_skills,
                    match_percent
                )

            if explanation:
                st.markdown("### 🧠 Why this match?")
                st.info(explanation)
