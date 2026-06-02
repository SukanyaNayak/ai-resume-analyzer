from dotenv import load_dotenv
import streamlit as st
import PyPDF2
from docx import Document
import plotly.express as px
import pandas as pd
import time
import google.generativeai as genai
import os
import logging
import json
load_dotenv()
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

logging.getLogger("transformers").setLevel(logging.ERROR)
load_dotenv()

# Get API key from .env
api_key = os.getenv("GEMINI_API_KEY")

# Configure Gemini
genai.configure(api_key=api_key)
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
@st.cache_resource
def load_sentence_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model_nlp = load_sentence_model()

import joblib

ML_model = joblib.load("logistic_model.pkl")
gemini_model = genai.GenerativeModel("models/gemini-2.5-flash")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
scaler = joblib.load("scaler.pkl")

def extract_skills_ai(job_description, resume_text):

    # =====================================================
    # CASE 1 → ONLY ROLE NAME
    # =====================================================

    if len(job_description.split()) <= 3:

        required_skills = get_role_skills(job_description)

        prompt = f"""
        Compare these required skills with the resume.

        Return ONLY valid JSON.

        {{
            "matched_skills": [],
            "missing_skills": []
        }}

        Required Skills:
        {required_skills}

        Resume:
        {resume_text}
        """

    # =====================================================
    # CASE 2 → FULL JOB DESCRIPTION
    # =====================================================

    else:

        prompt = f"""
        Compare the resume and job description.

        Return ONLY valid JSON.

        {{
            "matched_skills": [],
            "missing_skills": []
        }}

        Job Description:
        {job_description}

        Resume:
        {resume_text}
        """

    # =====================================================
    # GEMINI RESPONSE
    # =====================================================

    response = gemini_model.generate_content(
        prompt,
        generation_config={
            "response_mime_type": "application/json"
        }
    )

    raw_text = response.text.strip()

    try:

        result = json.loads(raw_text)

        matched_skills = result.get("matched_skills", [])
        missing_skills = result.get("missing_skills", [])

        return matched_skills, missing_skills

    except json.JSONDecodeError:

        st.error("Invalid JSON received")
        st.write(raw_text)

        return [], []
# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="AI ATS Resume Analyzer",
    page_icon="🚀",
    layout="wide"
)

# =====================================================
# CUSTOM CSS
# =====================================================

st.markdown("""
<style>

/* Main App */
.stApp {
    background: linear-gradient(to right, #0f2027, #203a43, #2c5364);
    color: white;
}

/* Main Title */
.main-title {
    font-size: 55px;
    font-weight: bold;
    text-align: center;
    color: #00FFAA;
    margin-top: 20px;
}

/* Subtitle */
.sub-title {
    text-align: center;
    color: #DDDDDD;
    font-size: 22px;
    margin-bottom: 30px;
}

/* Cards */
.card {
    background-color: rgba(255,255,255,0.05);
    padding: 25px;
    border-radius: 20px;
    backdrop-filter: blur(10px);
    box-shadow: 0px 0px 15px rgba(0,255,170,0.2);
    margin-bottom: 20px;
}

/* Skill Tags */
.skill-box {
    background-color: #00FFAA;
    color: black;
    padding: 8px 15px;
    border-radius: 12px;
    display: inline-block;
    margin: 5px;
    font-weight: bold;
}

/* Missing Skills */
.missing-skill {
    background-color: #FF4B4B;
    color: white;
    padding: 8px 15px;
    border-radius: 12px;
    display: inline-block;
    margin: 5px;
    font-weight: bold;
}

/* Suggestions */
.suggestion {
    background-color: rgba(255,255,255,0.08);
    padding: 15px;
    border-left: 5px solid #00FFAA;
    border-radius: 10px;
    margin-bottom: 10px;
}

/* File Upload */
[data-testid="stFileUploader"] {
    background-color: rgba(255,255,255,0.05);
    padding: 20px;
    border-radius: 15px;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #111827;
}

/* Button */
.stButton button {

    background-color: #00FFAA;
    color: black;
    font-size: 18px;
    font-weight: bold;
    border-radius: 12px;
    height: 50px;
    width: 100%;
    border: none;
}

.stButton button:hover {

    background-color: #00cc88;
    color: white;
}

</style>
""", unsafe_allow_html=True)

# =====================================================
# HEADER
# =====================================================

st.markdown(
    "<div class='main-title'>TalentMatch AI</div>",
    unsafe_allow_html=True
)

st.markdown(
    "<div class='sub-title'>Smart Resume Analysis, ATS Scoring & Skill Matching</div>",
    unsafe_allow_html=True
)

# =====================================================
# SIDEBAR
# =====================================================

st.sidebar.title("🚀 Features")

st.sidebar.info(
    """
    🚀 Features

✔ Resume Analysis & Evaluation

✔ ATS Score Prediction

✔ Resume and Job Description Matching

✔ Skill Extraction & Analysis

✔ Resume Improvement Suggestions

✔ AI-Based Candidate Screening

    """
)

# =====================================================
# JOB DESCRIPTION INPUT
# =====================================================

job_description = st.text_area(
    "📋 Enter Job Role or Paste Job Description",
    height=250,
    placeholder="""Examples:
    Marketing
    Data Scientist
    UI/UX Designer

    OR paste full LinkedIn job description here..."""
)

# =====================================================
# FILE UPLOADER
# =====================================================

uploaded_file = st.file_uploader(
    "📄 Upload Resume",
    type=["pdf", "docx"]
)

# =====================================================
# ANALYZE BUTTON
# =====================================================

submit_button = st.button("🚀 Analyze Resume")

# =====================================================
# PDF TEXT EXTRACTION
# =====================================================

def extract_text_from_pdf(file):

    text = ""

    pdf_reader = PyPDF2.PdfReader(file)

    for page in pdf_reader.pages:

        extracted_text = page.extract_text()

        if extracted_text:

            text += extracted_text

    return text

# =====================================================
# DOCX TEXT EXTRACTION
# =====================================================

def extract_text_from_docx(file):

    doc = Document(file)

    text = ""

    for para in doc.paragraphs:

        text += para.text + "\n"

    return text
# =====================================================
# AI SEMANTIC MATCHING FUNCTION
# =====================================================

def get_similarity(resume_text, job_description):

    embeddings = model_nlp.encode([
        resume_text,
        job_description
    ])

    similarity = cosine_similarity(
        [embeddings[0]],
        [embeddings[1]]
    )[0][0]

    return similarity
# =====================================================
# ROLE SKILLS GENERATOR
# =====================================================

def get_role_skills(job_role):

    prompt = f"""
    Give ATS skills required for a {job_role} role.

    Return ONLY valid JSON.

    Example:
    {{
        "skills": ["SEO", "Marketing", "Google Analytics"]
    }}
    """

    response = gemini_model.generate_content(
        prompt,
        generation_config={
            "response_mime_type": "application/json"
        }
    )

    raw_text = response.text.strip()

    try:

        data = json.loads(raw_text)

        return data.get("skills", [])

    except json.JSONDecodeError:

        st.error("Invalid JSON from Gemini")
        st.write(raw_text)

        return []
# =====================================================
# ANALYSIS FUNCTION
# =====================================================
def analyze_resume(resume_text, job_description):

    resume_text = resume_text.lower()
    job_description = job_description.lower()

    # =====================================================
    # SEMANTIC SIMILARITY
    # =====================================================

    similarity_score = get_similarity(
        resume_text,
        job_description
    )

    ats_score = int(similarity_score * 100)

    # =====================================================
    # AI SKILL EXTRACTION
    # =====================================================

    matched_skills, missing_skills = extract_skills_ai(
        job_description,
        resume_text
    )

    # =====================================================
    # PROJECTS & EXPERIENCE
    # =====================================================

    projects = resume_text.count("project")

    experience = 0

    if "internship" in resume_text:
        experience += 1

    if "experience" in resume_text:
        experience += 2

    github = "github" in resume_text

    # =====================================================
    # MACHINE LEARNING FEATURES
    # =====================================================

    features = pd.DataFrame([{

        "years_experience": experience,

        "skills_match_score": ats_score,

        "education_level": 1,

        "project_count": projects,

        "resume_length": len(resume_text),

        "github_activity": 1 if github else 0

    }])

    # =====================================================
    # SCALING
    # =====================================================

    features_scaled = scaler.transform(features)

    # =====================================================
    # PREDICTION
    # =====================================================

    prediction_value = ML_model.predict(
        features_scaled
    )[0]

    if prediction_value == 1:
        prediction = "SELECTED"

    else:
        prediction = "REJECTED"

    # =====================================================
    # SUGGESTIONS
    # =====================================================

    suggestions = []

    if ats_score < 70:

        suggestions.append(
            "Improve ATS Score by adding more relevant skills."
        )

    if not github:

        suggestions.append(
            "Add GitHub profile link."
        )

    if projects < 2:

        suggestions.append(
            "Add more projects."
        )

    if experience == 0:

        suggestions.append(
            "Add internship or work experience."
        )

    # =====================================================
    # RETURN
    # =====================================================

    return (

        ats_score,

        matched_skills,

        missing_skills,

        prediction,

        suggestions,

        similarity_score

    )

# =====================================================
# MAIN APP
# =====================================================

if uploaded_file is not None and submit_button:

    if job_description.strip() == "":

        st.warning("Please paste a LinkedIn Job Description.")

    else:

        with st.spinner("Analyzing Resume..."):

            file_type = uploaded_file.name.split(".")[-1]

            # =====================================================
            # TEXT EXTRACTION
            # =====================================================

            if file_type == "pdf":

                resume_text = extract_text_from_pdf(uploaded_file)

            elif file_type == "docx":

                resume_text = extract_text_from_docx(uploaded_file)

            else:

                st.error("Unsupported File Format")

            # =====================================================
            # ANALYSIS
            # =====================================================

            (
                ats_score,
                matched_skills,
                missing_skills,
                prediction,
                suggestions,
                similarity_score
            ) = analyze_resume(
                resume_text,
                job_description
            )

        st.success("Resume Analysis Completed Successfully!")

        # =====================================================
        # DASHBOARD
        # =====================================================

        col1, col2, col3 = st.columns(3)

        with col1:

            st.markdown(
                f"""
                <div class="card">
                <h2>ATS Score</h2>
                <h1>{ats_score}%</h1>
                </div>
                """,
                unsafe_allow_html=True
            )

            st.progress(ats_score / 100)

        with col2:

            color = "#00FFAA" if prediction == "SELECTED" else "#FF4B4B"

            st.markdown(
                f"""
                <div class="card">
                <h2>Resume Status</h2>
                <h1 style='color:{color};'>{prediction}</h1>
                </div>
                """,
                unsafe_allow_html=True
            )

        with col3:

            st.markdown(
                f"""
                <div class="card">
                <h2>Similarity</h2>
                <h1>{round(similarity_score * 100, 2)}%</h1>
                </div>
                """,
                unsafe_allow_html=True
            )

        # =====================================================
        # MATCHED SKILLS
        # =====================================================

        st.markdown(
            "<div class='card'><h2>✅ Matched Skills</h2>",
            unsafe_allow_html=True
        )

        if matched_skills:

            for skill in matched_skills:

                st.markdown(
                    f"<span class='skill-box'>{skill}</span>",
                    unsafe_allow_html=True
                )

        else:

            st.warning("No matching skills found.")

        st.markdown("</div>", unsafe_allow_html=True)

        # =====================================================
        # MISSING SKILLS
        # =====================================================

        st.markdown(
            "<div class='card'><h2>❌ Missing Skills</h2>",
            unsafe_allow_html=True
        )

        if missing_skills:

            for skill in missing_skills:

                st.markdown(
                    f"<span class='missing-skill'>{skill}</span>",
                    unsafe_allow_html=True
                )

        else:

            st.success("No missing skills.")

        st.markdown("</div>", unsafe_allow_html=True)

        # =====================================================
        # PIE CHART
        # =====================================================

        skill_data = pd.DataFrame({

            "Category": [
                "Matched Skills",
                "Missing Skills"
            ],

            "Count": [
                len(matched_skills),
                len(missing_skills)
            ]
        })

        fig = px.pie(
            skill_data,
            names="Category",
            values="Count",
            title="Skills Analysis"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        # =====================================================
        # SUGGESTIONS
        # =====================================================

        st.markdown(
            "<div class='card'><h2>💡 Resume Improvement Suggestions</h2>",
            unsafe_allow_html=True
        )

        for suggestion in suggestions:

            st.markdown(
                f"""
                <div class='suggestion'>
                {suggestion}
                </div>
                """,
                unsafe_allow_html=True
            )

        st.markdown("</div>", unsafe_allow_html=True)

        # =====================================================
        # RESUME TEXT
        # =====================================================

        with st.expander("📄 View Extracted Resume Text"):

            st.write(resume_text)

# =====================================================
# WARNINGS
# =====================================================

elif submit_button and uploaded_file is None:

    st.warning("Please upload a resume first.")
