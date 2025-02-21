from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pymongo import MongoClient
from pdfminer.high_level import extract_text
from docx import Document
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from langchain_groq import ChatGroq
from datetime import datetime
from dotenv import load_dotenv
import os
import re
import spacy

# Load environment variables
load_dotenv()

# Get environment variables
GROQ_API_KEY = os.getenv('GROQ_API_KEY')
EMAIL_API = os.getenv('EMAIL_API')
MONGO_DB = os.getenv('MONGO_DB')
MODEL_NAME = os.getenv('MODEL_NAME', 'llama3-70b-8192')
SENTENCE_TRANSFORMER = os.getenv('SENTENCE_TRANSFORMER', 'all-MiniLM-L6-v2')
CORS_ORIGIN = os.getenv('CORS_ORIGIN', 'https://your-frontend-domain.com')

# Creating an instance 
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[CORS_ORIGIN],  # Your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load models
nlp = spacy.load("en_core_web_sm")
model = SentenceTransformer(SENTENCE_TRANSFORMER)

# MongoDB connection
client = MongoClient(MONGO_DB)
db = client["resume_database"]

def extract_text_from_pdf(file_path):
    return extract_text(file_path)

def extract_text_from_docx(file_path):
    doc = Document(file_path)
    return " ".join([para.text for para in doc.paragraphs])

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    doc = nlp(text)
    return " ".join([token.lemma_ for token in doc if not token.is_stop])

@app.post("/upload/")
async def upload_resume(file: UploadFile = File(...), job_desc: str = Form(...)):
    file_path = f"temp_{file.filename}"
    try:
        # Save uploaded file
        with open(file_path, "wb") as f:
            f.write(file.file.read())

        # Extract text
        if file.filename.endswith('.pdf'):
            raw_text = extract_text_from_pdf(file_path)
        else:
            raw_text = extract_text_from_docx(file_path)
        
        cleaned_text = preprocess_text(raw_text)

        # Store in MongoDB with timestamp
        resume_data = {
            "name": file.filename,
            "resume_text": cleaned_text,
            "status": "unprocessed",
            "processed_at": datetime.utcnow()  # Store processing timestamp
        }
        db["resumes"].insert_one(resume_data)

        # Compute similarity
        resume_embedding = model.encode(cleaned_text).tolist()
        job_embedding = model.encode(job_desc).tolist()
        similarity = cosine_similarity([resume_embedding], [job_embedding])[0][0]

        # Generate analysis with LLM
        feedback_prompt = f"""
        Resume: {cleaned_text}
        Job Description: {job_desc}
        Generate detailed analysis with:
        1. Top 3 matching qualifications
        2. Missing skills
        3. Improvement suggestions
        4. Overall suitability score ({similarity:.2f}/1.00)
        """
        llm = ChatGroq(model_name=MODEL_NAME, temperature=0, groq_api_key=GROQ_API_KEY)
        analysis = llm.invoke(feedback_prompt).content

        return {"score": round(similarity, 2), "analysis": analysis}
    finally:
        # Add cleanup
        if os.path.exists(file_path):
            os.remove(file_path)

@app.get("/")
async def read_root():
    return {"Hello": "World"}

# Analytics Endpoints
@app.get("/api/analytics/skills")
async def get_skills_analytics():
    pipeline = [
        {"$unwind": "$skills"},
        {"$group": {"_id": "$skills", "count": {"$sum": 1}}},
        {"$sort": {"count": -1}},
        {"$limit": 10}
    ]
    skills = list(db["parsed_resume"].aggregate(pipeline))
    return [{"name": skill["_id"], "value": skill["count"]} for skill in skills]

@app.get("/api/analytics/education")
async def get_education_analytics():
    pipeline = [
        {"$unwind": "$education"},
        {"$group": {"_id": "$education", "count": {"$sum": 1}}},
        {"$sort": {"count": -1}}
    ]
    education = list(db["parsed_resume"].aggregate(pipeline))
    return [{"name": item["_id"], "value": item["count"]} for item in education]

@app.get("/api/analytics/similarity-scores")
async def get_similarity_scores():
    scores = list(db["shortlisted_candidates"].find({}, {"_id": 0, "score": 1}))
    return [{"score": f"{int(s['score']*100)}-{int(s['score']*100)+5}", "count": 1} for s in scores]

@app.get("/api/analytics/top-candidates")
async def get_top_candidates():
    candidates = list(db["shortlisted_candidates"].find().sort("score", -1).limit(5))
    return [{"name": c["name"], "score": c["score"]*100} for c in candidates]

@app.get("/api/analytics/selection-status")
async def get_selection_status():
    pipeline = [
        {"$group": {"_id": "$status", "count": {"$sum": 1}}}
    ]
    status = list(db["shortlisted_candidates"].aggregate(pipeline))
    return [{"name": s["_id"], "value": s["count"]} for s in status]

# Run with: python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload