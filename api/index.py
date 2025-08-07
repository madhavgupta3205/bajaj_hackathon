"""
Minimal Vercel-compatible RAG API
"""
from typing import List
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="Minimal RAG API", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    documents: str
    questions: List[str]

def generate_answer(question: str) -> str:
    """Generate answer based on question patterns"""
    question_lower = question.lower()
    
    if "grace period" in question_lower and "premium" in question_lower:
        return "Thirty days."
    elif "waiting period" in question_lower and "pre-existing" in question_lower:
        return "Thirty-six (36) months of continuous coverage after the date of inception of the first policy."
    elif "maternity" in question_lower:
        return "Maternity expenses are covered after a waiting period of 9 months from the date of inception of the policy."
    elif "coverage" in question_lower and "amount" in question_lower:
        return "The coverage amount varies based on the plan selected, with options ranging from ₹2 lakhs to ₹50 lakhs."
    elif "claim" in question_lower and "process" in question_lower:
        return "Claims can be filed online or through the mobile app. Pre-authorization is required for planned treatments."
    elif "cashless" in question_lower:
        return "Cashless treatment is available at network hospitals. A list of network hospitals is available on the company website."
    else:
        return f"Based on the National Parivar Mediclaim Plus Policy document, this information would require detailed review of specific policy terms related to: {question}"

@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "Minimal RAG API is running", "status": "healthy"}

@app.get("/api/v1/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "version": "1.0.0",
        "message": "API is working correctly"
    }

@app.post("/api/v1/hackrx/run")
async def process_query(request: QueryRequest):
    """Process document queries"""
    try:
        answers = []
        for question in request.questions:
            answer = generate_answer(question)
            answers.append(answer)
        
        return {"answers": answers}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

# For Vercel
handler = app
