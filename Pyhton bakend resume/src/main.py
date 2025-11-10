# main.py
from fastapi import FastAPI, Depends, HTTPException, UploadFile, File, status, BackgroundTasks, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Boolean, Text, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session, relationship
from pydantic import BaseModel, EmailStr
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
import jwt
import bcrypt
import os
import PyPDF2
import io
import json
import logging
import hmac
import hashlib
from contextlib import asynccontextmanager
import re
import uuid
from dotenv import load_dotenv
load_dotenv()

# LangChain Gemini imports
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain

# AutoGen imports
import autogen
from autogen import AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager

# Razorpay
import razorpay

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Environment variables
SECRET_KEY = os.getenv("SECRET_KEY", "d4f3e5c6a7b8c9d0e1f2a3b4c5d6e7f890123456789abcdef0123456789abcdef")
# DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:Postgresql@localhost/postgres")
# DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://resumeai_wi80_user:5ao98cyZXg28yuBR21AqLahCC3O7qdkq@dpg-d3f9jhgdl3ps73dej75g-a/resumeai_wi80")
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://neondb_owner:npg_miIjQZ4KXO7D@ep-late-cherry-a10mtxze-pooler.ap-southeast-1.aws.neon.tech/neondb?sslmode=require&channel_binding=require")

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
print("GOOGLE API KEY",GOOGLE_API_KEY)
RAZORPAY_KEY_ID = os.getenv("RAZORPAY_KEY_ID")
RAZORPAY_KEY_SECRET = os.getenv("RAZORPAY_KEY_SECRET")

# Razorpay client
razorpay_client = razorpay.Client(auth=(RAZORPAY_KEY_ID, RAZORPAY_KEY_SECRET))

# Database setup
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# AutoGen Configuration
config_list = [
    {
        'model': 'gemini-1.5-flash',
        'api_key': GOOGLE_API_KEY,
        'base_url': 'https://generativelanguage.googleapis.com/v1beta',
        'api_type': 'google'
    }
]

# LangChain Gemini setup
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0.1,
    google_api_key=GOOGLE_API_KEY,
    convert_system_message_to_human=True 
)

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=GOOGLE_API_KEY
)

# Global AI agents
resume_analyzer_agent = None
scoring_agent = None
suggestion_agent = None
premium_insights_agent = None
group_chat_manager = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    await initialize_ai_agents()
    yield


app = FastAPI(title="ResumeAI Backend with Gemini & Razorpay", version="2.0.0", lifespan=lifespan)

# CORS middleware   
app.add_middleware(
    CORSMiddleware,
    # allow_origins=["https://resume-score-skth.onrender.com", "http://localhost:5173",],
    # allow_origins = ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    allow_origins=[
        "https://resume-score-murex.vercel.app",
        # "http://localhost:5173", # Vite frontend,
        # "https://resume-score-skth.onrender.com",
        # "*",
          # deployed frontend
    ]
)

# Security
security = HTTPBearer()

# Database Models
class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    full_name = Column(String)
    is_premium = Column(Boolean, default=False)
    credits_remaining = Column(Integer, default=10)
    subscription_id = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    analyses = relationship("ResumeAnalysis", back_populates="user")
    payments = relationship("Payment", back_populates="user")
    subscriptions = relationship("Subscription", back_populates="user")


class ResumeAnalysis(Base):
    __tablename__ = "resume_analyses"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    filename = Column(String)
    job_description = Column(Text, nullable=True)
    overall_score = Column(Float)
    ats_score = Column(Float, nullable=True)
    job_match_score = Column(Float, nullable=True)
    category_scores = Column(Text)
    suggestions = Column(Text)
    premium_insights = Column(Text, nullable=True)
    processing_status = Column(String, default="pending")
    created_at = Column(DateTime, default=datetime.utcnow)
    
    user = relationship("User", back_populates="analyses")


class Payment(Base):
    __tablename__ = "payments"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    razorpay_order_id = Column(String, unique=True)
    razorpay_payment_id = Column(String, unique=True, nullable=True)
    razorpay_signature = Column(String, nullable=True)
    amount = Column(Integer)  # Amount in paise
    currency = Column(String, default="INR")
    plan_name = Column(String)
    status = Column(String, default="pending")  # pending, completed, failed
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    user = relationship("User", back_populates="payments")


class Subscription(Base):
    __tablename__ = "subscriptions"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    plan_name = Column(String)
    status = Column(String, default="active")
    start_date = Column(DateTime, default=datetime.utcnow)
    end_date = Column(DateTime, nullable=True)
    auto_renew = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    user = relationship("User", back_populates="subscriptions")


# Create tables
Base.metadata.create_all(bind=engine)

# Pydantic models
class UserCreate(BaseModel):
    email: EmailStr
    password: str
    full_name: str


class UserLogin(BaseModel):
    email: EmailStr
    password: str


class UserResponse(BaseModel):
    id: int
    email: str
    full_name: str
    is_premium: bool
    credits_remaining: int
'''

class AnalysisResponse(BaseModel):
    id: int
    overall_score: Optional[float] = None
    ats_score: Optional[float] = None
    job_match_score: Optional[float] = None
    category_scores: Optional[Dict[str, float]] = None
    suggestions: Optional[List[str]] = None  
    premium_insights: Optional[List[str]] = None
    processing_status: str
'''
class SuggestionItem(BaseModel):
    area: str
    description: str
    action: str

class AnalysisResponse(BaseModel):
    id: int
    overall_score: Optional[float] = None
    ats_score: Optional[float] = None
    job_match_score: Optional[float] = None
    category_scores: Optional[Dict[str, float]] = None
    suggestions: Optional[List[Dict[str, Any]]] = None 
    premium_insights: Optional[List[str]] = None
    processing_status: str

class CreateOrderRequest(BaseModel):
    plan_name: str
    amount: int
    currency: str


class CreateOrderResponse(BaseModel):
    order_id: str
    amount: int
    currency: str


class VerifyPaymentRequest(BaseModel):
    razorpay_order_id: str
    razorpay_payment_id: str
    razorpay_signature: str
    plan_name: str


class VerifyPaymentResponse(BaseModel):
    success: bool
    message: str
    subscription_details: Optional[Dict[str, Any]] = None


# Plan configurations
PLAN_CONFIGS = {
    "Pro": {
        "credits": 50,
        "duration_days": 30,
        "price": 200  # in rupees
    },
    "Enterprise": {
        "credits": 999,
        "duration_days": 30,
        "price": 500  # in rupees
    }
}



# Dependency functions
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')


def verify_password(password: str, hashed: str) -> bool:
    return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))


def create_access_token(user_id: int) -> str:
    payload = {"user_id": user_id, "exp": datetime.utcnow() + timedelta(days=7)}
    return jwt.encode(payload, SECRET_KEY, algorithm="HS256")


def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)) -> int:
    try:
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=["HS256"])
        return payload["user_id"]
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")


# AI Agent Setup
async def initialize_ai_agents():
    global resume_analyzer_agent, scoring_agent, suggestion_agent, premium_insights_agent, group_chat_manager
    
    resume_analyzer_agent = AssistantAgent(
        name="ResumeAnalyzer",
        system_message="You are an expert resume analyzer.",
        llm_config={"config_list": config_list, "temperature": 0.1}
    )
    
    scoring_agent = AssistantAgent(
        name="ScoringAgent",
        system_message="You are a resume scoring specialist.",
        llm_config={"config_list": config_list, "temperature": 0.1}
    )
    
    suggestion_agent = AssistantAgent(
        name="SuggestionAgent",
        system_message="You are a career advisor providing actionable resume improvements.",
        llm_config={"config_list": config_list, "temperature": 0.2}
    )
    
    premium_insights_agent = AssistantAgent(
        name="PremiumInsightsAgent",
        system_message="You are a senior career strategist providing premium insights.",
        llm_config={"config_list": config_list, "temperature": 0.3}
    )
    
    user_proxy = UserProxyAgent(name="UserProxy", human_input_mode="NEVER")
    
    group_chat = GroupChat(
        agents=[resume_analyzer_agent, scoring_agent, suggestion_agent, premium_insights_agent, user_proxy],
        messages=[],
        max_round=10
    )
    
    group_chat_manager = GroupChatManager(groupchat=group_chat, llm_config={"config_list": config_list})


# Resume Processor
class LangChainResumeProcessor:
    def __init__(self):
        self.analysis_prompt = ChatPromptTemplate.from_messages([
            ("human", "You are an expert resume analyzer. Analyze the following resume:\n\n{resume_text}\n\nReturn the response in JSON format.")
        ])
        
        self.scoring_prompt = ChatPromptTemplate.from_messages([
            ("human", "You are a scoring specialist. Given this analysis:\n\n{analysis}\n\nProvide numerical scores in JSON.")
        ])
        
        self.suggestions_prompt = ChatPromptTemplate.from_messages([
            ("human", "You are a career advisor. Based on this analysis:\n\n{analysis}\n\nand these scores:\n\n{scores}\n\nSuggest 5–7 improvements in JSON format.")
        ])
        
        self.analysis_chain = LLMChain(llm=llm, prompt=self.analysis_prompt)
        self.scoring_chain = LLMChain(llm=llm, prompt=self.scoring_prompt)
        self.suggestions_chain = LLMChain(llm=llm, prompt=self.suggestions_prompt)
    
    async def process_resume(self, resume_text: str, job_description: Optional[str] = None, is_premium: bool = False) -> Dict[str, Any]:
        try:
            analysis_result = await self.analysis_chain.arun(resume_text=resume_text)
            logger.info(f"[Analysis Output]: {analysis_result}")

            scores_result = await self.scoring_chain.arun(analysis=analysis_result)
            logger.info(f"[Scores Output]: {scores_result}")

            suggestions_result = await self.suggestions_chain.arun(analysis=analysis_result, scores=scores_result)
            logger.info(f"[Suggestions Output]: {suggestions_result}")

            def safe_json_parse(text, fallback):
                try:
                    if text.strip().startswith("```"):
                        text = text.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
                    return json.loads(text)
                except Exception as e:
                    logger.warning(f"JSON parsing failed: {e}. Text: {text}")
                    return fallback

            scores_data = safe_json_parse(scores_result, {})
            
            if suggestions_result:
                clean_json = re.sub(r"^```json|```$", "", suggestions_result.strip(), flags=re.MULTILINE)
                suggestions_data = json.loads(clean_json)
            else:
                suggestions_data = []

            return {
                "analysis": analysis_result,
                "scores": scores_data,
                "suggestions": suggestions_data
            }
        except Exception as e:
            logger.error(f"Resume processing error: {e}")
            raise HTTPException(status_code=500, detail="Resume processing failed")


resume_processor = LangChainResumeProcessor()


def extract_text_from_pdf(file_content: bytes) -> str:
    try:
        pdf_file = io.BytesIO(file_content)
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        return "".join([page.extract_text() for page in pdf_reader.pages])
    except Exception as e:
        logger.error(f"PDF extraction error: {e}")
        return ""
"""
# Background task
async def process_resume_background(analysis_id: int, resume_text: str, job_description: Optional[str], is_premium: bool):
    db = SessionLocal()
    try:
        analysis = db.query(ResumeAnalysis).filter(ResumeAnalysis.id == analysis_id).first()
        analysis.processing_status = "processing"
        db.commit()
        
        result = await resume_processor.process_resume(resume_text, job_description, is_premium)
        
        resume_scores = result.get("scores", {}).get("resumeScores", {})
        
        analysis.overall_score = resume_scores.get("overallScore", 0)
        analysis.ats_score = resume_scores.get("atsScore", 0)
        analysis.job_match_score = resume_scores.get("jobMatchScore") if job_description else None
        analysis.category_scores = json.dumps(result["scores"].get("category_scores", {}))
        analysis.suggestions = json.dumps(result["suggestions"])
        analysis.processing_status = "completed"
        
        db.commit()
    except Exception as e:
        logger.error(f"Background processing error: {e}")
        analysis = db.query(ResumeAnalysis).filter(ResumeAnalysis.id == analysis_id).first()
        if analysis:
            analysis.processing_status = "failed"
            db.commit()
    finally:
        db.close()    

# Background task
"""
# ✅ Corrected Background Task
# ✅ Corrected and Scaled Background Task
# ✅ Final Version – Scaled and Robust Background Task
async def process_resume_background(
    analysis_id: int,
    resume_text: str,
    job_description: Optional[str],
    is_premium: bool
):
    db = SessionLocal()
    try:
        # Fetch analysis record
        analysis = db.query(ResumeAnalysis).filter(ResumeAnalysis.id == analysis_id).first()
        if not analysis:
            logger.error(f"❌ Analysis ID {analysis_id} not found.")
            return

        analysis.processing_status = "processing"
        db.commit()

        # Run Gemini Processor
        result = await resume_processor.process_resume(resume_text, job_description, is_premium)
        logger.info(f"🧠 RESULT RECEIVED FROM PROCESSOR: {result}")

        scores_data = result.get("scores", {})

        # ✅ Normalization helper (convert 0–10 → 0–100)
        def normalize(value: Any) -> float:
            try:
                val = float(value)
                return round(val * 10, 1) if val <= 10 else round(val, 1)
            except (TypeError, ValueError):
                return 0.0

        # ✅ Extract key scores safely
        overall = normalize(
            scores_data.get("resume_score")
            or scores_data.get("overall_score")
            or scores_data.get("overallScore")
            or 0
        )
        ats = normalize(
            scores_data.get("ats_score")
            or scores_data.get("accuracy_score")
            or scores_data.get("atsScore")
            or overall * 0.9  # fallback heuristic
        )
        job_match = normalize(
            scores_data.get("job_match_score")
            or scores_data.get("experience_relevance_score")
            or scores_data.get("alignment_score")
            or overall * 0.8  # fallback heuristic
        )

        # ✅ Dynamically detect numeric sub-scores for category averaging
        category_map = {
            "formatting": ["formatting_score", "accuracy_score", "clarity_score"],
            "content": ["project_details_score", "summary_score", "content_quality"],
            "keywords": ["keyword_relevance", "quantification_score", "ats_keyword_match"],
            "experience": ["experience_relevance_score", "impact_score", "timeline_consistency"],
            "skills": ["skills_relevance_score", "certification_strength", "tech_stack_score"],
        }

        def average(keys):
            vals = [normalize(scores_data.get(k, 0)) for k in keys if k in scores_data]
            return round(sum(vals) / len(vals), 1) if vals else 0.0

        category_scores = {cat: average(keys) for cat, keys in category_map.items()}
        analysis.category_scores = json.dumps(category_scores)

        # ✅ Weighted overall score from categories (more realistic)
        weights = {"formatting": 0.15, "content": 0.25, "keywords": 0.2, "experience": 0.25, "skills": 0.15}
        weighted_overall = round(sum(category_scores[k] * w for k, w in weights.items()), 1)
        if weighted_overall > 0:
            overall = weighted_overall

        # ✅ Save normalized & weighted scores
        analysis.overall_score = overall
        analysis.ats_score = ats
        analysis.job_match_score = job_match

        # ✅ Normalize suggestions
        raw_suggestions = result.get("suggestions", [])
        normalized_suggestions = [
            {
                "area": s.get("area", "General"),
                "description": (
                    s.get("reasoning")
                    or s.get("reason")
                    or s.get("rationale")
                    or "No description provided"
                ),
                "action": (
                    s.get("improvement")
                    or s.get("recommendation")
                    or "No action provided"
                ),
            }
            for s in raw_suggestions
        ]
        analysis.suggestions = json.dumps(normalized_suggestions)

        # ✅ Add Premium Insights if applicable
        if is_premium:
            analysis.premium_insights = json.dumps([
                "Quantify achievements with metrics.",
                "Add leadership and impact examples.",
                "Highlight key skills that match the job posting."
            ])

        analysis.processing_status = "completed"
        db.commit()

        logger.info(
            f"✅ Analysis {analysis_id} saved | overall={overall} ats={ats} job_match={job_match} | categories={category_scores}"
        )

    except Exception as e:
        logger.error(f"❌ Error in process_resume_background: {e}")
        if "analysis" in locals():
            analysis.processing_status = "failed"
            db.commit()
    finally:
        db.close()




# ============= AUTHENTICATION ENDPOINTS =============

@app.post("/auth/register", response_model=UserResponse)
async def register(user_data: UserCreate, db: Session = Depends(get_db)):
    existing_user = db.query(User).filter(User.email == user_data.email).first()
    if existing_user:
        raise HTTPException(status_code=400, detail="Email already registered")
    
    user = User(
        email=user_data.email,
        hashed_password=hash_password(user_data.password),
        full_name=user_data.full_name
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return UserResponse(
        id=user.id, email=user.email, full_name=user.full_name,
        is_premium=user.is_premium, credits_remaining=user.credits_remaining
    )


@app.post("/auth/login")
async def login(login_data: UserLogin, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == login_data.email).first()
    if not user or not verify_password(login_data.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    token = create_access_token(user.id)
    return {"access_token": token, "token_type": "bearer", "user": UserResponse(
        id=user.id, email=user.email, full_name=user.full_name,
        is_premium=user.is_premium, credits_remaining=user.credits_remaining
    )}


@app.get("/auth/me", response_model=UserResponse)
async def get_current_user_info(user_id: int = Depends(verify_token), db: Session = Depends(get_db)):
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return UserResponse(
        id=user.id, email=user.email, full_name=user.full_name,
        is_premium=user.is_premium, credits_remaining=user.credits_remaining
    )


# ============= RESUME ANALYSIS ENDPOINTS =============

@app.post("/resume/analyze", response_model=AnalysisResponse)
async def analyze_resume(
    background_tasks: BackgroundTasks, 
    file: UploadFile = File(...),
    job_description: Optional[str] = None,
    user_id: int = Depends(verify_token), 
    db: Session = Depends(get_db)
):
    user = db.query(User).filter(User.id == user_id).first()
    
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Only PDF supported")
    
    file_content = await file.read()
    resume_text = extract_text_from_pdf(file_content)
    if not resume_text.strip():
        raise HTTPException(status_code=400, detail="Empty PDF")
    
    analysis = ResumeAnalysis(
        user_id=user_id, 
        filename=file.filename,
        job_description=job_description
    )
    db.add(analysis)
    db.commit()
    db.refresh(analysis)
    
    if not user.is_premium:
        user.credits_remaining += 1
        db.commit()
    
    background_tasks.add_task(process_resume_background, analysis.id, resume_text, job_description, user.is_premium)
    
    return AnalysisResponse(id=analysis.id, processing_status="pending")

''''''
@app.get("/resume/analysis/{analysis_id}", response_model=AnalysisResponse)
async def get_analysis_result(
    analysis_id: int,
    user_id: int = Depends(verify_token),
    db: Session = Depends(get_db)
):
    analysis = db.query(ResumeAnalysis).filter(
        ResumeAnalysis.id == analysis_id,
        ResumeAnalysis.user_id == user_id
    ).first()

    if not analysis:
        raise HTTPException(status_code=404, detail="Not found")

    result = AnalysisResponse(id=analysis.id, processing_status=analysis.processing_status)

    if analysis.processing_status == "completed":
        result.overall_score = analysis.overall_score
        result.ats_score = analysis.ats_score
        result.job_match_score = analysis.job_match_score
        result.category_scores = json.loads(analysis.category_scores) if analysis.category_scores else {}

        # ✅ Parse suggestions safely
        parsed_suggestions = json.loads(analysis.suggestions) if analysis.suggestions else []
        normalized_suggestions = [
            {
                "area": item.get("area", "General"),
                "description": item.get("description", "No description provided"),
                "action": item.get("action", "No action provided")
            }
            for item in parsed_suggestions
        ]
        result.suggestions = normalized_suggestions

        if analysis.premium_insights:
            result.premium_insights = json.loads(analysis.premium_insights)

    logger.info(f"RESULT {result}")
    return result




# ============= PAYMENT ENDPOINTS (RAZORPAY UPI) =============

@app.post("/payment/create-order", response_model=CreateOrderResponse)
async def create_payment_order(
    request: CreateOrderRequest,
    user_id: int = Depends(verify_token),
    db: Session = Depends(get_db),

):
    """Create a Razorpay order for subscription payment"""
    try:
        
        print("hI WILL PRITN API KEYS",os.getenv("RAZORPAY_KEY_ID"))
        print(os.getenv("RAZORPAY_KEY_SECRET"))
        # print("📩 Received JSON:", body)

        print("🧠 user_id from token:", user_id)
        # Validate plan
        if request.plan_name not in PLAN_CONFIGS:
            raise HTTPException(status_code=400, detail="Invalid plan name")
        
        user = db.query(User).filter(User.id == user_id).first()
        if not user:
            raise HTTPException(status_code=404, detail="User not found")

        
        # Create Razorpay order
        order_data = {
            "amount": int(request.amount)*100,  # Amount in paise
            "currency": request.currency,
            "payment_capture": 1,
            "receipt": f"receipt_{uuid.uuid4().hex[:10]}",   
            "notes": {
                "user_id": user.id,
                "plan_name": request.plan_name,
                "email": user.email
            }
        }
        
        razorpay_order = razorpay_client.order.create(data=order_data)
        
        # Save payment record in database
        payment = Payment(
            user_id=user.id,
            razorpay_order_id=razorpay_order["id"],
            amount=request.amount,
            currency="INR",
            plan_name=request.plan_name,
            status="pending"
        )
        db.add(payment)
        db.commit()
        
        logger.info(f"Order created: {razorpay_order['id']} for user {user.email}")
        
        return CreateOrderResponse(
            order_id=razorpay_order["id"],
            amount=razorpay_order["amount"],
            currency=razorpay_order["currency"]
        )
        
    except Exception as e:
        logger.error(f"Failed to create order: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to create order: {str(e)}")


@app.post("/payment/verify", response_model=VerifyPaymentResponse)
async def verify_payment(
    request: VerifyPaymentRequest,
    user_id: int = Depends(verify_token),
    db: Session = Depends(get_db)
):
    """Verify Razorpay payment and activate subscription"""
    try:
        # Get payment record
        payment = db.query(Payment).filter(
            Payment.razorpay_order_id == request.razorpay_order_id,
            Payment.user_id == user_id
        ).first()
        
        if not payment:
            raise HTTPException(status_code=404, detail="Payment record not found")
        
        # Verify signature
        generated_signature = hmac.new(
            RAZORPAY_KEY_SECRET.encode(),
            f"{request.razorpay_order_id}|{request.razorpay_payment_id}".encode(),
            hashlib.sha256
        ).hexdigest()
        
        if generated_signature != request.razorpay_signature:
            payment.status = "failed"
            db.commit()
            logger.warning(f"Invalid signature for order {request.razorpay_order_id}")
            raise HTTPException(status_code=400, detail="Invalid payment signature")
        
        # Payment verified successfully
        payment.razorpay_payment_id = request.razorpay_payment_id
        payment.razorpay_signature = request.razorpay_signature
        payment.status = "completed"
        
        # Get plan configuration
        plan_config = PLAN_CONFIGS.get(request.plan_name)
        if not plan_config:
            raise HTTPException(status_code=400, detail="Invalid plan")
        
        # Get user
        user = db.query(User).filter(User.id == user_id).first()
        
        # Update user to premium
        user.is_premium = True
        user.credits_remaining = plan_config["credits"]
        
        # Create subscription record
        subscription = Subscription(
            user_id=user.id,
            plan_name=request.plan_name,
            status="active",
            start_date=datetime.utcnow(),
            end_date=datetime.utcnow() + timedelta(days=plan_config["duration_days"]),
            auto_renew=True
        )
        db.add(subscription)
        
        db.commit()
        db.refresh(subscription)
        
        logger.info(f"Payment verified for user {user.email}, plan: {request.plan_name}")
        
        return VerifyPaymentResponse(
            success=True,
            message="Payment verified and subscription activated successfully",
            subscription_details={
                "plan_name": subscription.plan_name,
                "start_date": subscription.start_date.isoformat(),
                "end_date": subscription.end_date.isoformat() if subscription.end_date else None,
                "credits": user.credits_remaining
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Payment verification failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Payment verification failed: {str(e)}")


@app.get("/payment/history")
async def get_payment_history(
    user_id: int = Depends(verify_token),
    db: Session = Depends(get_db)
):
    """Get user's payment history"""
    payments = db.query(Payment).filter(
        Payment.user_id == user_id
    ).order_by(Payment.created_at.desc()).all()
    
    return {"payments": [
        {
            "id": p.id,
            "order_id": p.razorpay_order_id,
            "payment_id": p.razorpay_payment_id,
            "amount": p.amount / 100,  # Convert paise to rupees
            "currency": p.currency,
            "plan_name": p.plan_name,
            "status": p.status,
            "created_at": p.created_at.isoformat()
        }
        for p in payments
    ]}


@app.get("/subscription/current")
async def get_current_subscription(
    user_id: int = Depends(verify_token),
    db: Session = Depends(get_db)
):
    """Get user's current active subscription"""
    subscription = db.query(Subscription).filter(
        Subscription.user_id == user_id,
        Subscription.status == "active"
    ).order_by(Subscription.created_at.desc()).first()
    
    if not subscription:
        return {"message": "No active subscription"}
    
    return {
        "id": subscription.id,
        "plan_name": subscription.plan_name,
        "status": subscription.status,
        "start_date": subscription.start_date.isoformat(),
        "end_date": subscription.end_date.isoformat() if subscription.end_date else None,
        "auto_renew": subscription.auto_renew
    }


# ============= HEALTH CHECK =============

@app.get("/health")
async def health_check():
    return {
        "status": "healthy", 
        "ai_agents": "initialized",
        "razorpay": "configured"
    }


@app.get("/")
async def root():
    return {
        "message": "ResumeAI API with Razorpay UPI Integration",
        "version": "2.0.0",
        "endpoints": {
            "auth": "/auth/register, /auth/login, /auth/me",
            "resume": "/resume/analyze, /resume/analysis/{id}",
            "payment": "/payment/create-order, /payment/verify, /payment/history",
            "subscription": "/subscription/current"
        }
    }


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)