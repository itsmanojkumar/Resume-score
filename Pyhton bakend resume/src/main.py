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
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:Postgresql@localhost/postgres")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
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
    model="gemini-1.5-flash",
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
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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
    credits_remaining = Column(Integer, default=2)
    subscription_id = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    analyses = relationship("ResumeAnalysis", back_populates="user")


class ResumeAnalysis(Base):
    __tablename__ = "resume_analyses"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    filename = Column(String)
    overall_score = Column(Float)
    category_scores = Column(Text)
    suggestions = Column(Text)
    premium_insights = Column(Text, nullable=True)
    processing_status = Column(String, default="pending")
    created_at = Column(DateTime, default=datetime.utcnow)
    
    user = relationship("User", back_populates="analyses")


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

""" class SuggestionItem(BaseModel):
    area: str
    description: str
    priority: str  """


class AnalysisResponse(BaseModel):
    id: int
    overall_score: Optional[float] = None
    category_scores: Optional[Dict[str, float]] = None
    # suggestions: Optional[List[str]] = None
    suggestions: Optional[List[str]] = None  

    premium_insights: Optional[List[str]] = None
    processing_status: str


class SubscriptionRequest(BaseModel):
    plan_id: int


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
'''class LangChainResumeProcessor:
    def __init__(self):
        self.analysis_prompt = ChatPromptTemplate.from_messages([
            ("system", "Analyze the following resume text: {resume_text}. Return JSON."),
            ("human", "Please analyze this resume.")
        ])
        
        self.scoring_prompt = ChatPromptTemplate.from_messages([
            ("system", "Provide numerical scores based on analysis: {analysis}. Return JSON."),
            ("human", "Generate detailed scores.")
        ])
        
        self.suggestions_prompt = ChatPromptTemplate.from_messages([
            ("system", "Generate 5-7 improvement suggestions. Analysis: {analysis}, Scores: {scores}. Return JSON."),
            ("human", "What improvements should be made?")
        ])
        
        self.analysis_chain = LLMChain(llm=llm, prompt=self.analysis_prompt)
        self.scoring_chain = LLMChain(llm=llm, prompt=self.scoring_prompt)
        self.suggestions_chain = LLMChain(llm=llm, prompt=self.suggestions_prompt)
    
    async def process_resume(self, resume_text: str, is_premium: bool = False) -> Dict[str, Any]:
        try:
            analysis_result = await self.analysis_chain.arun(resume_text=resume_text)
            scores_result = await self.scoring_chain.arun(analysis=analysis_result)
            suggestions_result = await self.suggestions_chain.arun(analysis=analysis_result, scores=scores_result)
            
            scores_data = json.loads(scores_result) if scores_result else {}
            suggestions_data = json.loads(suggestions_result) if suggestions_result else []
            
            return {
                "analysis": analysis_result,
                "scores": scores_data,
                "suggestions": suggestions_data
            }
        except Exception as e:
            logger.error(f"Resume processing error: {e}")
            raise HTTPException(status_code=500, detail="Resume processing failed")   '''
        


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
    
    async def process_resume(self, resume_text: str, is_premium: bool = False) -> Dict[str, Any]:
        try:
            analysis_result = await self.analysis_chain.arun(resume_text=resume_text)
            logger.info(f"[Analysis Output]: {analysis_result}")
            print("[DEBUG] Analysis Result:", analysis_result)

            scores_result = await self.scoring_chain.arun(analysis=analysis_result)
            logger.info(f"[Scores Output]: {scores_result}")
            print("[DEBUG] Scores Result:", scores_result)
            logger.debug(f"scores_result raw: {repr(scores_result)}")


            suggestions_result = await self.suggestions_chain.arun(analysis=analysis_result, scores=scores_result)
            logger.info(f"[Suggestions Output]: {suggestions_result}")

            '''def safe_json_parse(text, fallback):
                try:
                    return json.loads(text)
                except Exception as e:
                    logger.warning(f"JSON parsing failed: {e}. Text: {text}")
                    return fallback        '''
                

            def safe_json_parse(text, fallback):
                try:
                 # Remove Markdown-style ```json ... ``` if present
                    if text.strip().startswith("```"):
                        text = text.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
                    return json.loads(text)
                except Exception as e:
                    logger.warning(f"JSON parsing failed: {e}. Text: {text}")
                    return fallback

            scores_data = safe_json_parse(scores_result, {})
            # suggestions_data = safe_json_parse(suggestions_result, [])

            if suggestions_result:
            # Remove markdown-style code block markers
                clean_json = re.sub(r"^```json|```$", "", suggestions_result.strip(), flags=re.MULTILINE)
                suggestions_data = json.loads(clean_json)
            else:
                suggestions_data = []
            print("suggestions_data",suggestions_data)
            print("type of suggestions_data",type(suggestions_data))
            # scores_data = json.loads(scores_result) if scores_result else {}
            # suggestions_data = json.loads(suggestions_result) if suggestions_result else []

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
    print("[DEBUG] extract_text_from_pdf called")
    try:
        pdf_file = io.BytesIO(file_content)
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        print(f"Number of pages in PDF: {len(pdf_reader.pages)}")
        return "".join([page.extract_text() for page in pdf_reader.pages])
    except Exception as e:
        logger.error(f"PDF extraction error: {e}")
        return ""


# Background task
async def process_resume_background(analysis_id: int, resume_text: str, is_premium: bool):
    db = SessionLocal()
    try:
        analysis = db.query(ResumeAnalysis).filter(ResumeAnalysis.id == analysis_id).first()
        analysis.processing_status = "processing"
        db.commit()
        
        result = await resume_processor.process_resume(resume_text, is_premium)
        print("[DEBUG] LangChainResumeProcessor initialized", resume_processor)
        print(result)
        print("tyepe of result",type(result))
        print("analysis all",analysis.overall_score,analysis.category_scores,analysis.suggestions)
        print("scores",result["scores"])

        resume_scores = result.get("scores", {}).get("resumeScores", {})

        # Assign overall score
        analysis.overall_score = resume_scores.get("overallScore", 0)
    
        # analysis.overall_score = result["scores"].get("overall_score", 0)
        print("analysis.overall score",analysis.overall_score)
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


# API Endpoints
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


@app.post("/resume/analyze", response_model=AnalysisResponse)
async def analyze_resume(background_tasks: BackgroundTasks, file: UploadFile = File(...),
                         user_id: int = Depends(verify_token), db: Session = Depends(get_db)):
    print("[DEBUG] analyze_resume endpoint called")
    user = db.query(User).filter(User.id == user_id).first()
    # if not user.is_premium and user.credits_remaining <= 0:
        # raise HTTPException(status_code=402, detail="No credits remaining.")
    
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Only PDF supported")
    
    file_content = await file.read()
    resume_text = extract_text_from_pdf(file_content)
    if not resume_text.strip():
        raise HTTPException(status_code=400, detail="Empty PDF")
    
    analysis = ResumeAnalysis(user_id=user_id, filename=file.filename)
    db.add(analysis)
    db.commit()
    db.refresh(analysis)
    
    if not user.is_premium:
        user.credits_remaining -= 1
        db.commit()
    
    background_tasks.add_task(process_resume_background, analysis.id, resume_text, user.is_premium)
    print("analysis is",analysis)
    # print("analysis all",analysis.overall_score,analysis.category_scores,analysis.suggestions)
    
    return AnalysisResponse(id=analysis.id, processing_status="pending")
    # return AnalysisResponse( processing_status="pending")



@app.get("/resume/analysis/{analysis_id}", response_model=AnalysisResponse)
async def get_analysis_result(analysis_id: int, user_id: int = Depends(verify_token), db: Session = Depends(get_db)):
    analysis = db.query(ResumeAnalysis).filter(ResumeAnalysis.id == analysis_id, ResumeAnalysis.user_id == user_id).first()
    print("db anlysis",analysis)
    if not analysis:
        raise HTTPException(status_code=404, detail="Not found")
    
    result = AnalysisResponse(id=analysis.id, processing_status=analysis.processing_status)
    print(result)
    if analysis.processing_status == "completed":
        print("result2",result)
        result.overall_score = analysis.overall_score
        result.category_scores = json.loads(analysis.category_scores)


        # suggestions_raw = json.loads(analysis.suggestions)
        # if isinstance(suggestions_raw, dict) and "improvements" in suggestions_raw:
            # suggestions_raw = suggestions_raw

        # result.suggestions = [SuggestionItem(**item) for item in suggestions_raw]

        
        parsed_suggestions = json.loads(analysis.suggestions) if analysis.suggestions else {}
        result.suggestions = parsed_suggestions.get("improvements", [])

        # result.suggestions = json.loads(analysis.suggestions) if analysis.suggestions else []
        # suggestions_dict = json.loads(analysis.suggestions)
        # result.suggestions = SuggestionsObject(**suggestions_dict)
        if analysis.premium_insights:
            result.premium_insights = json.loads(analysis.premium_insights)
    print("manojkumarhi",result)
    return result


@app.get("/subscription/plans")
async def get_subscription_plans():
    return {"plans": [
        {"id": 1, "name": "Pro", "price": 999, "credits_per_month": 50},
        {"id": 2, "name": "Enterprise", "price": 2999, "credits_per_month": 999}
    ]}


@app.post("/subscription/subscribe")
async def subscribe_to_plan(subscription_data: SubscriptionRequest,
                            user_id: int = Depends(verify_token), db: Session = Depends(get_db)):
    try:
        user = db.query(User).filter(User.id == user_id).first()
        amount = 999 if subscription_data.plan_id == 1 else 2999
        order = razorpay_client.order.create({"amount": amount, "currency": "INR", "payment_capture": "1"})
        user.subscription_id = order["id"]
        db.commit()
        return {"status": "created", "order_id": order["id"]}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/payment/webhook")
async def razorpay_webhook(request: Request, db: Session = Depends(get_db)):
    body = await request.body()
    signature = request.headers.get("X-Razorpay-Signature")
    try:
        generated_sig = hmac.new(RAZORPAY_KEY_SECRET.encode(), body, hashlib.sha256).hexdigest()
        if generated_sig != signature:
            raise HTTPException(status_code=400, detail="Invalid signature")
        
        payload = await request.json()
        order_id = payload.get("payload", {}).get("payment", {}).get("entity", {}).get("order_id")
        if order_id:
            user = db.query(User).filter(User.subscription_id == order_id).first()
            if user:
                user.is_premium = True
                user.credits_remaining = 50 if "pro" in order_id else 999
                db.commit()
        return {"status": "ok"}
    except Exception as e:
        logger.error(f"Webhook error: {e}")
        raise HTTPException(status_code=400, detail="Webhook handling failed")


@app.get("/health")
async def health_check():
    return {"status": "healthy", "ai_agents": "initialized"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
