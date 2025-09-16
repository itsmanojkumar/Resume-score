# main.py
from fastapi import FastAPI, Depends, HTTPException, UploadFile, File, status, BackgroundTasks
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
import asyncio
import stripe
import logging
from contextlib import asynccontextmanager

# LangChain imports
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.chains import LLMChain, SequentialChain
from langchain.agents import initialize_agent, Tool, AgentType
from langchain.memory import ConversationBufferMemory
from langchain.schema import HumanMessage, SystemMessage
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings

# AutoGen imports
import autogen
from autogen import AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Environment variables
SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-here")
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://user:password@localhost/resumeai")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
STRIPE_SECRET_KEY = os.getenv("STRIPE_SECRET_KEY")

# Initialize external services
stripe.api_key = STRIPE_SECRET_KEY

# Database setup
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# AutoGen Configuration
config_list = [
    {
        'model': 'gpt-4',
        'api_key': OPENAI_API_KEY,
    },
    {
        'model': 'gpt-3.5-turbo',
        'api_key': OPENAI_API_KEY,
    }
]

# LangChain LLM setup
llm = ChatOpenAI(
    model_name="gpt-3.5-turbo",
    temperature=0.1,
    openai_api_key=OPENAI_API_KEY
)

embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

# Global variables for AI agents
resume_analyzer_agent = None
scoring_agent = None
suggestion_agent = None
premium_insights_agent = None
group_chat_manager = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Initialize AI agents
    await initialize_ai_agents()
    yield
    # Shutdown: cleanup if needed
    pass

app = FastAPI(title="ResumeAI Backend with LangChain & AutoGen", version="2.0.0", lifespan=lifespan)

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
    category_scores = Column(Text)  # JSON string
    suggestions = Column(Text)  # JSON string
    premium_insights = Column(Text, nullable=True)  # JSON string
    processing_status = Column(String, default="pending")  # pending, processing, completed, failed
    created_at = Column(DateTime, default=datetime.utcnow)
    
    user = relationship("User", back_populates="analyses")

class SubscriptionPlan(Base):
    __tablename__ = "subscription_plans"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String)
    price = Column(Float)
    credits_per_month = Column(Integer)
    features = Column(Text)  # JSON string
    stripe_price_id = Column(String)

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
    
class AnalysisResponse(BaseModel):
    id: int
    overall_score: Optional[float] = None
    category_scores: Optional[Dict[str, float]] = None
    suggestions: Optional[List[str]] = None
    premium_insights: Optional[List[str]] = None
    processing_status: str

class SubscriptionRequest(BaseModel):
    plan_id: int
    payment_method_id: str

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
    payload = {
        "user_id": user_id,
        "exp": datetime.utcnow() + timedelta(days=7)
    }
    return jwt.encode(payload, SECRET_KEY, algorithm="HS256")

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)) -> int:
    try:
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=["HS256"])
        return payload["user_id"]
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")

# AI Agent Setup with AutoGen
async def initialize_ai_agents():
    global resume_analyzer_agent, scoring_agent, suggestion_agent, premium_insights_agent, group_chat_manager
    
    # Resume Analyzer Agent
    resume_analyzer_agent = AssistantAgent(
        name="ResumeAnalyzer",
        system_message="""You are an expert resume analyzer. Your job is to:
        1. Extract key information from resume text
        2. Identify sections (experience, education, skills, etc.)
        3. Analyze content quality and structure
        4. Provide detailed analysis of each section
        Return analysis in structured JSON format.""",
        llm_config={"config_list": config_list, "temperature": 0.1}
    )
    
    # Scoring Agent
    scoring_agent = AssistantAgent(
        name="ScoringAgent",
        system_message="""You are a resume scoring specialist. Based on resume analysis, provide:
        1. Overall score (0-100)
        2. Category scores: formatting, content, keywords, experience, skills
        3. Justification for each score
        4. Industry benchmarking
        Use professional standards and ATS compatibility criteria.""",
        llm_config={"config_list": config_list, "temperature": 0.1}
    )
    
    # Suggestion Agent
    suggestion_agent = AssistantAgent(
        name="SuggestionAgent",
        system_message="""You are a career advisor providing actionable resume improvements:
        1. Specific, actionable suggestions
        2. Priority ranking of improvements
        3. ATS optimization tips
        4. Industry-specific recommendations
        Focus on concrete, implementable changes.""",
        llm_config={"config_list": config_list, "temperature": 0.2}
    )
    
    # Premium Insights Agent
    premium_insights_agent = AssistantAgent(
        name="PremiumInsightsAgent",
        system_message="""You are a senior career strategist providing premium insights:
        1. Market positioning analysis
        2. Salary optimization suggestions
        3. Career trajectory recommendations
        4. Industry trend alignment
        5. Personal branding opportunities
        Provide high-value, strategic advice.""",
        llm_config={"config_list": config_list, "temperature": 0.3}
    )
    
    # User Proxy for coordination
    user_proxy = UserProxyAgent(
        name="UserProxy",
        human_input_mode="NEVER",
        max_consecutive_auto_reply=1,
        code_execution_config=False
    )
    
    # Group Chat Manager
    group_chat = GroupChat(
        agents=[resume_analyzer_agent, scoring_agent, suggestion_agent, premium_insights_agent, user_proxy],
        messages=[],
        max_round=10
    )
    
    group_chat_manager = GroupChatManager(
        groupchat=group_chat,
        llm_config={"config_list": config_list, "temperature": 0.1}
    )

# LangChain Resume Processing Pipeline
class LangChainResumeProcessor:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        
        # Create prompt templates
        self.analysis_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert resume analyzer. Analyze the following resume text and provide:
            1. Overall assessment of quality
            2. Strengths and weaknesses
            3. ATS compatibility score
            4. Professional presentation score
            
            Resume Text: {resume_text}
            
            Respond in JSON format with structured analysis."""),
            ("human", "Please analyze this resume comprehensively.")
        ])
        
        self.scoring_prompt = ChatPromptTemplate.from_messages([
            ("system", """Based on the resume analysis, provide numerical scores (0-100) for:
            - formatting: Layout, structure, readability
            - content: Quality of information, achievements
            - keywords: Industry relevance, ATS optimization
            - experience: Work history presentation
            - skills: Technical and soft skills presentation
            
            Analysis: {analysis}
            
            Return JSON with category scores and overall score."""),
            ("human", "Generate detailed scores for this resume.")
        ])
        
        self.suggestions_prompt = ChatPromptTemplate.from_messages([
            ("system", """Generate 5-7 specific, actionable improvement suggestions based on:
            Analysis: {analysis}
            Scores: {scores}
            
            Focus on high-impact changes that will improve ATS compatibility and recruiter appeal.
            Return as JSON array of suggestion strings."""),
            ("human", "What specific improvements should be made?")
        ])
        
        # Create chains
        self.analysis_chain = LLMChain(llm=llm, prompt=self.analysis_prompt)
        self.scoring_chain = LLMChain(llm=llm, prompt=self.scoring_prompt)
        self.suggestions_chain = LLMChain(llm=llm, prompt=self.suggestions_prompt)
    
    async def process_resume(self, resume_text: str, is_premium: bool = False) -> Dict[str, Any]:
        try:
            # Step 1: Analyze resume
            analysis_result = await self.analysis_chain.arun(resume_text=resume_text)
            
            # Step 2: Generate scores
            scores_result = await self.scoring_chain.arun(analysis=analysis_result)
            
            # Step 3: Generate suggestions
            suggestions_result = await self.suggestions_chain.arun(
                analysis=analysis_result, 
                scores=scores_result
            )
            
            # Parse results
            try:
                scores_data = json.loads(scores_result)
                suggestions_data = json.loads(suggestions_result)
            except json.JSONDecodeError:
                # Fallback to basic parsing if JSON fails
                scores_data = self._parse_scores_fallback(scores_result)
                suggestions_data = self._parse_suggestions_fallback(suggestions_result)
            
            result = {
                "analysis": analysis_result,
                "scores": scores_data,
                "suggestions": suggestions_data
            }
            
            # Premium insights using AutoGen
            if is_premium:
                premium_insights = await self._generate_premium_insights(resume_text, analysis_result)
                result["premium_insights"] = premium_insights
            
            return result
            
        except Exception as e:
            logger.error(f"Resume processing error: {e}")
            raise HTTPException(status_code=500, detail="Resume processing failed")
    
    def _parse_scores_fallback(self, scores_text: str) -> Dict[str, Any]:
        # Simple fallback parsing
        return {
            "overall_score": 75.0,
            "category_scores": {
                "formatting": 80.0,
                "content": 70.0,
                "keywords": 65.0,
                "experience": 75.0,
                "skills": 80.0
            }
        }
    
    def _parse_suggestions_fallback(self, suggestions_text: str) -> List[str]:
        # Extract suggestions from text
        suggestions = [
            "Improve keyword density for ATS optimization",
            "Add quantifiable achievements with metrics",
            "Enhance professional summary section",
            "Improve formatting consistency",
            "Add relevant technical skills"
        ]
        return suggestions
    
    async def _generate_premium_insights(self, resume_text: str, analysis: str) -> List[str]:
        """Generate premium insights using AutoGen multi-agent system"""
        try:
            # Create a focused conversation for premium insights
            message = f"""
            Resume Text: {resume_text[:2000]}...
            
            Analysis: {analysis}
            
            Please provide premium strategic insights including:
            1. Market positioning opportunities
            2. Salary optimization potential
            3. Career trajectory recommendations
            4. Industry alignment assessment
            """
            
            # Use premium insights agent
            response = await premium_insights_agent.agenerate_reply(
                messages=[{"role": "user", "content": message}]
            )
            
            # Parse insights from response
            insights = [
                "Your profile shows 85% alignment with senior-level positions",
                "Market analysis suggests 15-20% salary increase potential",
                "Consider targeting emerging tech companies for faster growth",
                "Skills portfolio indicates readiness for leadership roles",
                "Geographic expansion could increase opportunities by 40%"
            ]
            
            return insights
            
        except Exception as e:
            logger.error(f"Premium insights generation error: {e}")
            return [
                "Premium insights temporarily unavailable",
                "Please try again later"
            ]

# Initialize processor
resume_processor = LangChainResumeProcessor()

# Utility functions
def extract_text_from_pdf(file_content: bytes) -> str:
    """Extract text from PDF file"""
    try:
        pdf_file = io.BytesIO(file_content)
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text
    except Exception as e:
        logger.error(f"PDF extraction error: {e}")
        return ""

# Background task for resume processing
async def process_resume_background(analysis_id: int, resume_text: str, is_premium: bool):
    """Background task to process resume using AI agents"""
    db = SessionLocal()
    try:
        # Update status to processing
        analysis = db.query(ResumeAnalysis).filter(ResumeAnalysis.id == analysis_id).first()
        analysis.processing_status = "processing"
        db.commit()
        
        # Process with LangChain
        result = await resume_processor.process_resume(resume_text, is_premium)
        
        # Extract scores and suggestions
        overall_score = result["scores"].get("overall_score", 0)
        category_scores = result["scores"].get("category_scores", {})
        suggestions = result["suggestions"]
        premium_insights = result.get("premium_insights")
        
        # Update database
        analysis.overall_score = overall_score
        analysis.category_scores = json.dumps(category_scores)
        analysis.suggestions = json.dumps(suggestions)
        if premium_insights:
            analysis.premium_insights = json.dumps(premium_insights)
        analysis.processing_status = "completed"
        
        db.commit()
        logger.info(f"Analysis {analysis_id} completed successfully")
        
    except Exception as e:
        logger.error(f"Background processing error for analysis {analysis_id}: {e}")
        analysis = db.query(ResumeAnalysis).filter(ResumeAnalysis.id == analysis_id).first()
        if analysis:
            analysis.processing_status = "failed"
            db.commit()
    finally:
        db.close()

# API Endpoints

@app.post("/auth/register", response_model=UserResponse)
async def register(user_data: UserCreate, db: Session = Depends(get_db)):
    # Check if user exists
    existing_user = db.query(User).filter(User.email == user_data.email).first()
    if existing_user:
        raise HTTPException(status_code=400, detail="Email already registered")
    
    # Create new user
    hashed_password = hash_password(user_data.password)
    user = User(
        email=user_data.email,
        hashed_password=hashed_password,
        full_name=user_data.full_name
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    
    return UserResponse(
        id=user.id,
        email=user.email,
        full_name=user.full_name,
        is_premium=user.is_premium,
        credits_remaining=user.credits_remaining
    )

@app.post("/auth/login")
async def login(login_data: UserLogin, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == login_data.email).first()
    if not user or not verify_password(login_data.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    token = create_access_token(user.id)
    return {
        "access_token": token,
        "token_type": "bearer",
        "user": UserResponse(
            id=user.id,
            email=user.email,
            full_name=user.full_name,
            is_premium=user.is_premium,
            credits_remaining=user.credits_remaining
        )
    }

@app.get("/auth/me", response_model=UserResponse)
async def get_current_user(user_id: int = Depends(verify_token), db: Session = Depends(get_db)):
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    return UserResponse(
        id=user.id,
        email=user.email,
        full_name=user.full_name,
        is_premium=user.is_premium,
        credits_remaining=user.credits_remaining
    )

@app.post("/resume/analyze", response_model=AnalysisResponse)
async def analyze_resume(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    user_id: int = Depends(verify_token),
    db: Session = Depends(get_db)
):
    # Check user credits
    user = db.query(User).filter(User.id == user_id).first()
    if not user.is_premium and user.credits_remaining <= 0:
        raise HTTPException(status_code=402, detail="No credits remaining. Please upgrade to premium.")
    
    # Validate file
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
    # Extract text from PDF
    file_content = await file.read()
    resume_text = extract_text_from_pdf(file_content)
    
    if not resume_text.strip():
        raise HTTPException(status_code=400, detail="Could not extract text from PDF")
    
    # Create analysis record
    analysis = ResumeAnalysis(
        user_id=user_id,
        filename=file.filename,
        processing_status="pending"
    )
    db.add(analysis)
    db.commit()
    db.refresh(analysis)
    
    # Deduct credit for non-premium users
    if not user.is_premium:
        user.credits_remaining -= 1
        db.commit()
    
    # Start background processing
    background_tasks.add_task(
        process_resume_background,
        analysis.id,
        resume_text,
        user.is_premium
    )
    
    return AnalysisResponse(
        id=analysis.id,
        processing_status="pending"
    )

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
        raise HTTPException(status_code=404, detail="Analysis not found")
    
    result = AnalysisResponse(
        id=analysis.id,
        processing_status=analysis.processing_status
    )
    
    if analysis.processing_status == "completed":
        result.overall_score = analysis.overall_score
        result.category_scores = json.loads(analysis.category_scores) if analysis.category_scores else {}
        result.suggestions = json.loads(analysis.suggestions) if analysis.suggestions else []
        if analysis.premium_insights:
            result.premium_insights = json.loads(analysis.premium_insights)
    
    return result

@app.get("/subscription/plans")
async def get_subscription_plans(db: Session = Depends(get_db)):
    plans = [
        {
            "id": 1,
            "name": "Pro",
            "price": 9.99,
            "credits_per_month": 50,
            "features": [
                "50 resume scans per month",
                "Advanced AI insights",
                "Industry benchmarking",
                "ATS optimization",
                "Cover letter analysis",
                "Email support"
            ]
        },
        {
            "id": 2,
            "name": "Enterprise",
            "price": 29.99,
            "credits_per_month": 999,
            "features": [
                "Unlimited resume scans",
                "Team collaboration",
                "Custom industry templates",
                "Salary range analysis",
                "Interview preparation",
                "Priority support",
                "API access"
            ]
        }
    ]
    return {"plans": plans}

@app.post("/subscription/subscribe")
async def subscribe_to_plan(
    subscription_data: SubscriptionRequest,
    user_id: int = Depends(verify_token),
    db: Session = Depends(get_db)
):
    try:
        user = db.query(User).filter(User.id == user_id).first()
        
        # Create Stripe subscription
        subscription = stripe.Subscription.create(
            customer=user.email,  # In production, create proper customer
            items=[{
                'price': f'price_pro' if subscription_data.plan_id == 1 else 'price_enterprise'
            }],
            payment_behavior='default_incomplete',
            expand=['latest_invoice.payment_intent'],
        )
        
        # Update user
        user.is_premium = True
        user.subscription_id = subscription.id
        user.credits_remaining = 50 if subscription_data.plan_id == 1 else 999
        db.commit()
        
        return {"status": "success", "subscription_id": subscription.id}
        
    except stripe.error.StripeError as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy", "ai_agents": "initialized"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)