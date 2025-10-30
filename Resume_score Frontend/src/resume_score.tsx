import React, { useState, useRef, useEffect, useCallback } from 'react';
import { Upload, Star, CheckCircle, Crown, CreditCard, FileText, BarChart3, Target, Zap, Shield, ArrowRight, Briefcase, Users, TrendingUp, AlertCircle, RefreshCw, LogIn, UserPlus } from 'lucide-react';

// Type definitions
interface User {
  id?: number;
  email?: string;
  full_name?: string;
  isPremium: boolean;
  creditsRemaining: number;
  subscription: SubscriptionPlan | null;
  token: string | null;
  isAuthenticated: boolean;
}


interface SubscriptionPlan {
  id?: number;
  name: string;
  price: string;
  features: string[];
  popular?: boolean;
}

interface CategoryScores {
  formatting: number;
  content: number;
  keywords: number;
  experience: number;
  skills: number;
}

 interface Suggestion {
  area: string;
  description: string;
  action: string;
}

interface AnalysisResult {
  id: number;
  overall_score: number | null;
  ats_score: number | null;
  job_match_score: number | null;
  category_scores: CategoryScores | null;
  // suggestions: string[] | null;

  suggestions: Suggestion[] | null;

 
  premium_insights: string[] | null;
  processing_status: 'pending' | 'processing' | 'completed' | 'failed';
} 


interface AuthForm {
  email: string;
  password: string;
  fullName: string;
}

interface AuthResponse {
  access_token: string;
  token_type: string;
  user: {
    id: number;
    email: string;
    full_name: string;
    is_premium: boolean;
    credits_remaining: number;
  };
}

interface ScoreCircleProps {
  score: number;
  label: string;
  color:string;
}

type TabType = 'upload' | 'results' | 'pricing';
type AuthMode = 'login' | 'register';

const ResumeScoringSoftware: React.FC = () => {
  const [user, setUser] = useState<User>({ 
    isPremium: false, 
    creditsRemaining: 100,
    subscription: null,
    token: localStorage.getItem('token'),
    isAuthenticated: !!localStorage.getItem('token')
  });
  
  const [uploadedFile, setUploadedFile] = useState<File | null>(null);
  const [jobDescription, setJobDescription] = useState<string>('');
  const [analysisId, setAnalysisId] = useState<number | null>(null);
  const [scoreResult, setScoreResult] = useState<AnalysisResult | null>(null);
  const [activeTab, setActiveTab] = useState<TabType>('upload');
  const [showPayment, setShowPayment] = useState<boolean>(false);
  const [showAuth, setShowAuth] = useState<boolean>(true);
  const [authMode, setAuthMode] = useState<AuthMode>('login');
  const [selectedPlan, setSelectedPlan] = useState<SubscriptionPlan | null>(null);
  const [isProcessing, setIsProcessing] = useState<boolean>(false);
  const [processingStatus, setProcessingStatus] = useState<string>('');
  const [authForm, setAuthForm] = useState<AuthForm>({
    email: '',
    password: '',
    fullName: ''
  });
  
  const fileInputRef = useRef<HTMLInputElement>(null);

  // API base URL
  // const API_BASE: string = 'http://localhost:8000';
  // const API_BASE: string = 'https://resume-score-1.onrender.com';
  const API_BASE: string = 'https://resume-score-skth.onrender.com';



  // Load user data on component mount
  useEffect(() => {
    const token = localStorage.getItem('token');
    if (token) {
      fetchUserData(token);
    }
  }, []);    

  const fetchUserData = async (token: string): Promise<void> => {
    try {
      const response = await fetch(`${API_BASE}/auth/me`, {
        headers: {
          'Authorization': `Bearer ${token}`
        }
      });
      
      if (response.ok) {
        const userData = await response.json();
        setUser(prev => ({
          ...prev,
          id: userData.id,
          email: userData.email,
          full_name: userData.full_name,
          isPremium: userData.is_premium,
          creditsRemaining: userData.credits_remaining,
          token,
          isAuthenticated: true
        }));
      } else {
        localStorage.removeItem('token');
        setUser(prev => ({
          ...prev,
          token: null,
          isAuthenticated: false
        }));
      }
    } catch (error) {
      console.error('Error fetching user data:', error);
    }
  };

  // Authentication handlers
  const handleAuth = async (): Promise<void> => {
    try {
      const endpoint = authMode === 'login' ? '/auth/login' : '/auth/register';
      const payload = authMode === 'login' 
        ? { email: authForm.email, password: authForm.password }
        : { email: authForm.email, password: authForm.password, full_name: authForm.fullName };

      const response = await fetch(`${API_BASE}${endpoint}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(payload)
      });

      const data = await response.json();

      if (response.ok) {
        if (authMode === 'login') {
          const authData: AuthResponse = data;
          localStorage.setItem('token', authData.access_token);
          setUser({
            id: authData.user.id,
            email: authData.user.email,
            full_name: authData.user.full_name,
            isPremium: authData.user.is_premium,
            creditsRemaining: authData.user.credits_remaining,
            subscription: null,
            token: authData.access_token,
            isAuthenticated: true
          });
        } else {
          // For registration, auto-login
          const loginResponse = await fetch(`${API_BASE}/auth/login`, {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json'
            },
            body: JSON.stringify({ email: authForm.email, password: authForm.password })
          });
          
          if (loginResponse.ok) {
            const loginData: AuthResponse = await loginResponse.json();
            localStorage.setItem('token', loginData.access_token);
            setUser({
              id: loginData.user.id,
              email: loginData.user.email,
              full_name: loginData.user.full_name,
              isPremium: loginData.user.is_premium,
              creditsRemaining: loginData.user.credits_remaining,
              subscription: null,
              token: loginData.access_token,
              isAuthenticated: true
            });
          }
        }
        setShowAuth(false);
        setAuthForm({ email: '', password: '', fullName: '' });
      } else {
        alert(data.detail || 'Authentication failed');
      }
    } catch (error) {
      console.error('Authentication error:', error);
      alert('Authentication failed. Please try again.');
    }
  };

  const logout = (): void => {
    localStorage.removeItem('token');
    setUser({
      isPremium: false,
      creditsRemaining: 2,
      subscription: null,
      token: null,
      isAuthenticated: false
    });
    setScoreResult(null);
    setActiveTab('upload');
  };

  // Check processing status
  useEffect(() => {
    let interval: NodeJS.Timeout | undefined;
    
    if (analysisId && isProcessing) {
      interval = setInterval(async () => {
        try {
          const response = await fetch(`${API_BASE}/resume/analysis/${analysisId}`, {
            headers: {
              'Authorization': `Bearer ${user.token}`
            }
          });
          
          if (response.ok) {
            const result: AnalysisResult = await response.json();
            console.log('Analysis result:', result); // Debug log
            
            if (result.processing_status === 'completed') {
              setScoreResult(result);
              console.log('Final scoreResult set:', result); // Debug log
              setIsProcessing(false);
              setActiveTab('results');
            } else if (result.processing_status === 'failed') {
              setIsProcessing(false);
              alert('Processing failed. Please try again.');
            } else {
              setProcessingStatus(result.processing_status);
            }
          }
        } catch (error) {
          console.error('Error checking status:', error);
        }
      }, 3000);
    }
    
    return () => {
      if (interval) {
        clearInterval(interval);
      }
    };
  }, [analysisId, isProcessing, user.token]);

  const analyzeResume = async (): Promise<void> => {
    if (!uploadedFile || !user.token) return;
    
    setIsProcessing(true);
    setProcessingStatus('uploading');
    
    try {
      const formData = new FormData();
      formData.append('file', uploadedFile);
      
      // Add job description if provided
      if (jobDescription.trim()) {
        formData.append('job_description', jobDescription.trim());
      }
      
      console.log('Uploading with job description:', jobDescription.trim()); // Debug log
      
      const response = await fetch(`${API_BASE}/resume/analyze`, {
        method: 'POST',
        
        headers: {
          'Authorization': `Bearer ${user.token}`
        },
        body: formData
      });
      
      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || 'Analysis failed');
      }
      
      const result = await response.json();
      console.log('Upload result:', result); // Debug log
      setAnalysisId(result.id);
      setProcessingStatus('processing');
      
      // Update user credits
      if (!user.isPremium) {
        setUser(prev => ({
          ...prev,
          creditsRemaining: Math.max(0, prev.creditsRemaining - 1)
        }));
      }
      
    } catch (error) {
      setIsProcessing(false);
      const errorMessage = error instanceof Error ? error.message : 'An error occurred';
      alert(errorMessage);
    }
  };

  const handleFileUpload = (event: React.ChangeEvent<HTMLInputElement>): void => {
    const file = event.target.files?.[0];
    if (file && file.type === 'application/pdf') {
      setUploadedFile(file);
    } else {
      alert('Please upload a PDF file');
    }
  };

  const subscribeToPlan = (plan: SubscriptionPlan): void => {
    setSelectedPlan(plan);
    setShowPayment(true);
  };

  const processPayment = (): void => {
    if (!selectedPlan) return;
    
    // Simulate payment processing
    setTimeout(() => {
      setUser(prev => ({
        ...prev,
        isPremium: true,
        creditsRemaining: selectedPlan.name === 'Pro' ? 50 : 999,
        subscription: selectedPlan
      }));
      setShowPayment(false);
      setSelectedPlan(null);
      alert('Payment successful! Welcome to Premium!');
    }, 1500);
  };

  const plans: SubscriptionPlan[] = [
    {
      name: 'Pro',
      price: '$9.99/month',
      features: [
        '50 resume scans per month',
        'Advanced AI insights with Gemini',
        'Job description matching',
        'ATS optimization scoring',
        'Industry benchmarking',
        'Cover letter analysis',
        'Email support'
      ],
      popular: true
    },
    {
      name: 'Enterprise',
      price: '$29.99/month',
      features: [
        'Unlimited resume scans',
        'Advanced Gemini AI analysis',
        'Team collaboration features',
        'Custom industry templates',
        'Salary range analysis',
        'Interview preparation',
        'Priority support',
        'API access'
      ],
      popular: false
    }
  ];

  const ScoreCircle: React.FC<ScoreCircleProps> = ({ score, label, color = "blue" }) => {
    const colorClasses = {
      blue: "text-blue-500",
      green: "text-green-500",
      purple: "text-purple-500",
      orange: "text-orange-500"
      
    };

    return (
      <div className="flex flex-col items-center">
        <div className="relative w-20 h-20">
          <svg className="w-20 h-20 transform -rotate-90" viewBox="0 0 100 100">
            <circle
              cx="50"
              cy="50"
              r="40"
              stroke="currentColor"
              strokeWidth="8"
              fill="transparent"
              className="text-gray-200"
            />
            <circle
              cx="50"
              cy="50"
              r="40"
              stroke="currentColor"
              strokeWidth="8"
              fill="transparent"
              strokeDasharray={`${score * 2.51} 251`}
              className={score >= 80 ? "text-green-500" : score >= 60 ? "text-yellow-500" : "text-red-500"}
            />
          </svg>
          <div className="absolute inset-0 flex items-center justify-center">
            <span className="text-xl font-bold">{Math.round(score || 0)}</span>
          </div>
        </div>
        <span className="text-sm text-gray-600 mt-2 text-center">{label}</span>
      </div>
    );
  };

  const ProcessingStatus: React.FC = () => (
    <div className="bg-white rounded-2xl shadow-xl p-8 text-center">
      <div className="animate-spin h-16 w-16 border-4 border-blue-500 border-t-transparent rounded-full mx-auto mb-6"></div>
      <h3 className="text-2xl font-bold text-gray-900 mb-4">
        {processingStatus === 'uploading' ? 'Uploading Resume...' :
         processingStatus === 'processing' ? 'Analyzing with Gemini AI...' :
         'Processing Resume...'}
      </h3>
      <p className="text-gray-600 mb-4">
        {processingStatus === 'uploading' ? 'Extracting text from your PDF' :
         processingStatus === 'processing' ? 'Our AI is analyzing your resume and matching it with job requirements' :
         'Please wait while we process your resume'}
      </p>
      <div className="flex items-center justify-center space-x-2 text-sm text-blue-600">
        <RefreshCw className="h-4 w-4 animate-spin" />
        <span>Powered by Google Gemini 1.5 Flash</span>
      </div>
    </div>
  );

  // Auth Modal
  const AuthModal: React.FC = React.memo(() => {
    const handleFormSubmit = useCallback((e: React.FormEvent) => {
      e.preventDefault();
      // handleAuth();
    }, []);

    const handleInputChange = useCallback((field: keyof AuthForm) => (e: React.ChangeEvent<HTMLInputElement>) => {
      setAuthForm(prev => ({ 
        ...prev, 
        [field]: e.target.value 
      }));
    },[]);

    return (
      <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center p-4 z-50">
        <div className="bg-white rounded-2xl shadow-2xl max-w-md w-full p-8">
          <div className="text-center mb-6">
            {authMode === 'login' ? 
              <LogIn className="h-12 w-12 text-blue-500 mx-auto mb-4" /> : 
              <UserPlus className="h-12 w-12 text-green-500 mx-auto mb-4" />
            }
            <h3 className="text-2xl font-bold text-gray-900 mb-2">
              {authMode === 'login' ? 'Sign In' : 'Create Account'}
            </h3>
            <p className="text-gray-600">
              {authMode === 'login' ? 'Welcome back! Please sign in to continue.' : 'Join us to start analyzing your resume with AI.'}
            </p>
          </div>

          <form onSubmit={handleFormSubmit} className="space-y-4 mb-6">
            {authMode === 'register' && (
              <div>
                <label htmlFor="fullName" className="block text-sm font-medium text-gray-700 mb-2">
                  Full Name *
                </label>
                <input
                  id="fullName"
                  type="text"
                  value={authForm.fullName}
                  onChange={handleInputChange('fullName')}
                  className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  placeholder="Enter your full name"
                  required={authMode === 'register'}
                />
              </div>
            )}
            
            <div>
              <label htmlFor="email" className="block text-sm font-medium text-gray-700 mb-2">
                Email *
              </label>
              <input
                id="email"
                type="email"
                value={authForm.email}
                onChange={handleInputChange('email')}
                className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                placeholder="Enter your email"
                required
                autoComplete="email"
              />
            </div>
            
            <div>
              <label htmlFor="password" className="block text-sm font-medium text-gray-700 mb-2">
                Password *
              </label>
              <input
                id="password"
                type="password"
                value={authForm.password}
                onChange={handleInputChange('password')}
                className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                placeholder="Enter your password"
                required
                autoComplete={authMode === 'login' ? 'current-password' : 'new-password'}
              />
            </div>

          <div className="flex flex-col space-y-3">
            <button
              type="submit"
              disabled={!authForm.email || !authForm.password || (authMode === 'register' && !authForm.fullName)}
              className="w-full bg-gradient-to-r from-blue-600 to-purple-600 text-white py-3 rounded-lg font-semibold hover:from-blue-700 hover:to-purple-700 transition-all duration-200 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {authMode === 'login' ? 'Sign In' : 'Create Account'}
            </button>
          </div>
          </form>
            
            <button
              type="button"
              onClick={() => {
                setAuthMode(authMode === 'login' ? 'register' : 'login');
                setAuthForm({ email: '', password: '', fullName: '' });
              }}
              className="text-blue-600 hover:text-blue-700 text-sm"
            >
              {authMode === 'login' ? 'Need an account? Sign up' : 'Already have an account? Sign in'}
            </button>
            
            <button
              type="button"
              onClick={() => {
                setShowAuth(false);
                setAuthForm({ email: '', password: '', fullName: '' });
              }}
              className="text-gray-500 hover:text-gray-700 text-sm"
            >
              Cancel
            </button>
          </div>
        </div>
      // </div>
    );
  });

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-purple-50">
      {/* Header */}
      <nav className="bg-white shadow-sm border-b">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center h-16">
            <div className="flex items-center space-x-2">
              <Target className="h-8 w-8 text-blue-600" />
              <span className="text-2xl font-bold text-gray-900">ResumeAI Pro</span>
              <span className="text-xs bg-green-100 text-green-800 px-2 py-1 rounded-full">Powered by Gemini</span>
            </div>
            <div className="flex items-center space-x-4">
              {user.isAuthenticated ? (
                <>
                  {user.isPremium ? (
                    <div className="flex items-center space-x-2 bg-gradient-to-r from-purple-500 to-pink-500 text-white px-4 py-2 rounded-full">
                      <Crown className="h-4 w-4" />
                      <span className="text-sm font-medium">Premium</span>
                    </div>
                  ) : (
                    <div className="flex items-center space-x-2">
                      <span className="text-sm text-gray-600">Free credits: {user.creditsRemaining}</span>
                      <button
                        onClick={() => setActiveTab('pricing')}
                        className="bg-gradient-to-r from-blue-600 to-purple-600 text-white px-4 py-2 rounded-lg hover:from-blue-700 hover:to-purple-700 transition-all duration-200"
                      >
                        Upgrade
                      </button>
                    </div>
                  )}
                  <button
                    onClick={logout}
                    className="text-gray-600 hover:text-gray-800 px-3 py-2 rounded-lg transition-colors"
                  >
                    Logout
                  </button>
                </>
              ) : (
                <button
                  onClick={() => setShowAuth(true)}
                  className="bg-gradient-to-r from-blue-600 to-purple-600 text-white px-4 py-2 rounded-lg hover:from-blue-700 hover:to-purple-700 transition-all duration-200"
                >
                  Sign In
                </button>
              )}
            </div>
          </div>
        </div>
      </nav>

      {/* Main Content */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Tab Navigation */}
        <div className="flex space-x-1 bg-gray-100 p-1 rounded-lg mb-8 w-fit">
          {[
            { id: 'upload' as TabType, label: 'Upload & Analyze', icon: Upload },
            { id: 'results' as TabType, label: 'Results', icon: BarChart3 },
            { id: 'pricing' as TabType, label: 'Pricing', icon: CreditCard }
          ].map(tab => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={`flex items-center space-x-2 px-4 py-2 rounded-md transition-all duration-200 ${
                activeTab === tab.id
                  ? 'bg-white text-blue-600 shadow-sm'
                  : 'text-gray-600 hover:text-gray-800'
              }`}
            >
              <tab.icon className="h-4 w-4" />
              <span>{tab.label}</span>
            </button>
          ))}
        </div>

        {/* Upload Tab */}
        {activeTab === 'upload' && !isProcessing && (
          <div className="max-w-4xl mx-auto">
            <div className="text-center mb-8">
              <h1 className="text-4xl font-bold text-gray-900 mb-4">
                AI-Powered Resume Analysis with Job Matching
              </h1>
              <p className="text-xl text-gray-600">
                Get instant feedback with Google Gemini 1.5 Flash and optimize for specific job opportunities
              </p>
            </div>

            <div className="grid lg:grid-cols-2 gap-8">
              {/* Resume Upload */}
              <div className="bg-white rounded-2xl shadow-xl p-6">
                <h3 className="text-xl font-semibold text-gray-900 mb-4">Upload Resume</h3>
                {!uploadedFile ? (
                  <div
                    onClick={() => fileInputRef.current?.click()}
                    className="border-2 border-dashed border-gray-300 rounded-xl p-8 text-center cursor-pointer hover:border-blue-400 hover:bg-blue-50 transition-all duration-200"
                  >
                    <Upload className="h-12 w-12 text-gray-400 mx-auto mb-4" />
                    <h4 className="text-lg font-medium text-gray-700 mb-2">
                      Upload Your Resume
                    </h4>
                    <p className="text-gray-500 mb-4">
                      Drag and drop your PDF resume or click to browse
                    </p>
                    <input
                      ref={fileInputRef}
                      type="file"
                      accept=".pdf"
                      onChange={handleFileUpload}
                      className="hidden"
                    />
                  </div>
                ) : (
                  <div className="text-center">
                    <div className="flex items-center justify-center mb-4">
                      <FileText className="h-12 w-12 text-blue-600" />
                    </div>
                    <h4 className="text-lg font-medium text-gray-700 mb-2">
                      {uploadedFile.name}
                    </h4>
                    <p className="text-gray-500 mb-4">
                      Ready for AI analysis
                    </p>
                    <button
                      onClick={() => {
                        setUploadedFile(null);
                        setJobDescription('');
                      }}
                      className="text-blue-600 hover:text-blue-700"
                    >
                      Change File
                    </button>
                  </div>
                )}
              </div>

              {/* Job Description (Optional) */}
              <div className="bg-white rounded-2xl shadow-xl p-6">
                <div className="flex items-center space-x-2 mb-4">
                  <Briefcase className="h-5 w-5 text-purple-600" />
                  <h3 className="text-xl font-semibold text-gray-900">Job Description (Optional)</h3>
                  <span className="text-xs bg-purple-100 text-purple-800 px-2 py-1 rounded-full">Enhanced Matching</span>
                </div>
                <textarea
                  value={jobDescription}
                  onChange={(e: React.ChangeEvent<HTMLTextAreaElement>) => setJobDescription(e.target.value)}
                  placeholder="Paste the job description here to get targeted matching scores and ATS optimization recommendations..."
                  className="w-full h-40 p-4 border border-gray-300 rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-transparent resize-none"
                />
                <div className="mt-3 text-sm text-gray-600">
                  <div className="flex items-center space-x-2 mb-2">
                    <TrendingUp className="h-4 w-4 text-green-500" />
                    <span>Get job-specific matching score</span>
                  </div>
                  <div className="flex items-center space-x-2 mb-2">
                    <Target className="h-4 w-4 text-blue-500" />
                    <span>ATS optimization for this role</span>
                  </div>
                  <div className="flex items-center space-x-2">
                    <Zap className="h-4 w-4 text-purple-500" />
                    <span>Tailored improvement suggestions</span>
                  </div>
                </div>
              </div>
            </div>

            {/* Analysis Button */}
            <div className="text-center mt-8">
              {!user.isAuthenticated ? (
                <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-6 mb-6">
                  <AlertCircle className="h-8 w-8 text-yellow-600 mx-auto mb-3" />
                  <h3 className="text-lg font-semibold text-yellow-800 mb-2">Sign in Required</h3>
                  <p className="text-yellow-700 mb-4">Please sign in to analyze your resume with our AI-powered system</p>
                  <button
                    onClick={() => setShowAuth(true)}
                    className="bg-blue-600 text-white px-6 py-2 rounded-lg hover:bg-blue-700 transition-colors"
                  >
                    Sign In Now
                  </button>
                </div>
              ) : !user.isPremium && user.creditsRemaining === 0 ? (
                <div className="bg-orange-50 border border-orange-200 rounded-lg p-6 mb-6">
                  <CreditCard className="h-8 w-8 text-orange-600 mx-auto mb-3" />
                  <h3 className="text-lg font-semibold text-orange-800 mb-2">No Credits Remaining</h3>
                  <p className="text-orange-700 mb-4">
                    You've used all your free credits. Upgrade to Premium for unlimited scans!
                  </p>
                  <button
                    onClick={() => setActiveTab('pricing')}
                    className="bg-orange-600 text-white px-6 py-2 rounded-lg hover:bg-orange-700 transition-colors"
                  >
                    Upgrade Now
                  </button>
                </div>
              ) : (
                uploadedFile && (
                  <button
                    onClick={analyzeResume}
                    disabled={isProcessing}
                    className="bg-gradient-to-r from-blue-600 to-purple-600 text-white px-8 py-4 rounded-lg font-semibold hover:from-blue-700 hover:to-purple-700 transition-all duration-200 flex items-center space-x-3 mx-auto text-lg disabled:opacity-50 disabled:cursor-not-allowed"
                  >
                    <Zap className="h-6 w-6" />
                    <span>Analyze with Gemini AI</span>
                    {jobDescription.trim() && <Briefcase className="h-5 w-5" />}
                  </button>
                )
              )}
            </div>
          </div>
        )}

        {/* Processing Tab */}
        {isProcessing && (
          <div className="max-w-2xl mx-auto">
            <ProcessingStatus />
          </div>
        )}

        {/* Results Tab */}
        {activeTab === 'results' && scoreResult && !isProcessing && (
          console.log('Rendering results with scoreResult:', scoreResult) ,
          <div className="max-w-6xl mx-auto">
            <div className="bg-white rounded-2xl shadow-xl p-8 mb-8">
              <div className="flex items-center justify-between mb-8">
                <h2 className="text-3xl font-bold text-gray-900">
                  Resume Analysis Results
                </h2>   
                <div className="flex items-center space-x-2 text-sm text-green-600">
                  <CheckCircle className="h-4 w-4" />
                  <span>Analyzed by Gemini 1.5 Flash</span>
                </div>
              </div>
              
              Main Scores
              <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-12">
                Overall Score
                <div className="text-center">
                  <div className="relative w-28 h-28 mx-auto mb-3">
                    <svg className="w-28 h-28 transform -rotate-90" viewBox="0 0 100 100">
                      <circle
                        cx="50"
                        cy="50"
                        r="40"
                        stroke="currentColor"
                        strokeWidth="6"
                        fill="transparent"
                        className="text-gray-200"
                      />
                      <circle
                        cx="50"
                        cy="50"
                        r="40"
                        stroke="currentColor"
                        strokeWidth="6"
                        fill="transparent"
                        strokeDasharray={`${(scoreResult.overall_score || 0) * 2.51} 251`}
                        className="text-blue-600"
                      />
                    </svg>
                    <div className="absolute inset-0 flex items-center justify-center">
                      <div className="text-center">
                        <span className="text-2xl font-bold text-gray-900">{Math.round(scoreResult.overall_score || 0)}</span>
                        <div className="text-xs text-gray-600">Overall</div>
                      </div>
                    </div>
                  </div>
                  <div className="text-sm font-medium text-gray-700">Resume Quality</div>
                </div>

                {/* ATS Score */}
                <div className="text-center">
                  <div className="relative w-28 h-28 mx-auto mb-3">
                    <svg className="w-28 h-28 transform -rotate-90" viewBox="0 0 100 100">
                      <circle
                        cx="50"
                        cy="50"
                        r="40"
                        stroke="currentColor"
                        strokeWidth="6"
                        fill="transparent"
                        className="text-gray-200"
                      />
                      <circle
                        cx="50"
                        cy="50"
                        r="40"
                        stroke="currentColor"
                        strokeWidth="6"
                        fill="transparent"
                        strokeDasharray={`${(scoreResult.ats_score || 0) * 2.51} 251`}
                        className="text-green-600"
                      />
                    </svg>
                    <div className="absolute inset-0 flex items-center justify-center">
                      <div className="text-center">
                        <span className="text-2xl font-bold text-gray-900">{Math.round(scoreResult.ats_score || 0)}</span>
                        <div className="text-xs text-gray-600">ATS</div>
                      </div>
                    </div>
                  </div>
                  <div className="text-sm font-medium text-gray-700">ATS Compatibility</div>
                </div>

                {/* Job Match Score */}
                {scoreResult.job_match_score !== null && scoreResult.job_match_score !== undefined && (
                  <div className="text-center">
                    <div className="relative w-28 h-28 mx-auto mb-3">
                      <svg className="w-28 h-28 transform -rotate-90" viewBox="0 0 100 100">
                        <circle
                          cx="50"
                          cy="50"
                          r="40"
                          stroke="currentColor"
                          strokeWidth="6"
                          fill="transparent"
                          className="text-gray-200"
                        />
                        <circle
                          cx="50"
                          cy="50"
                          r="40"
                          stroke="currentColor"
                          strokeWidth="6"
                          fill="transparent"
                          strokeDasharray={`${scoreResult.job_match_score * 2.51} 251`}
                          className="text-purple-600"
                        />
                      </svg>
                      <div className="absolute inset-0 flex items-center justify-center">
                        <div className="text-center">
                          <span className="text-2xl font-bold text-gray-900">{Math.round(scoreResult.job_match_score)}</span>
                          <div className="text-xs text-gray-600">Match</div>
                        </div>
                      </div>
                    </div>
                    <div className="text-sm font-medium text-gray-700">Job Match</div>
                  </div>
                )}

                {/* Industry Benchmark (Premium Feature) */}
                {user.isPremium && (
                  <div className="text-center">
                    <div className="relative w-28 h-28 mx-auto mb-3">
                      <svg className="w-28 h-28 transform -rotate-90" viewBox="0 0 100 100">
                        <circle
                          cx="50"
                          cy="50"
                          r="40"
                          stroke="currentColor"
                          strokeWidth="6"
                          fill="transparent"
                          className="text-gray-200"
                        />
                        <circle
                          cx="50"
                          cy="50"
                          r="40"
                          stroke="currentColor"
                          strokeWidth="6"
                          fill="transparent"
                          strokeDasharray={`${85 * 2.51} 251`}
                          className="text-orange-600"
                        />
                      </svg>
                      <div className="absolute inset-0 flex items-center justify-center">
                        <div className="text-center">
                          <span className="text-2xl font-bold text-gray-900">85</span>
                          <div className="text-xs text-gray-600">Rank</div>
                        </div>
                      </div>
                    </div>
                    <div className="text-sm font-medium text-gray-700">Industry Ranking</div>
                  </div>
                )}
              </div>

              {/* ATS Details Section */}
              <div className="bg-green-50 border border-green-200 rounded-lg p-6 mb-8">
                <div className="flex items-center space-x-3 mb-4">
                  <div className="bg-green-100 p-2 rounded-full">
                    <Target className="h-6 w-6 text-green-600" />
                  </div>
                  <h3 className="text-xl font-semibold text-green-900">ATS Analysis Report</h3>
                  <span className="bg-green-100 text-green-800 text-sm font-medium px-2.5 py-0.5 rounded">
                    Score: {Math.round(scoreResult.ats_score || 0)}%
                  </span>
                </div>
                <div className="grid md:grid-cols-2 gap-6">
                  <div>
                    <h4 className="font-semibold text-green-800 mb-3">ATS Compatibility Factors</h4>
                    <div className="space-y-2">
                      <div className="flex justify-between">
                        <span className="text-green-700">Keyword Density</span>
                        <span className="font-medium text-green-800">{Math.round((scoreResult.category_scores?.keywords || 0))}%</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-green-700">Format Structure</span>
                        <span className="font-medium text-green-800">{Math.round((scoreResult.category_scores?.formatting || 0))}%</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-green-700">Content Quality</span>
                        <span className="font-medium text-green-800">{Math.round((scoreResult.category_scores?.content || 0))}%</span>
                      </div>
                    </div>
                  </div>
                  <div>
                    <h4 className="font-semibold text-green-800 mb-3">ATS Recommendations</h4>
                    <ul className="space-y-2 text-green-700">
                      <li className="flex items-start space-x-2">
                        <CheckCircle className="h-4 w-4 text-green-600 mt-0.5 flex-shrink-0" />
                        <span className="text-sm">Use standard section headers</span>
                      </li>
                      <li className="flex items-start space-x-2">
                        <CheckCircle className="h-4 w-4 text-green-600 mt-0.5 flex-shrink-0" />
                        <span className="text-sm">Include relevant industry keywords</span>
                      </li>
                      <li className="flex items-start space-x-2">
                        <CheckCircle className="h-4 w-4 text-green-600 mt-0.5 flex-shrink-0" />
                        <span className="text-sm">Avoid complex formatting elements</span>
                      </li>
                    </ul>
                  </div>
                </div>
              </div>

              {/* Category Scores */}
              <div className="mb-12">
                <h3 className="text-xl font-semibold text-gray-900 mb-6">Category Breakdown</h3>
                <div className="grid grid-cols-2 md:grid-cols-5 gap-6">
                  {scoreResult.category_scores && Object.entries(scoreResult.category_scores).map(([category, score]) => (
                    <ScoreCircle
                      key={category}
                      score={score}
                      label={category.charAt(0).toUpperCase() + category.slice(1)}  
                      color={category === 'keywords' ? 'blue' : category === 'formatting' ? 'green' : category === 'content' ? 'purple' : 'orange'}
                    />
                  ))}
                </div>
              </div>

              {/* Suggestions */}
              <div className="mb-8">
                <h3 className="text-xl font-semibold text-gray-900 mb-4">AI-Powered Suggestions</h3>
                <div className="space-y-3">
                  {(scoreResult.suggestions || []).map((suggestion, index) => (
                    <div key={index} className="flex items-start space-x-3 p-4 bg-blue-50 rounded-lg">
                      <ArrowRight className="h-5 w-5 text-blue-600 mt-0.5 flex-shrink-0" />
                      <span className="text-gray-700">{suggestion.area}</span>
                      <span className="text-gray-700">{suggestion.description}</span>
                      <span className="text-gray-700">{suggestion.action}</span>
                    </div>  
                  ))}
                </div>
              </div>

              {/* Premium Insights */}
              {user.isPremium && scoreResult.premium_insights && (
                <div className="bg-gradient-to-r from-purple-50 to-pink-50 border border-purple-200 rounded-lg p-6 mb-8">
                  <div className="flex items-center space-x-2 mb-4">
                    <Crown className="h-5 w-5 text-purple-600" />
                    <h3 className="text-xl font-semibold text-purple-900">Premium Strategic Insights</h3>
                  </div>
                  <div className="space-y-3">
                    {scoreResult.premium_insights.map((insight, index) => (
                      <div key={index} className="flex items-start space-x-3">
                        <Star className="h-5 w-5 text-purple-600 mt-0.5 flex-shrink-0" />
                        <span className="text-purple-800">{insight}</span>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Job Match Analysis */}
              {scoreResult.job_match_score !== null && scoreResult.job_match_score !== undefined && (
                <div className="bg-gradient-to-r from-purple-50 to-blue-50 border border-purple-200 rounded-lg p-6 mb-8">
                  <div className="flex items-center space-x-3 mb-4">
                    <div className="bg-purple-100 p-2 rounded-full">
                      <Briefcase className="h-6 w-6 text-purple-600" />
                    </div>
                    <h3 className="text-xl font-semibold text-purple-900">Job Description Match Analysis</h3>
                    <span className="bg-purple-100 text-purple-800 text-sm font-medium px-2.5 py-0.5 rounded">
                      Match: {Math.round(scoreResult.job_match_score)}%
                    </span>
                  </div>
                  <div className="grid md:grid-cols-2 gap-6">
                    <div>
                      <h4 className="font-semibold text-purple-800 mb-3">Alignment Strengths</h4>
                      <ul className="space-y-2 text-purple-700">
                        <li className="flex items-start space-x-2">
                          <CheckCircle className="h-4 w-4 text-purple-600 mt-0.5 flex-shrink-0" />
                          <span className="text-sm">Strong skill match with job requirements</span>
                        </li>
                        <li className="flex items-start space-x-2">
                          <CheckCircle className="h-4 w-4 text-purple-600 mt-0.5 flex-shrink-0" />
                          <span className="text-sm">Relevant experience level</span>
                        </li>
                        <li className="flex items-start space-x-2">
                          <CheckCircle className="h-4 w-4 text-purple-600 mt-0.5 flex-shrink-0" />
                          <span className="text-sm">Industry-specific keywords present</span>
                        </li>
                      </ul>
                    </div>
                    <div>
                      <h4 className="font-semibold text-blue-800 mb-3">Areas for Improvement</h4>
                      <ul className="space-y-2 text-blue-700">
                        <li className="flex items-start space-x-2">
                          <ArrowRight className="h-4 w-4 text-blue-600 mt-0.5 flex-shrink-0" />
                          <span className="text-sm">Add more specific technical skills mentioned in job posting</span>
                        </li>
                        <li className="flex items-start space-x-2">
                          <ArrowRight className="h-4 w-4 text-blue-600 mt-0.5 flex-shrink-0" />
                          <span className="text-sm">Include quantifiable achievements related to job requirements</span>
                        </li>
                        <li className="flex items-start space-x-2">
                          <ArrowRight className="h-4 w-4 text-blue-600 mt-0.5 flex-shrink-0" />
                          <span className="text-sm">Align language with job description terminology</span>
                        </li>
                      </ul>
                    </div>
                  </div>
                </div>
              )}

              {!user.isPremium && (
                <div className="bg-gradient-to-r from-blue-50 to-purple-50 border border-blue-200 rounded-lg p-6 text-center">
                  <Crown className="h-12 w-12 text-purple-600 mx-auto mb-4" />
                  <h3 className="text-xl font-semibold text-gray-900 mb-2">
                    Unlock Premium Insights
                  </h3>
                  <p className="text-gray-600 mb-4">
                    Get advanced career strategy, salary analysis, and personalized recommendations with Gemini Pro
                  </p>
                  <button
                    onClick={() => setActiveTab('pricing')}
                    className="bg-gradient-to-r from-purple-600 to-pink-600 text-white px-6 py-3 rounded-lg font-semibold hover:from-purple-700 hover:to-pink-700 transition-all duration-200"
                  >
                    Upgrade to Premium
                  </button>
                </div>
              )}
            </div>
          </div>
        )}

        {/* Pricing Tab */}
        {activeTab === 'pricing' && (
          <div className="max-w-5xl mx-auto">
            <div className="text-center mb-12">
              <h2 className="text-4xl font-bold text-gray-900 mb-4">
                Choose Your Plan
              </h2>
              <p className="text-xl text-gray-600">
                Unlock the full power of Google Gemini AI for resume optimization
              </p>
            </div>

            <div className="grid md:grid-cols-2 gap-8 mb-8">
              {plans.map((plan, index) => (
                <div
                  key={index}
                  className={`bg-white rounded-2xl shadow-xl p-8 relative ${
                    plan.popular ? 'ring-2 ring-purple-500' : ''
                  }`}
                >
                  {plan.popular && (
                    <div className="absolute -top-4 left-1/2 transform -translate-x-1/2">
                      <span className="bg-gradient-to-r from-purple-500 to-pink-500 text-white px-4 py-1 rounded-full text-sm font-medium">
                        Most Popular
                      </span>
                    </div>
                  )}
                  
                  <div className="text-center mb-8">
                    <h3 className="text-2xl font-bold text-gray-900 mb-2">{plan.name}</h3>
                    <p className="text-4xl font-bold text-blue-600">{plan.price}</p>
                  </div>

                  <ul className="space-y-4 mb-8">
                    {plan.features.map((feature, featureIndex) => (
                      <li key={featureIndex} className="flex items-center space-x-3">
                        <CheckCircle className="h-5 w-5 text-green-500 flex-shrink-0" />
                        <span className="text-gray-700">{feature}</span>
                      </li>
                    ))}
                  </ul>

                  <button
                    onClick={() => subscribeToPlan(plan)}
                    className={`w-full py-3 rounded-lg font-semibold transition-all duration-200 ${
                      plan.popular
                        ? 'bg-gradient-to-r from-purple-600 to-pink-600 text-white hover:from-purple-700 hover:to-pink-700'
                        : 'bg-gray-100 text-gray-900 hover:bg-gray-200'
                    }`}
                  >
                    Get Started
                  </button>
                </div>
              ))}
            </div>

            {/* Free Plan */}
            <div className="bg-gray-50 rounded-2xl p-8 text-center">
              <h3 className="text-xl font-semibold text-gray-900 mb-4">Free Plan</h3>
              <div className="grid md:grid-cols-3 gap-4 text-gray-600 mb-4">
                <div>• 2 resume scans per month</div>
                <div>• Basic Gemini analysis</div>
                <div>• Standard suggestions</div>
              </div>
              <p className="text-sm text-gray-500">
                Perfect for trying out our Gemini-powered analysis
              </p>
            </div>
          </div>
        )}
      </div>

      {/* Auth Modal */}
      {showAuth && <AuthModal />}

      {/* Payment Modal */}
      {showPayment && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center p-4 z-50">
          <div className="bg-white rounded-2xl shadow-2xl max-w-md w-full p-8">
            <div className="text-center mb-6">
              <Shield className="h-12 w-12 text-green-500 mx-auto mb-4" />
              <h3 className="text-2xl font-bold text-gray-900 mb-2">
                Subscribe to {selectedPlan?.name}
              </h3>
              <p className="text-gray-600">{selectedPlan?.price}</p>
            </div>

            <div className="space-y-4 mb-6">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Card Number
                </label>
                <input
                  type="text"
                  placeholder="1234 5678 9012 3456"
                  className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                />
              </div>
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Expiry
                  </label>
                  <input
                    type="text"
                    placeholder="MM/YY"
                    className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    CVC
                  </label>
                  <input
                    type="text"
                    placeholder="123"
                    className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  />
                </div>
              </div>
            </div>

            <div className="flex space-x-4">
              <button
                onClick={() => setShowPayment(false)}
                className="flex-1 bg-gray-100 text-gray-700 py-3 rounded-lg font-semibold hover:bg-gray-200 transition-colors"
              >
                Cancel
              </button>
              <button
                onClick={processPayment}
                className="flex-1 bg-gradient-to-r from-green-600 to-blue-600 text-white py-3 rounded-lg font-semibold hover:from-green-700 hover:to-blue-700 transition-all duration-200"
              >
                Subscribe
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default ResumeScoringSoftware;