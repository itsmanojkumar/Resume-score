// src/components/AuthModal.tsx
import React from 'react';
import { LogIn, UserPlus } from 'lucide-react';

// ----------------------
// 🔹 Type Definitions
// ----------------------
export type AuthMode = 'login' | 'register';

export interface AuthForm {
  email: string;
  password: string;
  fullName: string;
}

interface AuthModalProps {
  authMode: AuthMode;
  authForm: AuthForm;
  setAuthForm: React.Dispatch<React.SetStateAction<AuthForm>>;
  setAuthMode: React.Dispatch<React.SetStateAction<AuthMode>>;
  setShowAuth: React.Dispatch<React.SetStateAction<boolean>>;
  handleAuth: () => void;
}

// ----------------------
// 🔹 AuthModal Component
// ----------------------
const AuthModal: React.FC<AuthModalProps> = ({
  authMode,
  authForm,
  setAuthForm,
  setAuthMode,
  setShowAuth,
  handleAuth
}) => {
  // form submit
  const handleFormSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    handleAuth();
  };

  // input change handler
  const handleInputChange =
    (field: keyof AuthForm) =>
    (e: React.ChangeEvent<HTMLInputElement>) => {
      setAuthForm(prev => ({
        ...prev,
        [field]: e.target.value
      }));
    };

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center p-4 z-50">
      <div className="bg-white rounded-2xl shadow-2xl max-w-md w-full p-8 animate-fadeIn">
        {/* Header */}
        <div className="text-center mb-6">
          {authMode === 'login' ? (
            <LogIn className="h-12 w-12 text-blue-500 mx-auto mb-4" />
          ) : (
            <UserPlus className="h-12 w-12 text-green-500 mx-auto mb-4" />
          )}
          <h3 className="text-2xl font-bold text-gray-900 mb-2">
            {authMode === 'login' ? 'Sign In' : 'Create Account'}
          </h3>
          <p className="text-gray-600">
            {authMode === 'login'
              ? 'Welcome back! Please sign in to continue.'
              : 'Join us to start analyzing your resume with AI.'}
          </p>
        </div>

        {/* Form */}
        <form onSubmit={handleFormSubmit} className="space-y-4 mb-6">
          {authMode === 'register' && (
            <div>
              <label
                htmlFor="fullName"
                className="block text-sm font-medium text-gray-700 mb-2"
              >
                Full Name *
              </label>
              <input
                id="fullName"
                type="text"
                value={authForm.fullName}
                onChange={handleInputChange('fullName')}
                className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                placeholder="Enter your full name"
                required
              />
            </div>
          )}

          <div>
            <label
              htmlFor="email"
              className="block text-sm font-medium text-gray-700 mb-2"
            >
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
            <label
              htmlFor="password"
              className="block text-sm font-medium text-gray-700 mb-2"
            >
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
              autoComplete={
                authMode === 'login' ? 'current-password' : 'new-password'
              }
            />
          </div>

          {/* Buttons */}
          <div className="flex flex-col space-y-3">
            <button
              type="submit"
              disabled={
                !authForm.email ||
                !authForm.password ||
                (authMode === 'register' && !authForm.fullName)
              }
              className="w-full bg-gradient-to-r from-blue-600 to-purple-600 text-white py-3 rounded-lg font-semibold hover:from-blue-700 hover:to-purple-700 transition-all duration-200 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {authMode === 'login' ? 'Sign In' : 'Create Account'}
            </button>

            <button
              type="button"
              onClick={() => {
                setAuthMode(authMode === 'login' ? 'register' : 'login');
                setAuthForm({ email: '', password: '', fullName: '' });
              }}
              className="text-blue-600 hover:text-blue-700 text-sm"
            >
              {authMode === 'login'
                ? 'Need an account? Sign up'
                : 'Already have an account? Sign in'}
            </button>

            <button
              type="button"
              onClick={() => setShowAuth(false)}
              className="text-gray-500 hover:text-gray-700 text-sm"
            >
              Cancel
            </button>
          </div>
        </form>
      </div>
    </div>
  );
};

// ----------------------
// 🔹 Memoize to prevent lag/re-renders
// ----------------------
export default React.memo(AuthModal);
