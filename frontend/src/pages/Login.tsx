import React, { useState } from 'react';
import { Link, useNavigate, useLocation } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';
import { useTheme } from '../context/ThemeContext';
import { Scroll, Lock, Mail, AlertCircle, ArrowRight, ShieldCheck, Sun, Moon } from 'lucide-react';

export const Login: React.FC = () => {
  const { login } = useAuth();
  const { isDark, toggleTheme } = useTheme();
  const navigate = useNavigate();
  const location = useLocation();

  const [usernameOrEmail, setUsernameOrEmail] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState<string | null>(null);
  const [isSubmitting, setIsSubmitting] = useState(false);

  const from = (location.state as any)?.from?.pathname || '/dashboard';

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!usernameOrEmail.trim() || !password.trim()) {
      setError('Please provide both username/email and password.');
      return;
    }

    setError(null);
    setIsSubmitting(true);

    try {
      await login(usernameOrEmail, password);
      navigate(from, { replace: true });
    } catch (err: any) {
      setError(err.message || 'Authentication failed. Please verify your credentials.');
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleFillDemo = () => {
    setUsernameOrEmail('researcher');
    setPassword('ScholarPass2026!');
  };

  return (
    <div className="min-h-screen flex items-center justify-center bg-archival-paper py-12 px-4 sm:px-6 lg:px-8 relative">
      <div className="absolute top-6 right-6">
        <button
          onClick={toggleTheme}
          type="button"
          className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-stone-100 hover:bg-stone-200 dark:bg-[#18221D] dark:hover:bg-[#1E2B25] text-stone-700 dark:text-stone-300 border border-stone-300 dark:border-[#27362E] transition shadow-xs cursor-pointer text-xs font-medium"
          aria-label={isDark ? 'Switch to Light Mode' : 'Switch to Dark Mode'}
          title={isDark ? 'Switch to Light Mode' : 'Switch to Dark Mode'}
        >
          {isDark ? (
            <>
              <Sun className="w-4 h-4 text-amber-400" />
              <span>Light</span>
            </>
          ) : (
            <>
              <Moon className="w-4 h-4 text-emerald-700" />
              <span>Dark</span>
            </>
          )}
        </button>
      </div>
      <div className="max-w-md w-full space-y-8 bg-white p-8 rounded-xl border border-stone-300 shadow-xl">
        {/* Header */}
        <div className="text-center">
          <div className="mx-auto w-12 h-12 rounded-xl bg-emerald-800 flex items-center justify-center text-emerald-200 border border-emerald-600 shadow-md">
            <Scroll className="w-6 h-6" />
          </div>
          <h2 className="mt-4 font-serif-scholarly text-2xl font-bold tracking-tight text-stone-900">
            Past2Pixel Laboratory
          </h2>
          <p className="mt-1 text-xs text-stone-500">
            Ancient Manuscript Damage Localization & Neural Restoration System
          </p>
        </div>

        {/* Error Alert */}
        {error && (
          <div className="p-3 rounded-lg bg-rose-50 border border-rose-200 text-rose-800 text-xs flex items-start gap-2">
            <AlertCircle className="w-4 h-4 shrink-0 mt-0.5 text-rose-600" />
            <span>{error}</span>
          </div>
        )}

        {/* Form */}
        <form className="mt-6 space-y-4" onSubmit={handleSubmit}>
          <div>
            <label className="block text-xs font-medium text-stone-700">
              Username or Registered Email
            </label>
            <div className="mt-1 relative rounded-md shadow-xs">
              <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none text-stone-400">
                <Mail className="w-4 h-4" />
              </div>
              <input
                type="text"
                value={usernameOrEmail}
                onChange={(e) => setUsernameOrEmail(e.target.value)}
                placeholder="e.g. researcher@past2pixel.ai"
                required
                className="block w-full pl-9 pr-3 py-2 text-sm border border-stone-300 rounded-lg focus:ring-2 focus:ring-emerald-700 focus:border-emerald-700 bg-stone-50"
              />
            </div>
          </div>

          <div>
            <label className="block text-xs font-medium text-stone-700">Password</label>
            <div className="mt-1 relative rounded-md shadow-xs">
              <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none text-stone-400">
                <Lock className="w-4 h-4" />
              </div>
              <input
                type="password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                placeholder="••••••••••••"
                required
                className="block w-full pl-9 pr-3 py-2 text-sm border border-stone-300 rounded-lg focus:ring-2 focus:ring-emerald-700 focus:border-emerald-700 bg-stone-50"
              />
            </div>
          </div>

          <button
            type="submit"
            disabled={isSubmitting}
            className="w-full flex items-center justify-center gap-2 py-2.5 px-4 border border-transparent rounded-lg shadow-sm text-sm font-medium text-white bg-emerald-800 hover:bg-emerald-700 focus:outline-hidden focus:ring-2 focus:ring-offset-2 focus:ring-emerald-600 transition disabled:opacity-60 cursor-pointer"
          >
            {isSubmitting ? (
              <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" />
            ) : (
              <>
                <span>Sign In to Laboratory</span>
                <ArrowRight className="w-4 h-4" />
              </>
            )}
          </button>
        </form>

        {/* Demo Account Helper */}
        <div className="pt-2 border-t border-stone-200">
          <div className="flex items-center justify-between text-xs text-stone-500">
            <span>Academic Demonstration Access:</span>
            <button
              type="button"
              onClick={handleFillDemo}
              className="text-emerald-700 hover:text-emerald-800 font-semibold underline cursor-pointer"
            >
              Fill Demo Credentials
            </button>
          </div>
        </div>

        {/* Register link */}
        <p className="text-center text-xs text-stone-600">
          New manuscript researcher?{' '}
          <Link to="/register" className="font-semibold text-emerald-800 hover:text-emerald-700">
            Register new scholar account
          </Link>
        </p>
      </div>
    </div>
  );
};
