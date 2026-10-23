import React, { useEffect, useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';
import { api } from '../services/api';
import { SystemStatus } from '../types/api';
import { StatusBadge } from './StatusBadge';
import {
  Scroll,
  Cpu,
  LogOut,
  User as UserIcon,
  Database,
  ExternalLink,
  Sun,
  Moon,
} from 'lucide-react';
import { useTheme } from '../context/ThemeContext';

export const Navbar: React.FC = () => {
  const { user, isAuthenticated, logout } = useAuth();
  const { isDark, toggleTheme } = useTheme();
  const navigate = useNavigate();
  const [status, setStatus] = useState<SystemStatus | null>(null);

  useEffect(() => {
    api.system
      .getStatus()
      .then((res) => setStatus(res))
      .catch(() => setStatus(null));
  }, []);

  const handleLogout = () => {
    logout();
    navigate('/login');
  };

  return (
    <header className="sticky top-0 z-40 bg-stone-900 dark:bg-[#141A17] text-stone-100 border-b border-stone-800 dark:border-[#27362E] shadow-md transition-colors duration-200">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
        {/* Brand */}
        <Link to="/" className="flex items-center gap-3 group">
          <div className="w-10 h-10 rounded-lg bg-emerald-800 dark:bg-emerald-900 flex items-center justify-center text-emerald-200 border border-emerald-600 dark:border-emerald-700 shadow-sm group-hover:bg-emerald-700 dark:group-hover:bg-emerald-800 transition">
            <Scroll className="w-5 h-5" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <span className="font-serif-scholarly font-bold text-lg tracking-wide text-amber-100">
                Past2Pixel
              </span>
              <span className="text-[10px] uppercase tracking-wider px-1.5 py-0.5 rounded bg-emerald-950 text-emerald-400 border border-emerald-800/80 font-mono">
                DL AI Lab
              </span>
            </div>
            <p className="text-xs text-stone-400 hidden sm:block">
              Manuscript Damage Localization, Restoration & Preservation
            </p>
          </div>
        </Link>

        {/* Center Diagnostics Badges */}
        <div className="hidden md:flex items-center gap-3 text-xs">
          {status ? (
            <>
              <div className="flex items-center gap-1.5 px-2.5 py-1 rounded bg-stone-800/80 dark:bg-[#18221D] border border-stone-700 dark:border-[#27362E] text-stone-300">
                <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse" />
                <span>API {status.version}</span>
              </div>
            </>
          ) : (
            <div className="flex items-center gap-1.5 px-2.5 py-1 rounded bg-stone-800/80 dark:bg-[#18221D] border border-stone-700 dark:border-[#27362E] text-stone-400">
              <span className="w-2 h-2 rounded-full bg-amber-500" />
              <span>Connecting Backend...</span>
            </div>
          )}
        </div>

        {/* User / Authentication actions & Theme Toggle */}
        <div className="flex items-center gap-3">
          {/* Theme Toggle Button */}
          <button
            onClick={toggleTheme}
            type="button"
            className="flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg bg-stone-800/90 dark:bg-[#18221D] hover:bg-stone-700 dark:hover:bg-[#1E2B25] text-stone-200 hover:text-white border border-stone-700 dark:border-[#27362E] hover:border-amber-500/50 transition cursor-pointer shadow-xs"
            aria-label={isDark ? 'Switch to Archival Light Mode' : 'Switch to Nocturnal Dark Mode'}
            title={isDark ? 'Switch to Archival Light Mode' : 'Switch to Nocturnal Dark Mode'}
          >
            {isDark ? (
              <>
                <Sun className="w-4 h-4 text-amber-400" />
                <span className="text-xs font-medium text-amber-300 hidden sm:inline">Light</span>
              </>
            ) : (
              <>
                <Moon className="w-4 h-4 text-emerald-400" />
                <span className="text-xs font-medium text-stone-300 hidden sm:inline">Dark</span>
              </>
            )}
          </button>

          {isAuthenticated && user ? (
            <div className="flex items-center gap-3">
              <Link
                to="/profile"
                className="flex items-center gap-2 text-xs text-stone-300 hover:text-amber-200 bg-stone-800 dark:bg-[#18221D] hover:bg-stone-700/80 dark:hover:bg-[#1E2B25] px-3 py-1.5 rounded-md border border-stone-700 dark:border-[#27362E] transition"
              >
                <UserIcon className="w-3.5 h-3.5 text-emerald-400" />
                <span className="font-medium">{user.username}</span>
              </Link>
              <button
                onClick={handleLogout}
                className="flex items-center gap-1.5 text-xs text-stone-400 hover:text-rose-300 p-1.5 rounded hover:bg-stone-800 dark:hover:bg-[#18221D] transition cursor-pointer"
                title="Log out"
              >
                <LogOut className="w-4 h-4" />
                <span className="hidden sm:inline">Logout</span>
              </button>
            </div>
          ) : (
            <div className="flex items-center gap-2">
              <Link
                to="/login"
                className="text-xs text-stone-300 hover:text-white px-3 py-1.5 rounded hover:bg-stone-800 dark:hover:bg-[#18221D] transition"
              >
                Log In
              </Link>
              <Link
                to="/register"
                className="text-xs font-medium text-emerald-950 bg-amber-400 hover:bg-amber-300 px-3 py-1.5 rounded shadow-xs transition"
              >
                Register
              </Link>
            </div>
          )}
        </div>
      </div>
    </header>
  );
};
