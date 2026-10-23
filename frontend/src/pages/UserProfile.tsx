import React from 'react';
import { useAuth } from '../context/AuthContext';
import { useTheme } from '../context/ThemeContext';
import { User, ShieldCheck, Key, Database, LogOut, Calendar, Sun, Moon, Palette, Check } from 'lucide-react';

export const UserProfile: React.FC = () => {
  const { user, logout } = useAuth();
  const { theme, setTheme } = useTheme();

  return (
    <div className="space-y-6 max-w-3xl mx-auto">
      <div className="pb-4 border-b border-stone-200 dark:border-[#27362E]">
        <h1 className="font-serif-scholarly text-3xl font-bold tracking-tight text-stone-900 dark:text-stone-100">
          Scholar Profile & Security
        </h1>
        <p className="text-xs text-stone-500 dark:text-stone-400 mt-1">
          Researcher credentials, active JWT session state & persistence settings
        </p>
      </div>

      <div className="bg-white dark:bg-[#18221D] p-6 rounded-xl border border-stone-200 dark:border-[#27362E] shadow-xs space-y-6 transition-colors duration-200">
        <div className="flex items-center gap-4">
          <div className="w-16 h-16 rounded-full bg-emerald-800 dark:bg-emerald-900 text-emerald-200 flex items-center justify-center font-serif-scholarly text-2xl font-bold border border-emerald-600 dark:border-emerald-700 shadow-sm">
            {user?.username?.charAt(0).toUpperCase() || 'S'}
          </div>
          <div>
            <h2 className="font-serif-scholarly text-xl font-bold text-stone-900 dark:text-stone-100">
              {user?.username}
            </h2>
            <p className="text-xs text-stone-500 dark:text-stone-400">{user?.email}</p>
            <div className="flex items-center gap-1.5 mt-1 text-[11px] text-emerald-700 dark:text-emerald-400 font-medium">
              <ShieldCheck className="w-3.5 h-3.5" />
              <span>Verified Academic Researcher</span>
            </div>
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 pt-4 border-t border-stone-100 dark:border-[#27362E] text-xs">
          <div className="p-3 bg-stone-50 dark:bg-[#141A17] rounded-lg border border-stone-200 dark:border-[#27362E] space-y-1">
            <div className="text-stone-400 font-medium flex items-center gap-1.5">
              <User className="w-3.5 h-3.5" />
              <span>User ID</span>
            </div>
            <div className="font-mono text-stone-800 dark:text-stone-200 font-semibold">{user?.id || '—'}</div>
          </div>

          <div className="p-3 bg-stone-50 dark:bg-[#141A17] rounded-lg border border-stone-200 dark:border-[#27362E] space-y-1">
            <div className="text-stone-400 font-medium flex items-center gap-1.5">
              <Calendar className="w-3.5 h-3.5" />
              <span>Registration Date</span>
            </div>
            <div className="text-stone-800 dark:text-stone-200 font-semibold">
              {user?.created_at ? new Date(user.created_at).toLocaleDateString() : 'Active Session'}
            </div>
          </div>

          <div className="p-3 bg-stone-50 dark:bg-[#141A17] rounded-lg border border-stone-200 dark:border-[#27362E] space-y-1">
            <div className="text-stone-400 font-medium flex items-center gap-1.5">
              <Key className="w-3.5 h-3.5" />
              <span>Authentication Token</span>
            </div>
            <div className="text-stone-800 dark:text-stone-200 font-semibold">JWT Bearer (HMAC-SHA256)</div>
          </div>

          <div className="p-3 bg-stone-50 dark:bg-[#141A17] rounded-lg border border-stone-200 dark:border-[#27362E] space-y-1">
            <div className="text-stone-400 font-medium flex items-center gap-1.5">
              <Database className="w-3.5 h-3.5" />
              <span>Persistence Layer</span>
            </div>
            <div className="text-stone-800 dark:text-stone-200 font-semibold">MongoDB Atlas / JSON Fallback</div>
          </div>
        </div>

        {/* Visual Theme Preferences */}
        <div className="pt-4 border-t border-stone-100 dark:border-[#27362E] space-y-3">
          <div className="flex items-center gap-2 text-xs font-semibold text-stone-800 dark:text-stone-200">
            <Palette className="w-4 h-4 text-emerald-700 dark:text-emerald-400" />
            <span>Visual Theme & Interface Appearance</span>
          </div>
          <p className="text-xs text-stone-500 dark:text-stone-400">
            Choose your preferred laboratory workspace styling. Preferences persist across browsing sessions and logins.
          </p>

          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 pt-1">
            {/* Light Mode Option */}
            <button
              type="button"
              onClick={() => setTheme('light')}
              className={`p-4 rounded-xl border text-left flex items-start gap-3 transition cursor-pointer ${
                theme === 'light'
                  ? 'bg-amber-50/70 border-amber-600 ring-2 ring-amber-500/20 shadow-xs'
                  : 'bg-stone-50 dark:bg-[#141A17] border-stone-200 dark:border-[#27362E] hover:border-stone-400'
              }`}
            >
              <div className="p-2 rounded-lg bg-amber-100 text-amber-800 shrink-0">
                <Sun className="w-5 h-5" />
              </div>
              <div className="flex-1">
                <div className="flex items-center justify-between">
                  <span className="font-semibold text-xs text-stone-900 dark:text-stone-100">
                    Archival Light Mode
                  </span>
                  {theme === 'light' && <Check className="w-4 h-4 text-emerald-700" />}
                </div>
                <p className="text-[11px] text-stone-500 dark:text-stone-400 mt-0.5">
                  Warm parchment tones, sepia grid, and traditional academic contrast.
                </p>
              </div>
            </button>

            {/* Dark Mode Option */}
            <button
              type="button"
              onClick={() => setTheme('dark')}
              className={`p-4 rounded-xl border text-left flex items-start gap-3 transition cursor-pointer ${
                theme === 'dark'
                  ? 'bg-[#18221D] border-emerald-500 ring-2 ring-emerald-500/30 shadow-xs'
                  : 'bg-stone-50 dark:bg-[#141A17] border-stone-200 dark:border-[#27362E] hover:border-emerald-700/50'
              }`}
            >
              <div className="p-2 rounded-lg bg-emerald-950 text-emerald-400 border border-emerald-800/80 shrink-0">
                <Moon className="w-5 h-5" />
              </div>
              <div className="flex-1">
                <div className="flex items-center justify-between">
                  <span className="font-semibold text-xs text-stone-900 dark:text-stone-100">
                    Nocturnal Dark Mode
                  </span>
                  {theme === 'dark' && <Check className="w-4 h-4 text-emerald-400" />}
                </div>
                <p className="text-[11px] text-stone-500 dark:text-stone-400 mt-0.5">
                  Deep charcoal/forest-green background with elegant emerald accents.
                </p>
              </div>
            </button>
          </div>
        </div>

        <div className="pt-4 border-t border-stone-100 dark:border-[#27362E] flex justify-end">
          <button
            onClick={logout}
            className="inline-flex items-center gap-2 px-4 py-2 bg-rose-50 hover:bg-rose-100 dark:bg-rose-950/60 dark:hover:bg-rose-900/60 text-rose-700 dark:text-rose-300 text-xs font-semibold rounded-lg border border-rose-200 dark:border-rose-900/80 transition cursor-pointer"
          >
            <LogOut className="w-4 h-4" />
            <span>Sign Out of Laboratory</span>
          </button>
        </div>
      </div>
    </div>
  );
};
