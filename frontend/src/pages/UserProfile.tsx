import React from 'react';
import { useAuth } from '../context/AuthContext';
import { User, ShieldCheck, Key, Database, LogOut, Calendar } from 'lucide-react';

export const UserProfile: React.FC = () => {
  const { user, logout } = useAuth();

  return (
    <div className="space-y-6 max-w-3xl mx-auto">
      <div className="pb-4 border-b border-stone-200">
        <h1 className="font-serif-scholarly text-3xl font-bold tracking-tight text-stone-900">
          Scholar Profile & Security
        </h1>
        <p className="text-xs text-stone-500 mt-1">
          Researcher credentials, active JWT session state & persistence settings
        </p>
      </div>

      <div className="bg-white p-6 rounded-xl border border-stone-200 shadow-xs space-y-6">
        <div className="flex items-center gap-4">
          <div className="w-16 h-16 rounded-full bg-emerald-800 text-emerald-200 flex items-center justify-center font-serif-scholarly text-2xl font-bold border border-emerald-600 shadow-sm">
            {user?.username?.charAt(0).toUpperCase() || 'S'}
          </div>
          <div>
            <h2 className="font-serif-scholarly text-xl font-bold text-stone-900">
              {user?.username}
            </h2>
            <p className="text-xs text-stone-500">{user?.email}</p>
            <div className="flex items-center gap-1.5 mt-1 text-[11px] text-emerald-700 font-medium">
              <ShieldCheck className="w-3.5 h-3.5" />
              <span>Verified Academic Researcher</span>
            </div>
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 pt-4 border-t border-stone-100 text-xs">
          <div className="p-3 bg-stone-50 rounded-lg border border-stone-200 space-y-1">
            <div className="text-stone-400 font-medium flex items-center gap-1.5">
              <User className="w-3.5 h-3.5" />
              <span>User ID</span>
            </div>
            <div className="font-mono text-stone-800 font-semibold">{user?.id || '—'}</div>
          </div>

          <div className="p-3 bg-stone-50 rounded-lg border border-stone-200 space-y-1">
            <div className="text-stone-400 font-medium flex items-center gap-1.5">
              <Calendar className="w-3.5 h-3.5" />
              <span>Registration Date</span>
            </div>
            <div className="text-stone-800 font-semibold">
              {user?.created_at ? new Date(user.created_at).toLocaleDateString() : 'Active Session'}
            </div>
          </div>

          <div className="p-3 bg-stone-50 rounded-lg border border-stone-200 space-y-1">
            <div className="text-stone-400 font-medium flex items-center gap-1.5">
              <Key className="w-3.5 h-3.5" />
              <span>Authentication Token</span>
            </div>
            <div className="text-stone-800 font-semibold">JWT Bearer (HMAC-SHA256)</div>
          </div>

          <div className="p-3 bg-stone-50 rounded-lg border border-stone-200 space-y-1">
            <div className="text-stone-400 font-medium flex items-center gap-1.5">
              <Database className="w-3.5 h-3.5" />
              <span>Persistence Layer</span>
            </div>
            <div className="text-stone-800 font-semibold">MongoDB Atlas / JSON Fallback</div>
          </div>
        </div>

        <div className="pt-4 border-t border-stone-100 flex justify-end">
          <button
            onClick={logout}
            className="inline-flex items-center gap-2 px-4 py-2 bg-rose-50 hover:bg-rose-100 text-rose-700 text-xs font-semibold rounded-lg border border-rose-200 transition cursor-pointer"
          >
            <LogOut className="w-4 h-4" />
            <span>Sign Out of Laboratory</span>
          </button>
        </div>
      </div>
    </div>
  );
};
