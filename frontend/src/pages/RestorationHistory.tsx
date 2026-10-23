import React, { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import { api } from '../services/api';
import { SessionSummary } from '../types/api';
import { StatusBadge } from '../components/StatusBadge';
import {
  History,
  Trash2,
  ExternalLink,
  Search,
  AlertCircle,
  FileText,
  Wand2,
  RefreshCw,
} from 'lucide-react';

export const RestorationHistory: React.FC = () => {
  const [sessions, setSessions] = useState<SessionSummary[]>([]);
  const [isLoading, setIsLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const [search, setSearch] = useState<string>('');
  const [deletingId, setDeletingId] = useState<string | null>(null);

  const fetchSessions = async () => {
    setIsLoading(true);
    setError(null);
    try {
      const data = await api.sessions.listMySessions();
      setSessions(data);
    } catch (err: any) {
      setError(err.message || 'Failed to load historical manuscript ledger.');
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    fetchSessions();
  }, []);

  const handleDelete = async (sessionId: string) => {
    if (!window.confirm(`Are you sure you want to permanently delete session '${sessionId}'?`)) {
      return;
    }

    setDeletingId(sessionId);
    try {
      await api.sessions.deleteSession(sessionId);
      setSessions((prev) => prev.filter((s) => s.session_id !== sessionId));
    } catch (err: any) {
      alert(`Delete failed: ${err.message}`);
    } finally {
      setDeletingId(null);
    }
  };

  const filteredSessions = sessions.filter(
    (s) =>
      s.session_id.toLowerCase().includes(search.toLowerCase()) ||
      (s.filename && s.filename.toLowerCase().includes(search.toLowerCase()))
  );

  return (
    <div className="space-y-6 max-w-6xl mx-auto">
      {/* Title */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 pb-4 border-b border-stone-200">
        <div>
          <div className="flex items-center gap-2">
            <History className="w-6 h-6 text-emerald-800" />
            <h1 className="font-serif-scholarly text-3xl font-bold tracking-tight text-stone-900">
              Restoration History Ledger
            </h1>
          </div>
          <p className="text-xs text-stone-500 mt-1">
            Historical manuscript preservation sessions stored in your authenticated archive
          </p>
        </div>

        <div className="flex items-center gap-3">
          <button
            onClick={fetchSessions}
            className="inline-flex items-center gap-1.5 px-3 py-1.5 bg-stone-100 hover:bg-stone-200 text-stone-700 text-xs font-medium rounded-lg border border-stone-300 transition cursor-pointer"
          >
            <RefreshCw className="w-3.5 h-3.5" />
            <span>Refresh Ledger</span>
          </button>
          <Link
            to="/restore"
            className="inline-flex items-center gap-1.5 px-3 py-1.5 bg-emerald-800 hover:bg-emerald-700 text-white text-xs font-semibold rounded-lg shadow-xs transition"
          >
            <Wand2 className="w-3.5 h-3.5" />
            <span>New Restoration</span>
          </Link>
        </div>
      </div>

      {error && (
        <div className="p-4 rounded-lg bg-rose-50 border border-rose-200 text-rose-900 text-xs flex items-center gap-3">
          <AlertCircle className="w-5 h-5 text-rose-600 shrink-0" />
          <span>{error}</span>
        </div>
      )}

      {/* Filter / Search Bar */}
      <div className="flex items-center gap-3 bg-white p-3 rounded-xl border border-stone-200 shadow-xs">
        <Search className="w-4 h-4 text-stone-400 shrink-0 ml-2" />
        <input
          type="text"
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          placeholder="Filter by session ID or original filename..."
          className="w-full text-xs bg-transparent border-none focus:outline-hidden text-stone-800 placeholder-stone-400"
        />
      </div>

      {/* Sessions Table */}
      {isLoading ? (
        <div className="flex flex-col items-center justify-center min-h-[40vh] gap-3">
          <div className="w-8 h-8 border-3 border-emerald-800 border-t-transparent rounded-full animate-spin" />
          <p className="text-xs text-stone-500 font-medium">Querying ledger records...</p>
        </div>
      ) : filteredSessions.length === 0 ? (
        <div className="bg-white p-12 rounded-xl border border-stone-200 text-center space-y-3">
          <History className="w-10 h-10 text-stone-300 mx-auto" />
          <h3 className="font-serif-scholarly text-base font-bold text-stone-800">
            No Historical Sessions Recorded
          </h3>
          <p className="text-xs text-stone-500 max-w-sm mx-auto">
            Uploaded manuscript folios and their analytical results will appear here chronologically.
          </p>
          <Link
            to="/restore"
            className="inline-block mt-2 px-4 py-2 bg-emerald-800 text-white rounded-lg text-xs font-semibold shadow-xs"
          >
            Start First Restoration
          </Link>
        </div>
      ) : (
        <div className="bg-white rounded-xl border border-stone-200 shadow-xs overflow-hidden">
          <div className="overflow-x-auto">
            <table className="min-w-full divide-y divide-stone-200 text-left text-xs">
              <thead className="bg-stone-50 text-stone-500 font-medium">
                <tr>
                  <th className="px-4 py-3">Session ID</th>
                  <th className="px-4 py-3">Filename</th>
                  <th className="px-4 py-3">Date Created</th>
                  <th className="px-4 py-3">Damage %</th>
                  <th className="px-4 py-3">OCR Status</th>
                  <th className="px-4 py-3">Lifecycle State</th>
                  <th className="px-4 py-3 text-right">Actions</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-stone-200">
                {filteredSessions.map((s) => (
                  <tr key={s.session_id} className="hover:bg-stone-50/70 transition">
                    <td className="px-4 py-3 font-mono font-medium text-stone-900">
                      {s.session_id}
                    </td>
                    <td className="px-4 py-3 text-stone-700 font-medium">
                      {s.filename || 'manuscript.png'}
                    </td>
                    <td className="px-4 py-3 text-stone-500">
                      {new Date(s.created_at).toLocaleString()}
                    </td>
                    <td className="px-4 py-3 font-medium">
                      {s.damaged_percentage !== undefined && s.damaged_percentage !== null ? (
                        <span className="text-amber-700">
                          {s.damaged_percentage.toFixed(1)}%
                        </span>
                      ) : (
                        <span className="text-stone-400">—</span>
                      )}
                    </td>
                    <td className="px-4 py-3">
                      {s.has_ocr ? (
                        <span className="inline-flex items-center gap-1 text-[11px] font-medium text-emerald-800 bg-emerald-50 px-2 py-0.5 rounded border border-emerald-200">
                          <FileText className="w-3 h-3" />
                          <span>Extracted</span>
                        </span>
                      ) : (
                        <span className="text-stone-400 text-[11px]">Pending</span>
                      )}
                    </td>
                    <td className="px-4 py-3">
                      <StatusBadge status={s.status} size="sm" />
                    </td>
                    <td className="px-4 py-3 text-right space-x-2">
                      <Link
                        to={`/results/${s.session_id}`}
                        className="font-semibold text-emerald-800 hover:text-emerald-700"
                      >
                        Inspect
                      </Link>
                      <Link
                        to={`/explainable-ai?session=${s.session_id}`}
                        className="font-medium text-stone-600 hover:text-stone-900"
                      >
                        XAI Report
                      </Link>
                      <button
                        onClick={() => handleDelete(s.session_id)}
                        disabled={deletingId === s.session_id}
                        className="text-stone-400 hover:text-rose-600 p-1 transition cursor-pointer"
                        title="Delete Session"
                      >
                        <Trash2 className="w-3.5 h-3.5 inline" />
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
};
