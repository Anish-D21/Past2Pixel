import React, { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import { api } from '../services/api';
import { ReadinessResponse, SessionSummary } from '../types/api';
import { StatusBadge } from '../components/StatusBadge';
import {
  Cpu,
  Database,
  Layers,
  Wand2,
  FileText,
  History,
  Activity,
  AlertTriangle,
  CheckCircle,
  ExternalLink,
  Sparkles,
} from 'lucide-react';

export const Dashboard: React.FC = () => {
  const [readiness, setReadiness] = useState<ReadinessResponse | null>(null);
  const [sessions, setSessions] = useState<SessionSummary[]>([]);
  const [isLoading, setIsLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const [readinessData, sessionsData] = await Promise.allSettled([
          api.system.getReadiness(),
          api.sessions.listMySessions(),
        ]);

        if (readinessData.status === 'fulfilled') {
          setReadiness(readinessData.value);
        }
        if (sessionsData.status === 'fulfilled') {
          setSessions(sessionsData.value);
        }
      } catch (err: any) {
        setError(err.message || 'Failed to load laboratory telemetry.');
      } finally {
        setIsLoading(false);
      }
    };

    fetchData();
    const interval = setInterval(fetchData, 8000);
    return () => clearInterval(interval);
  }, []);

  if (isLoading) {
    return (
      <div className="flex flex-col items-center justify-center min-h-[60vh] gap-3">
        <div className="w-10 h-10 border-3 border-emerald-800 border-t-transparent rounded-full animate-spin" />
        <p className="text-sm text-stone-600 font-medium">Gathering laboratory telemetry...</p>
      </div>
    );
  }

  const compute = readiness?.compute;
  const models = readiness?.models || {};
  const db = readiness?.database;
  const ocr = readiness?.ocr;

  return (
    <div className="space-y-8">
      {/* Page Title & Subtitle */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 pb-6 border-b border-stone-200">
        <div>
          <h1 className="font-serif-scholarly text-3xl font-bold tracking-tight text-stone-900">
            Ancient Manuscript AI Laboratory
          </h1>
          <p className="text-xs text-stone-500 mt-1">
            Deep Learning Folio Restoration, Damage Localization & Explainable AI Workspace
          </p>
        </div>
        <div className="flex items-center gap-3">
          <Link
            to="/restore"
            className="inline-flex items-center gap-2 px-4 py-2 bg-emerald-800 hover:bg-emerald-700 text-white text-xs font-semibold rounded-lg shadow-sm transition"
          >
            <Wand2 className="w-4 h-4" />
            <span>Restore Manuscript</span>
          </Link>
          <Link
            to="/ocr"
            className="inline-flex items-center gap-2 px-4 py-2 bg-stone-800 hover:bg-stone-700 text-stone-100 text-xs font-semibold rounded-lg shadow-sm transition"
          >
            <FileText className="w-4 h-4" />
            <span>OCR Extraction</span>
          </Link>
        </div>
      </div>

      {error && (
        <div className="p-4 rounded-lg bg-amber-50 border border-amber-200 text-amber-900 text-xs flex items-center gap-3">
          <AlertTriangle className="w-5 h-5 text-amber-600 shrink-0" />
          <span>{error}</span>
        </div>
      )}

      {/* Top Telemetry Grid */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        {/* Compute Device */}
        <div className="bg-white p-5 rounded-xl border border-stone-200 shadow-xs flex flex-col justify-between">
          <div>
            <div className="flex items-center justify-between text-xs text-stone-500 mb-2 font-medium">
              <span className="flex items-center gap-1.5">
                <Cpu className="w-4 h-4 text-emerald-700" />
                <span>Compute Infrastructure</span>
              </span>
              <StatusBadge status={compute?.cuda_available ? 'ONLINE' : 'CPU_FALLBACK'} size="sm" />
            </div>
            <div className="font-semibold text-base text-stone-900">
              {compute?.device_name || 'Generic Compute Device'}
            </div>
            <p className="text-xs text-stone-500 mt-1">
              {compute?.platform} • {compute?.logical_cpus} Logical Cores
            </p>
          </div>
          <div className="mt-4 pt-3 border-t border-stone-100 flex items-center justify-between text-[11px] text-stone-500 font-mono">
            <span>CUDA Acceleration:</span>
            <span className={compute?.cuda_available ? 'text-emerald-700 font-bold' : 'text-stone-600'}>
              {compute?.cuda_available ? 'Active (CUDA 12.8)' : 'Disabled'}
            </span>
          </div>
        </div>

        {/* Database Persistence */}
        <div className="bg-white p-5 rounded-xl border border-stone-200 shadow-xs flex flex-col justify-between">
          <div>
            <div className="flex items-center justify-between text-xs text-stone-500 mb-2 font-medium">
              <span className="flex items-center gap-1.5">
                <Database className="w-4 h-4 text-emerald-700" />
                <span>Data Persistence</span>
              </span>
              <StatusBadge status={db?.status || 'UNKNOWN'} size="sm" />
            </div>
            <div className="font-semibold text-base text-stone-900 capitalize">
              {db?.storage_mode === 'mongodb_atlas' ? 'MongoDB Atlas Cluster' : 'Local JSON Persistence'}
            </div>
            <p className="text-xs text-stone-500 mt-1">
              Database: <span className="font-mono text-stone-700">{db?.database_name || 'past2pixel'}</span>
            </p>
          </div>
          <div className="mt-4 pt-3 border-t border-stone-100 flex items-center justify-between text-[11px] text-stone-500">
            <span>Storage Reliability:</span>
            <span className="font-medium text-emerald-800">
              {db?.is_persistent ? 'Persistent Document Store' : 'In-Memory Transient'}
            </span>
          </div>
        </div>

        {/* OCR Subsystem */}
        <div className="bg-white p-5 rounded-xl border border-stone-200 shadow-xs flex flex-col justify-between">
          <div>
            <div className="flex items-center justify-between text-xs text-stone-500 mb-2 font-medium">
              <span className="flex items-center gap-1.5">
                <FileText className="w-4 h-4 text-emerald-700" />
                <span>Tesseract OCR Engine</span>
              </span>
              <StatusBadge status={ocr?.tesseract_available ? 'READY' : 'UNAVAILABLE'} size="sm" />
            </div>
            <div className="font-semibold text-base text-stone-900">
              {ocr?.tesseract_available ? `Tesseract ${ocr.tesseract_version}` : 'Tesseract Not Detected'}
            </div>
            <p className="text-xs text-stone-500 mt-1">
              {ocr?.has_devanagari_support
                ? 'Devanagari (san/hin) language packs loaded'
                : 'Devanagari OCR requires Tesseract installation'}
            </p>
          </div>
          <div className="mt-4 pt-3 border-t border-stone-100 flex items-center justify-between text-[11px] text-stone-500">
            <span>Status:</span>
            <span className={ocr?.tesseract_available ? 'text-emerald-700 font-medium' : 'text-stone-600'}>
              {ocr?.tesseract_available ? 'Ready for Folio Ingestion' : 'Setup guidance provided'}
            </span>
          </div>
        </div>
      </div>

      {/* Model Readiness Section */}
      <div className="space-y-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Layers className="w-5 h-5 text-emerald-800" />
            <h2 className="font-serif-scholarly text-xl font-bold text-stone-900">
              Deep Learning Model Registry
            </h2>
          </div>
          <span className="text-xs text-stone-500">
            Actual checkpoint status loaded from disk
          </span>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-5">
          {Object.entries(models).map(([key, model]) => {
            const isReady = model.status === 'READY_FOR_INFERENCE';
            const meta = model.training_metadata;
            return (
              <div
                key={key}
                className="bg-white p-5 rounded-xl border border-stone-200 shadow-xs flex flex-col justify-between hover:border-emerald-700/40 transition"
              >
                <div>
                  <div className="flex items-start justify-between gap-2 mb-3">
                    <h3 className="font-semibold text-sm text-stone-900 line-clamp-1">
                      {model.name}
                    </h3>
                    <StatusBadge status={model.status} size="sm" />
                  </div>

                  <div className="text-xs space-y-1.5 text-stone-600">
                    <div className="flex justify-between">
                      <span className="text-stone-400">Checkpoint:</span>
                      <span className="font-mono text-[11px] text-stone-700 line-clamp-1">
                        {model.checkpoint_file}
                      </span>
                    </div>

                    {meta ? (
                      <>
                        <div className="flex justify-between">
                          <span className="text-stone-400">Epochs:</span>
                          <span className="font-medium text-stone-800">{meta.epoch}</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-stone-400">Val Loss:</span>
                          <span className="font-medium text-stone-800">
                            {meta.val_loss?.toFixed(4)}
                          </span>
                        </div>
                        {meta.metrics?.val_psnr && (
                          <div className="flex justify-between">
                            <span className="text-stone-400">PSNR:</span>
                            <span className="font-semibold text-emerald-700">
                              {meta.metrics.val_psnr} dB
                            </span>
                          </div>
                        )}
                        {meta.metrics?.val_ssim && (
                          <div className="flex justify-between">
                            <span className="text-stone-400">SSIM:</span>
                            <span className="font-semibold text-emerald-700">
                              {meta.metrics.val_ssim}
                            </span>
                          </div>
                        )}
                        {meta.metrics?.val_iou && (
                          <div className="flex justify-between">
                            <span className="text-stone-400">IoU:</span>
                            <span className="font-semibold text-emerald-700">
                              {meta.metrics.val_iou}
                            </span>
                          </div>
                        )}
                        {meta.metrics?.val_dice && (
                          <div className="flex justify-between">
                            <span className="text-stone-400">Dice:</span>
                            <span className="font-semibold text-emerald-700">
                              {meta.metrics.val_dice}
                            </span>
                          </div>
                        )}
                      </>
                    ) : (
                      <div className="py-2 text-[11px] text-stone-400 italic">
                        {isReady
                          ? 'Model ready for evaluation'
                          : 'Training in progress or pending'}
                      </div>
                    )}
                  </div>
                </div>

                <div className="mt-4 pt-3 border-t border-stone-100 flex items-center justify-between text-[11px] text-stone-400">
                  <span>Architecture</span>
                  <span className="font-mono text-stone-600">PyTorch</span>
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* Recent Sessions List */}
      <div className="space-y-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <History className="w-5 h-5 text-emerald-800" />
            <h2 className="font-serif-scholarly text-xl font-bold text-stone-900">
              Recent Manuscript Processing Sessions
            </h2>
          </div>
          <Link
            to="/history"
            className="text-xs font-semibold text-emerald-800 hover:text-emerald-700 flex items-center gap-1"
          >
            <span>View All Ledger</span>
            <ExternalLink className="w-3.5 h-3.5" />
          </Link>
        </div>

        {sessions.length === 0 ? (
          <div className="bg-white p-8 rounded-xl border border-stone-200 text-center space-y-3">
            <p className="text-sm text-stone-600">No manuscript sessions recorded yet.</p>
            <Link
              to="/restore"
              className="inline-flex items-center gap-2 px-4 py-2 bg-emerald-800 hover:bg-emerald-700 text-white text-xs font-semibold rounded-lg shadow-sm transition"
            >
              <Wand2 className="w-4 h-4" />
              <span>Upload First Manuscript Folio</span>
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
                    <th className="px-4 py-3">Date</th>
                    <th className="px-4 py-3">Damage %</th>
                    <th className="px-4 py-3">Pipeline Status</th>
                    <th className="px-4 py-3 text-right">Actions</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-stone-200">
                  {sessions.slice(0, 5).map((s) => (
                    <tr key={s.session_id} className="hover:bg-stone-50/80 transition">
                      <td className="px-4 py-3 font-mono font-medium text-stone-800">
                        {s.session_id}
                      </td>
                      <td className="px-4 py-3 text-stone-700">{s.filename || 'manuscript.png'}</td>
                      <td className="px-4 py-3 text-stone-500">
                        {new Date(s.created_at).toLocaleDateString()}
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
                        <StatusBadge status={s.status} size="sm" />
                      </td>
                      <td className="px-4 py-3 text-right space-x-2">
                        <Link
                          to={`/results/${s.session_id}`}
                          className="font-semibold text-emerald-800 hover:text-emerald-700"
                        >
                          Analysis
                        </Link>
                        <Link
                          to={`/explainable-ai?session=${s.session_id}`}
                          className="font-medium text-stone-600 hover:text-stone-900"
                        >
                          XAI Report
                        </Link>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};
