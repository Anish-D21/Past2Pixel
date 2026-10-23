import React, { useEffect, useState } from 'react';
import { useParams, Link } from 'react-router-dom';
import { api } from '../services/api';
import { SessionDetailResponse } from '../types/api';
import { StatusBadge } from '../components/StatusBadge';
import { ImageComparisonSlider } from '../components/ImageComparisonSlider';
import {
  Sparkles,
  Cpu,
  Clock,
  Layers,
  FileText,
  Download,
  AlertCircle,
  CheckCircle2,
  GitBranch,
} from 'lucide-react';

export const ResultsAnalysis: React.FC = () => {
  const { sessionId } = useParams<{ sessionId: string }>();

  const [session, setSession] = useState<SessionDetailResponse | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!sessionId) {
      setError('No session ID provided for analysis.');
      setIsLoading(false);
      return;
    }

    api.sessions
      .getSession(sessionId)
      .then((data) => setSession(data))
      .catch((err) => setError(err.message || 'Could not find session record.'))
      .finally(() => setIsLoading(false));
  }, [sessionId]);

  const handleDownloadJsonReport = () => {
    if (!session) return;
    const blob = new Blob([JSON.stringify(session, null, 2)], {
      type: 'application/json',
    });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `past2pixel_report_${session.session_id}.json`;
    a.click();
  };

  if (isLoading) {
    return (
      <div className="flex flex-col items-center justify-center min-h-[60vh] gap-3">
        <div className="w-10 h-10 border-3 border-emerald-800 border-t-transparent rounded-full animate-spin" />
        <p className="text-sm text-stone-600 font-medium">Loading session analysis...</p>
      </div>
    );
  }

  if (error || !session) {
    return (
      <div className="max-w-2xl mx-auto p-8 bg-white rounded-xl border border-stone-200 text-center space-y-4">
        <AlertCircle className="w-10 h-10 text-rose-600 mx-auto" />
        <h2 className="font-serif-scholarly text-xl font-bold text-stone-900">
          Session Analysis Unavailable
        </h2>
        <p className="text-xs text-stone-600">{error || 'Session record not found.'}</p>
        <Link
          to="/history"
          className="inline-block px-4 py-2 bg-emerald-800 text-white rounded-lg text-xs font-semibold"
        >
          Return to History Ledger
        </Link>
      </div>
    );
  }

  const res = session.restoration_result;
  const damage = res?.damage_detection;
  const restoration = res?.restoration;
  const timings = res?.timings_ms || {};
  const artifacts = session.artifacts || {};

  return (
    <div className="space-y-8 max-w-6xl mx-auto">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 pb-4 border-b border-stone-200">
        <div>
          <div className="flex items-center gap-3">
            <h1 className="font-serif-scholarly text-3xl font-bold tracking-tight text-stone-900">
              Restoration Analytical Report
            </h1>
            <StatusBadge status={session.status} size="md" />
          </div>
          <p className="text-xs text-stone-500 mt-1 font-mono">
            Session: {session.session_id} • Created: {new Date(session.created_at).toLocaleString()}
          </p>
        </div>

        <div className="flex items-center gap-3">
          <button
            onClick={handleDownloadJsonReport}
            className="inline-flex items-center gap-1.5 px-3 py-1.5 bg-stone-800 hover:bg-stone-700 text-white text-xs font-medium rounded-lg shadow-xs transition cursor-pointer"
          >
            <Download className="w-3.5 h-3.5" />
            <span>Export JSON Audit</span>
          </button>
          <Link
            to={`/explainable-ai?session=${session.session_id}`}
            className="inline-flex items-center gap-1.5 px-3 py-1.5 bg-emerald-800 hover:bg-emerald-700 text-white text-xs font-semibold rounded-lg shadow-xs transition"
          >
            <GitBranch className="w-3.5 h-3.5" />
            <span>Explainable AI View</span>
          </Link>
        </div>
      </div>

      {/* Main Comparison Slider */}
      {artifacts.original && artifacts.restored && (
        <div className="bg-white p-6 rounded-xl border border-stone-200 shadow-xs space-y-3">
          <h2 className="text-xs font-semibold uppercase tracking-wider text-stone-500">
            Interactive High-Fidelity Comparison
          </h2>
          <ImageComparisonSlider
            beforeImage={artifacts.original}
            afterImage={artifacts.restored}
            beforeLabel="Input Manuscript (Degraded)"
            afterLabel="Restored Folio (Neural Network)"
            className="h-[460px] w-full"
          />
        </div>
      )}

      {/* Performance & Metrics Grid */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        {/* Metric 1: Damage Localization */}
        <div className="bg-white p-5 rounded-xl border border-stone-200 shadow-xs space-y-3">
          <div className="text-xs font-medium text-stone-500 uppercase tracking-wider">
            Damage Localization
          </div>
          <div className="text-2xl font-serif-scholarly font-bold text-amber-700">
            {damage?.damaged_pixel_percentage !== undefined
              ? `${damage.damaged_pixel_percentage.toFixed(1)}%`
              : 'N/A'}
          </div>
          <p className="text-xs text-stone-600">
            Severity Assessment:{' '}
            <span className="font-semibold text-stone-800">
              {damage?.damage_severity || 'MODERATE'}
            </span>
          </p>
          <div className="pt-2 border-t border-stone-100 text-[11px] text-stone-400">
            Damaged Pixels: {damage?.damaged_pixels?.toLocaleString()} /{' '}
            {damage?.total_pixels?.toLocaleString()}
          </div>
        </div>

        {/* Metric 2: Reconstruction Quality */}
        <div className="bg-white p-5 rounded-xl border border-stone-200 shadow-xs space-y-3">
          <div className="text-xs font-medium text-stone-500 uppercase tracking-wider">
            Reconstruction Quality
          </div>
          <div className="text-2xl font-serif-scholarly font-bold text-emerald-800">
            {restoration?.evaluation_metrics?.psnr
              ? `${restoration.evaluation_metrics.psnr.toFixed(2)} dB`
              : 'Synthetic Paired Only'}
          </div>
          <p className="text-xs text-stone-600">
            SSIM Index:{' '}
            <span className="font-semibold text-stone-800">
              {restoration?.evaluation_metrics?.ssim
                ? restoration.evaluation_metrics.ssim.toFixed(4)
                : 'No clean ground-truth reference'}
            </span>
          </p>
          <div className="pt-2 border-t border-stone-100 text-[11px] text-stone-400">
            {restoration?.evaluation_metrics?.reason ||
              'Reference PSNR/SSIM requires paired ground-truth image'}
          </div>
        </div>

        {/* Metric 3: Inference Latencies */}
        <div className="bg-white p-5 rounded-xl border border-stone-200 shadow-xs space-y-3">
          <div className="text-xs font-medium text-stone-500 uppercase tracking-wider">
            Pipeline Latencies
          </div>
          <div className="text-2xl font-serif-scholarly font-bold text-stone-900">
            {timings.total_pipeline_ms
              ? `${timings.total_pipeline_ms.toFixed(0)} ms`
              : '—'}
          </div>
          <div className="space-y-1 text-[11px] text-stone-500">
            <div className="flex justify-between">
              <span>Damage Detector:</span>
              <span className="font-mono">{timings.damage_detector_ms?.toFixed(1) || 0} ms</span>
            </div>
            <div className="flex justify-between">
              <span>U-Net Restoration:</span>
              <span className="font-mono">{timings.unet_restoration_ms?.toFixed(1) || 0} ms</span>
            </div>
            <div className="flex justify-between">
              <span>CLAHE Enhancement:</span>
              <span className="font-mono">{timings.clahe_enhancement_ms?.toFixed(1) || 0} ms</span>
            </div>
          </div>
        </div>
      </div>

      {/* Artifact Gallery */}
      <div className="space-y-3">
        <h2 className="font-serif-scholarly font-bold text-lg text-stone-900">
          Generated Intermediate Folio Artifacts
        </h2>
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
          <div className="bg-white p-3 rounded-xl border border-stone-200 space-y-2">
            <div className="text-xs font-medium text-stone-700">1. Original Upload</div>
            <img
              src={artifacts.original}
              alt="Original"
              className="h-44 w-full object-contain rounded bg-stone-50 border border-stone-100"
            />
          </div>
          <div className="bg-white p-3 rounded-xl border border-stone-200 space-y-2">
            <div className="text-xs font-medium text-stone-700">2. Damage Binary Mask</div>
            <img
              src={artifacts.damage_mask || ''}
              alt="Damage Mask"
              className="h-44 w-full object-contain rounded bg-stone-900 border border-stone-100"
            />
          </div>
          <div className="bg-white p-3 rounded-xl border border-stone-200 space-y-2">
            <div className="text-xs font-medium text-stone-700">3. Damage Heatmap Overlay</div>
            <img
              src={artifacts.damage_overlay || ''}
              alt="Damage Overlay"
              className="h-44 w-full object-contain rounded bg-stone-50 border border-stone-100"
            />
          </div>
          <div className="bg-white p-3 rounded-xl border border-stone-200 space-y-2">
            <div className="text-xs font-medium text-stone-700">4. Final Restored Folio</div>
            <img
              src={artifacts.restored || ''}
              alt="Restored"
              className="h-44 w-full object-contain rounded bg-stone-50 border border-stone-100"
            />
          </div>
        </div>
      </div>
    </div>
  );
};
