import React, { useState, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import { api } from '../services/api';
import { RestorationResponse, UploadResponse } from '../types/api';
import { ImageComparisonSlider } from '../components/ImageComparisonSlider';
import { StatusBadge } from '../components/StatusBadge';
import { ArtifactImage } from '../components/ArtifactImage';
import {
  Upload,
  Wand2,
  AlertCircle,
  CheckCircle2,
  Download,
  FileText,
  GitBranch,
  Cpu,
  Layers,
  Sparkles,
  RefreshCw,
} from 'lucide-react';

export const RestoreManuscript: React.FC = () => {
  const navigate = useNavigate();
  const fileInputRef = useRef<HTMLInputElement>(null);

  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [uploadData, setUploadData] = useState<UploadResponse | null>(null);
  const [devicePref, setDevicePref] = useState<string>('auto');
  const [isProcessing, setIsProcessing] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<RestorationResponse | null>(null);

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (!e.target.files || e.target.files.length === 0) return;
    const file = e.target.files[0];

    // Reset previous run
    setError(null);
    setResult(null);
    setSelectedFile(file);

    // Create local object URL for instant preview
    const objUrl = URL.createObjectURL(file);
    setPreviewUrl(objUrl);
    setUploadData(null);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      const file = e.dataTransfer.files[0];
      setError(null);
      setResult(null);
      setSelectedFile(file);
      setPreviewUrl(URL.createObjectURL(file));
      setUploadData(null);
    }
  };

  const handleUploadAndRestore = async () => {
    if (!selectedFile) {
      setError('Please select or drop a manuscript image first.');
      return;
    }

    setIsProcessing(true);
    setError(null);

    try {
      // Step 1: Upload to FastAPI backend
      const upRes = await api.upload.uploadManuscript(selectedFile);
      setUploadData(upRes);

      // Step 2: Trigger Deep Learning Restoration Engine
      const restRes = await api.restore.processSession({
        session_id: upRes.session_id,
        device_preference: devicePref,
        enable_enhancement: true,
      });

      setResult(restRes);
    } catch (err: any) {
      setError(err.message || 'Restoration pipeline encountered a server failure.');
    } finally {
      setIsProcessing(false);
    }
  };

  return (
    <div className="space-y-8 max-w-6xl mx-auto">
      {/* Title */}
      <div className="pb-4 border-b border-stone-200">
        <h1 className="font-serif-scholarly text-3xl font-bold tracking-tight text-stone-900">
          Manuscript Restoration Workspace
        </h1>
        <p className="text-xs text-stone-500 mt-1">
          Paired deep neural restoration: U-Net feature skip-connections, damage localization & contrast enhancement
        </p>
      </div>

      {error && (
        <div className="p-4 rounded-lg bg-rose-50 border border-rose-200 text-rose-900 text-xs flex items-start gap-3">
          <AlertCircle className="w-5 h-5 text-rose-600 shrink-0 mt-0.5" />
          <div>
            <div className="font-semibold">Pipeline Execution Error</div>
            <div>{error}</div>
          </div>
        </div>
      )}

      {/* Upload & Controls Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left: Upload Dropzone */}
        <div className="lg:col-span-2 space-y-4">
          <div
            onDragOver={(e) => e.preventDefault()}
            onDrop={handleDrop}
            onClick={() => fileInputRef.current?.click()}
            className="border-2 border-dashed border-stone-300 hover:border-emerald-700/60 rounded-xl p-8 text-center cursor-pointer bg-white transition hover:bg-stone-50/50 flex flex-col items-center justify-center min-h-[280px]"
          >
            <input
              type="file"
              ref={fileInputRef}
              onChange={handleFileSelect}
              accept="image/png, image/jpeg, image/tiff, image/bmp"
              className="hidden"
            />

            {previewUrl ? (
              <div className="space-y-3">
                <img
                  src={previewUrl}
                  alt="Manuscript preview"
                  className="max-h-56 mx-auto rounded-lg shadow-xs border border-stone-200 object-contain"
                />
                <p className="text-xs text-stone-600 font-medium">
                  {selectedFile?.name} ({(selectedFile?.size! / (1024 * 1024)).toFixed(2)} MB)
                </p>
                <span className="text-[11px] text-emerald-800 underline">
                  Click or drop to replace folio image
                </span>
              </div>
            ) : (
              <div className="space-y-3">
                <div className="w-12 h-12 mx-auto rounded-full bg-emerald-50 text-emerald-800 flex items-center justify-center border border-emerald-200 shadow-xs">
                  <Upload className="w-6 h-6" />
                </div>
                <div>
                  <p className="text-sm font-semibold text-stone-800">
                    Upload Damaged Manuscript Folio
                  </p>
                  <p className="text-xs text-stone-500 mt-1">
                    Drag and drop high-resolution scans (PNG, JPG, TIFF, BMP up to 15 MB)
                  </p>
                </div>
                <button
                  type="button"
                  className="px-3 py-1.5 bg-stone-100 hover:bg-stone-200 text-stone-700 text-xs font-medium rounded-md border border-stone-300 transition"
                >
                  Browse Files
                </button>
              </div>
            )}
          </div>
        </div>

        {/* Right: Engine Parameters & Action */}
        <div className="bg-white p-6 rounded-xl border border-stone-200 shadow-xs flex flex-col justify-between space-y-6">
          <div className="space-y-4">
            <h2 className="font-serif-scholarly font-bold text-base text-stone-900 border-b border-stone-100 pb-2 flex items-center gap-2">
              <Layers className="w-4 h-4 text-emerald-800" />
              <span>Inference Parameters</span>
            </h2>

            <div>
              <label className="block text-xs font-medium text-stone-700 mb-1">
                Target Device Preference
              </label>
              <select
                value={devicePref}
                onChange={(e) => setDevicePref(e.target.value)}
                className="w-full text-xs border border-stone-300 rounded-lg p-2 bg-stone-50 focus:ring-2 focus:ring-emerald-700"
              >
                <option value="auto">Auto (Prefer GPU CUDA if available)</option>
                <option value="cuda">CUDA (Dedicated RTX GPU)</option>
                <option value="cpu">CPU (Torch Reference Execution)</option>
              </select>
            </div>

            <div className="text-xs text-stone-500 space-y-2 pt-2 border-t border-stone-100">
              <div className="flex items-center gap-2">
                <CheckCircle2 className="w-4 h-4 text-emerald-700 shrink-0" />
                <span>Damage Localization (U-Net BCE/Dice)</span>
              </div>
              <div className="flex items-center gap-2">
                <CheckCircle2 className="w-4 h-4 text-emerald-700 shrink-0" />
                <span>Primary Restoration (Deep U-Net)</span>
              </div>
              <div className="flex items-center gap-2">
                <CheckCircle2 className="w-4 h-4 text-emerald-700 shrink-0" />
                <span>Convolutional Autoencoder Baseline</span>
              </div>
              <div className="flex items-center gap-2">
                <CheckCircle2 className="w-4 h-4 text-emerald-700 shrink-0" />
                <span>Adaptive Contrast CLAHE Post-Processing</span>
              </div>
            </div>
          </div>

          <button
            type="button"
            onClick={handleUploadAndRestore}
            disabled={!selectedFile || isProcessing}
            className="w-full py-3 px-4 bg-emerald-800 hover:bg-emerald-700 text-white rounded-lg shadow-md text-sm font-semibold flex items-center justify-center gap-2 transition disabled:opacity-50 cursor-pointer"
          >
            {isProcessing ? (
              <>
                <RefreshCw className="w-4 h-4 animate-spin" />
                <span>Executing Neural Pipeline...</span>
              </>
            ) : (
              <>
                <Wand2 className="w-4 h-4" />
                <span>Execute Deep Restoration</span>
              </>
            )}
          </button>
        </div>
      </div>

      {/* Results Section */}
      {result && (
        <div className="space-y-6 pt-4 border-t border-stone-200">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
            <div>
              <div className="flex items-center gap-2">
                <Sparkles className="w-5 h-5 text-emerald-700" />
                <h2 className="font-serif-scholarly text-2xl font-bold text-stone-900">
                  Restoration Output & Analysis
                </h2>
              </div>
              <p className="text-xs text-stone-500 mt-1 font-mono">
                Session ID: {result.session_id} • Device: {result.device_info?.device_name}
              </p>
            </div>

            <div className="flex items-center gap-3">
              <a
                href={result.artifacts.restored || '#'}
                download={`restored_${result.session_id}.png`}
                className="inline-flex items-center gap-1.5 px-3 py-1.5 bg-stone-800 hover:bg-stone-700 text-white text-xs font-medium rounded-lg shadow-xs transition"
              >
                <Download className="w-3.5 h-3.5" />
                <span>Download Restored</span>
              </a>
              <button
                onClick={() => navigate(`/results/${result.session_id}`)}
                className="inline-flex items-center gap-1.5 px-3 py-1.5 bg-emerald-800 hover:bg-emerald-700 text-white text-xs font-semibold rounded-lg shadow-xs transition cursor-pointer"
              >
                <span>Full Metrics Analysis</span>
              </button>
            </div>
          </div>

          {/* Interactive Split Slider */}
          {result.artifacts.restored && (
            <div className="space-y-2">
              <h3 className="text-xs font-semibold uppercase tracking-wider text-stone-500">
                Interactive Before / After Comparison Slider
              </h3>
              <ImageComparisonSlider
                beforeImage={result.artifacts.original}
                afterImage={result.artifacts.restored}
                beforeLabel="Original Damaged Manuscript"
                afterLabel="Restored Folio (U-Net Deep Restoration)"
                className="h-[460px] w-full"
              />
            </div>
          )}

          {/* Artifacts Cards: Original, Damage Mask, Overlay, Restored */}
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
            {/* 1. Original */}
            <div className="bg-white p-3 rounded-xl border border-stone-200 space-y-2">
              <div className="text-xs font-semibold text-stone-700">1. Original Input</div>
              <ArtifactImage
                src={result.artifacts.original}
                alt="Original"
                fallbackLabel="Original Folio Unavailable"
              />
            </div>

            {/* 2. Damage Mask */}
            <div className="bg-white p-3 rounded-xl border border-stone-200 space-y-2">
              <div className="flex items-center justify-between text-xs font-semibold text-stone-700">
                <span>2. Detected Damage Mask</span>
                <span className="text-[11px] text-amber-700 font-mono">
                  {result.damage_detection?.damaged_pixel_percentage?.toFixed(1)}%
                </span>
              </div>
              <ArtifactImage
                src={result.artifacts.damage_mask}
                alt="Damage Mask"
                className="h-44 w-full object-contain rounded bg-stone-900 border border-stone-100"
                fallbackLabel="Damage Mask Unavailable"
              />
            </div>

            {/* 3. Damage Overlay */}
            <div className="bg-white p-3 rounded-xl border border-stone-200 space-y-2">
              <div className="text-xs font-semibold text-stone-700">3. Damage Heatmap Overlay</div>
              <ArtifactImage
                src={result.artifacts.damage_overlay}
                alt="Damage Overlay"
                fallbackLabel="Heatmap Overlay Unavailable"
              />
            </div>

            {/* 4. Restored Folio */}
            <div className="bg-white p-3 rounded-xl border border-stone-200 space-y-2">
              <div className="flex items-center justify-between text-xs font-semibold text-stone-700">
                <span>4. Restored Folio</span>
                <StatusBadge status={result.restoration?.model_readiness || 'READY'} size="sm" />
              </div>
              <ArtifactImage
                src={result.artifacts.restored}
                alt="Restored"
                fallbackLabel="Restored Folio Unavailable"
              />
            </div>
          </div>

          {/* Quick Action Navigation Links */}
          <div className="flex flex-wrap items-center gap-3 pt-4 border-t border-stone-200">
            <button
              onClick={() => navigate(`/ocr?session=${result.session_id}`)}
              className="inline-flex items-center gap-2 px-4 py-2 bg-stone-800 hover:bg-stone-700 text-white text-xs font-medium rounded-lg shadow-xs transition cursor-pointer"
            >
              <FileText className="w-4 h-4 text-emerald-400" />
              <span>Proceed to OCR & Text Extraction for this Folio</span>
            </button>
            <button
              onClick={() => navigate(`/explainable-ai?session=${result.session_id}`)}
              className="inline-flex items-center gap-2 px-4 py-2 bg-stone-100 hover:bg-stone-200 text-stone-800 text-xs font-medium rounded-lg border border-stone-300 transition cursor-pointer"
            >
              <GitBranch className="w-4 h-4 text-emerald-700" />
              <span>Inspect Explainable AI Pipeline Telemetry</span>
            </button>
          </div>
        </div>
      )}
    </div>
  );
};
