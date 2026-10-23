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

  const [activeSliderCandidate, setActiveSliderCandidate] = useState<'restored' | 'baseline' | 'model'>('restored');

  const getSliderAfterImage = () => {
    if (!result) return '';
    if (activeSliderCandidate === 'baseline' && result.artifacts.baseline) {
      return result.artifacts.baseline;
    }
    if (activeSliderCandidate === 'model' && result.artifacts.model) {
      return result.artifacts.model;
    }
    return result.artifacts.restored || '';
  };

  const getSliderAfterLabel = () => {
    if (activeSliderCandidate === 'baseline') return 'Traditional Enhancement Baseline';
    if (activeSliderCandidate === 'model') return 'Deep Neural Model Output';
    return `Final Selected Folio (${result?.candidate_selection?.selected_candidate || 'Text-Protected'})`;
  };

  return (
    <div className="space-y-8 max-w-6xl mx-auto">
      {/* Title */}
      <div className="pb-4 border-b border-stone-200">
        <h1 className="font-serif-scholarly text-3xl font-bold tracking-tight text-stone-900">
          Manuscript Restoration Workspace
        </h1>
        <p className="text-xs text-stone-500 mt-1">
          Conservative text-aware neural restoration: Sauvola ink protection, traditional illumination baseline & safe candidate selection
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
                <span>Text-Aware Ink Protection Mask</span>
              </div>
              <div className="flex items-center gap-2">
                <CheckCircle2 className="w-4 h-4 text-emerald-700 shrink-0" />
                <span>Illumination Normalization Baseline</span>
              </div>
              <div className="flex items-center gap-2">
                <CheckCircle2 className="w-4 h-4 text-emerald-700 shrink-0" />
                <span>Fully-Convolutional Native U-Net</span>
              </div>
              <div className="flex items-center gap-2">
                <CheckCircle2 className="w-4 h-4 text-emerald-700 shrink-0" />
                <span>Objective Quality Evidence Gating</span>
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
                <span>Executing Text-Aware Pipeline...</span>
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

          {/* Candidate Selection & Text Protection Decision Banner */}
          {result.candidate_selection && (
            <div className="p-4 rounded-xl border border-emerald-200 bg-emerald-50/60 text-stone-800 space-y-3">
              <div className="flex flex-col md:flex-row md:items-center justify-between gap-2 border-b border-emerald-200/80 pb-2">
                <div className="flex items-center gap-2">
                  <span className="text-xs font-semibold uppercase tracking-wider text-emerald-900">
                    Safe Candidate Selection:
                  </span>
                  <span className="px-2.5 py-0.5 rounded-full text-xs font-bold uppercase tracking-wider bg-emerald-800 text-white">
                    {result.candidate_selection.selected_candidate}
                  </span>
                </div>
                {result.text_protection && (
                  <div className="text-xs text-stone-600 flex items-center gap-3 font-mono">
                    <span>
                      Ink Density: {(result.text_protection.ink_pixel_ratio * 100).toFixed(1)}%
                    </span>
                    <span>•</span>
                    <span>
                      Stroke Protection: Active
                    </span>
                  </div>
                )}
              </div>

              <p className="text-xs text-stone-700 leading-relaxed">
                <strong className="text-stone-900 font-semibold">Selection Rationale: </strong>
                {result.candidate_selection.selection_reason}
              </p>

              {/* Comparative Scores Grid */}
              {result.candidate_selection.candidates_evaluated && (
                <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 pt-2">
                  {Object.entries(result.candidate_selection.candidates_evaluated).map(([cand, metrics]) => {
                    const isSelected = cand === result.candidate_selection?.selected_candidate;
                    return (
                      <div
                        key={cand}
                        className={`p-3 rounded-lg border text-xs space-y-1.5 transition ${
                          isSelected
                            ? 'bg-white border-emerald-500 shadow-xs ring-1 ring-emerald-500'
                            : 'bg-stone-50/80 border-stone-200 text-stone-600'
                        }`}
                      >
                        <div className="flex items-center justify-between font-semibold">
                          <span className="uppercase text-[11px] font-mono tracking-wider">
                            {cand.replace('_', ' ')}
                          </span>
                          {isSelected && (
                            <span className="text-[10px] text-emerald-700 font-bold">Selected</span>
                          )}
                        </div>
                        <div className="space-y-0.5 font-mono text-[11px]">
                          <div className="flex justify-between">
                            <span className="text-stone-500">Composite Score:</span>
                            <span className="font-bold text-stone-800">{metrics.composite_score.toFixed(3)}</span>
                          </div>
                          <div className="flex justify-between">
                            <span className="text-stone-500">Text Contrast:</span>
                            <span>{metrics.text_contrast.toFixed(3)}</span>
                          </div>
                          <div className="flex justify-between">
                            <span className="text-stone-500">Stroke Sharpness:</span>
                            <span>{metrics.stroke_sharpness.toFixed(3)}</span>
                          </div>
                          <div className="flex justify-between">
                            <span className="text-stone-500">Background Noise:</span>
                            <span>{metrics.background_noise_std.toFixed(2)}</span>
                          </div>
                        </div>
                      </div>
                    );
                  })}
                </div>
              )}
            </div>
          )}

          {/* Interactive Split Slider with Candidate Tabs */}
          {result.artifacts.restored && (
            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <h3 className="text-xs font-semibold uppercase tracking-wider text-stone-500">
                  Interactive Before / After Comparison Slider
                </h3>
                <div className="flex items-center gap-1 bg-stone-100 p-1 rounded-lg border border-stone-200 text-xs">
                  <button
                    type="button"
                    onClick={() => setActiveSliderCandidate('restored')}
                    className={`px-2.5 py-1 rounded transition cursor-pointer font-medium ${
                      activeSliderCandidate === 'restored'
                        ? 'bg-white text-emerald-800 font-bold shadow-xs'
                        : 'text-stone-600 hover:text-stone-900'
                    }`}
                  >
                    Selected Restoration
                  </button>
                  {result.artifacts.baseline && (
                    <button
                      type="button"
                      onClick={() => setActiveSliderCandidate('baseline')}
                      className={`px-2.5 py-1 rounded transition cursor-pointer font-medium ${
                        activeSliderCandidate === 'baseline'
                          ? 'bg-white text-emerald-800 font-bold shadow-xs'
                          : 'text-stone-600 hover:text-stone-900'
                      }`}
                    >
                      Traditional Baseline
                    </button>
                  )}
                  {result.artifacts.model && (
                    <button
                      type="button"
                      onClick={() => setActiveSliderCandidate('model')}
                      className={`px-2.5 py-1 rounded transition cursor-pointer font-medium ${
                        activeSliderCandidate === 'model'
                          ? 'bg-white text-emerald-800 font-bold shadow-xs'
                          : 'text-stone-600 hover:text-stone-900'
                      }`}
                    >
                      Neural Model Output
                    </button>
                  )}
                </div>
              </div>

              <ImageComparisonSlider
                beforeImage={result.artifacts.original}
                afterImage={getSliderAfterImage()}
                beforeLabel="Original Damaged Manuscript"
                afterLabel={getSliderAfterLabel()}
                className="h-[460px] w-full"
              />
            </div>
          )}

          {/* Comprehensive 6-Artifact Inspection Grid */}
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
            {/* 1. Original Input */}
            <div className="bg-white p-3 rounded-xl border border-stone-200 space-y-2">
              <div className="text-xs font-semibold text-stone-700">1. Original Folio Input</div>
              <ArtifactImage
                src={result.artifacts.original}
                alt="Original"
                fallbackLabel="Original Folio Unavailable"
              />
            </div>

            {/* 2. Text Protection Ink Mask */}
            <div className="bg-white p-3 rounded-xl border border-stone-200 space-y-2">
              <div className="flex items-center justify-between text-xs font-semibold text-stone-700">
                <span>2. Text Protection Ink Mask</span>
                {result.text_protection && (
                  <span className="text-[11px] text-emerald-800 font-mono">
                    {(result.text_protection.ink_pixel_ratio * 100).toFixed(1)}% ink
                  </span>
                )}
              </div>
              <ArtifactImage
                src={result.artifacts.text_mask || ''}
                alt="Text Mask"
                className="h-44 w-full object-contain rounded bg-stone-900 border border-stone-100"
                fallbackLabel="Text Mask Unavailable"
              />
            </div>

            {/* 3. Damage Mask */}
            <div className="bg-white p-3 rounded-xl border border-stone-200 space-y-2">
              <div className="flex items-center justify-between text-xs font-semibold text-stone-700">
                <span>3. Damage Localization Mask</span>
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

            {/* 4. Traditional Baseline */}
            <div className="bg-white p-3 rounded-xl border border-stone-200 space-y-2">
              <div className="text-xs font-semibold text-stone-700">4. Traditional Baseline</div>
              <ArtifactImage
                src={result.artifacts.baseline || ''}
                alt="Baseline Restoration"
                fallbackLabel="Baseline Unavailable"
              />
            </div>

            {/* 5. Deep Neural Model Output */}
            <div className="bg-white p-3 rounded-xl border border-stone-200 space-y-2">
              <div className="text-xs font-semibold text-stone-700">5. Neural Model Output</div>
              <ArtifactImage
                src={result.artifacts.model || ''}
                alt="Model Output"
                fallbackLabel="Model Output Unavailable"
              />
            </div>

            {/* 6. Selected Restored Folio */}
            <div className="bg-white p-3 rounded-xl border border-stone-200 space-y-2">
              <div className="flex items-center justify-between text-xs font-semibold text-stone-700">
                <span>6. Final Selected Restored Folio</span>
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
