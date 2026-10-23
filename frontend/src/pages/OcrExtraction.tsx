import React, { useState, useEffect, useRef } from 'react';
import { useSearchParams, Link } from 'react-router-dom';
import { api } from '../services/api';
import { OCRResult, OCRStatusInfo, TextPredictionCandidate } from '../types/api';
import { StatusBadge } from '../components/StatusBadge';
import {
  FileText,
  Copy,
  Check,
  AlertTriangle,
  Sparkles,
  RefreshCw,
  Terminal,
  Upload,
  Image as ImageIcon,
  CheckCircle2,
  Layers,
  Cpu,
  History,
} from 'lucide-react';

export const OcrExtraction: React.FC = () => {
  const [searchParams] = useSearchParams();
  const sessionFromQuery = searchParams.get('session') || '';

  const [inputMode, setInputMode] = useState<'session' | 'direct'>('session');
  const [sessionId, setSessionId] = useState<string>(sessionFromQuery);
  const [language, setLanguage] = useState<string>('san+hin');
  const [targetImage, setTargetImage] = useState<'restored' | 'original'>('restored');
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [filePreview, setFilePreview] = useState<string | null>(null);

  const [engineStatus, setEngineStatus] = useState<OCRStatusInfo | null>(null);
  const [isLoadingStatus, setIsLoadingStatus] = useState<boolean>(true);
  const [isProcessing, setIsProcessing] = useState<boolean>(false);
  const [ocrResult, setOcrResult] = useState<OCRResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [copied, setCopied] = useState<boolean>(false);

  // Contextual Prediction State
  const [leftContext, setLeftContext] = useState<string>('अग्निमीळे पुरोहितं यज्ञस्य देव');
  const [rightContext, setRightContext] = useState<string>('रत्नधातमम् ॥');
  const [isPredicting, setIsPredicting] = useState<boolean>(false);
  const [predictionCandidates, setPredictionCandidates] = useState<TextPredictionCandidate[]>([]);
  const [predictionDisclaimer, setPredictionDisclaimer] = useState<string | null>(null);

  const fileInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (sessionFromQuery) {
      setSessionId(sessionFromQuery);
      setInputMode('session');
    }
  }, [sessionFromQuery]);

  // Fetch engine readiness on mount
  useEffect(() => {
    let isMounted = true;
    const fetchStatus = async () => {
      setIsLoadingStatus(true);
      try {
        const res = await api.ocr.getStatus();
        if (isMounted) {
          setEngineStatus(res);
        }
      } catch (err: any) {
        console.warn('Failed to fetch OCR engine status:', err);
      } finally {
        if (isMounted) {
          setIsLoadingStatus(false);
        }
      }
    };
    fetchStatus();
    return () => {
      isMounted = false;
    };
  }, []);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      setSelectedFile(file);
      setFilePreview(URL.createObjectURL(file));
      setError(null);
    }
  };

  const handleExtractOcr = async () => {
    setError(null);
    setOcrResult(null);

    if (inputMode === 'session') {
      if (!sessionId.trim()) {
        setError('Please provide a valid session ID containing an uploaded manuscript folio.');
        return;
      }

      setIsProcessing(true);
      try {
        const res = await api.ocr.extractText(sessionId.trim(), language, targetImage);
        setOcrResult(res);
      } catch (err: any) {
        const msg = err.message || '';
        if (msg.includes('404') || msg.toLowerCase().includes('not found')) {
          setError(
            `Session '${sessionId.trim()}' was not found on the server. Please verify your session ID from History, or switch to the 'Direct Image Upload' tab.`
          );
        } else {
          setError(msg || 'Failed to execute OCR text extraction.');
        }
      } finally {
        setIsProcessing(false);
      }
    } else {
      // Direct file mode
      if (!selectedFile) {
        setError('Please select an image file to perform direct OCR text extraction.');
        return;
      }

      setIsProcessing(true);
      try {
        const res = await api.ocr.extractDirect(selectedFile, language);
        setOcrResult(res);
      } catch (err: any) {
        setError(err.message || 'Direct OCR text extraction failed.');
      } finally {
        setIsProcessing(false);
      }
    }
  };

  const handleCopyText = () => {
    if (!ocrResult?.extracted_text) return;
    navigator.clipboard.writeText(ocrResult.extracted_text);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const handlePredictMissingText = async () => {
    if (!leftContext.trim() && !rightContext.trim()) {
      return;
    }
    setIsPredicting(true);
    try {
      const res = await api.prediction.predictText({
        left_context: leftContext,
        right_context: rightContext,
        max_candidates: 4,
      });
      setPredictionCandidates(res.candidates || []);
      setPredictionDisclaimer(res.disclaimer);
    } catch (err: any) {
      setError(err.message || 'Linguistic completion prediction failed.');
    } finally {
      setIsPredicting(false);
    }
  };

  return (
    <div className="space-y-8 max-w-5xl mx-auto">
      {/* Title */}
      <div className="pb-4 border-b border-stone-200">
        <h1 className="font-serif-scholarly text-3xl font-bold tracking-tight text-stone-900">
          Optical Character Recognition & Text Preservation
        </h1>
        <p className="text-xs text-stone-500 mt-1">
          Authentic Tesseract OCR for Vedic Sanskrit & Hindi manuscripts with speculative linguistic completion
        </p>
      </div>

      {/* Engine Status Diagnostic Card */}
      <div className="p-4 rounded-xl border border-stone-200 bg-white shadow-xs">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
          <div className="flex items-center gap-3">
            <div className="p-2 rounded-lg bg-stone-100 text-stone-700">
              <Cpu className="w-5 h-5 text-emerald-800" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <span className="text-xs font-bold text-stone-900 uppercase tracking-wider">
                  Tesseract OCR Engine
                </span>
                {isLoadingStatus ? (
                  <span className="text-[11px] text-stone-400 font-mono">auditing...</span>
                ) : (
                  <StatusBadge status={engineStatus?.status || 'UNAVAILABLE'} size="sm" />
                )}
              </div>
              <p className="text-xs text-stone-600 mt-0.5">
                {engineStatus?.is_installed ? (
                  <>
                    <span className="font-semibold text-emerald-800">
                      Tesseract {engineStatus.version || ''}
                    </span>{' '}
                    detected • Installed packs:{' '}
                    <span className="font-mono text-[11px] text-stone-700 font-semibold">
                      {engineStatus.available_languages?.join(', ') || 'none'}
                    </span>
                  </>
                ) : (
                  'External engine not detected on system PATH or default Windows directories.'
                )}
              </p>
            </div>
          </div>

          {engineStatus?.is_installed && engineStatus.devanagari_support && (
            <div className="flex items-center gap-1.5 text-xs font-semibold text-emerald-800 bg-emerald-50 px-3 py-1.5 rounded-lg border border-emerald-200 shrink-0">
              <CheckCircle2 className="w-4 h-4 text-emerald-700" />
              <span>Devanagari / Sanskrit Ready</span>
            </div>
          )}
        </div>
      </div>

      {error && (
        <div className="p-4 rounded-lg bg-rose-50 border border-rose-200 text-rose-900 text-xs flex items-start gap-3">
          <AlertTriangle className="w-5 h-5 text-rose-600 shrink-0 mt-0.5" />
          <div>
            <div className="font-semibold">OCR Pipeline Notice</div>
            <div className="mt-0.5">{error}</div>
          </div>
        </div>
      )}

      {/* Control & Ingestion Card */}
      <div className="bg-white p-6 rounded-xl border border-stone-200 shadow-xs space-y-5">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between border-b border-stone-100 pb-3 gap-2">
          <h2 className="font-serif-scholarly font-bold text-base text-stone-900">
            Folio Ingestion & Script Parameters
          </h2>

          {/* Mode Switcher Tabs */}
          <div className="flex rounded-lg bg-stone-100 p-0.5 text-xs font-medium self-start sm:self-auto">
            <button
              type="button"
              onClick={() => setInputMode('session')}
              className={`px-3 py-1.5 rounded-md transition cursor-pointer ${
                inputMode === 'session'
                  ? 'bg-white text-stone-900 shadow-xs font-semibold'
                  : 'text-stone-600 hover:text-stone-900'
              }`}
            >
              Session Folio
            </button>
            <button
              type="button"
              onClick={() => setInputMode('direct')}
              className={`px-3 py-1.5 rounded-md transition cursor-pointer ${
                inputMode === 'direct'
                  ? 'bg-white text-stone-900 shadow-xs font-semibold'
                  : 'text-stone-600 hover:text-stone-900'
              }`}
            >
              Direct Image Upload
            </button>
          </div>
        </div>

        {/* Ingestion Mode 1: Restoration Session ID */}
        {inputMode === 'session' ? (
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="md:col-span-2 space-y-2">
              <div className="flex items-center justify-between">
                <label className="block text-xs font-medium text-stone-700">
                  Restoration Session ID
                </label>
                <Link
                  to="/history"
                  className="text-[11px] text-emerald-800 hover:underline flex items-center gap-1 font-medium"
                >
                  <History className="w-3 h-3" />
                  <span>Browse Recent Sessions</span>
                </Link>
              </div>
              <input
                type="text"
                value={sessionId}
                onChange={(e) => setSessionId(e.target.value)}
                placeholder="e.g. p2p_645fd155_1729000000"
                className="w-full text-xs font-mono border border-stone-300 rounded-lg p-2.5 bg-stone-50 focus:ring-2 focus:ring-emerald-700"
              />
            </div>

            <div>
              <label className="block text-xs font-medium text-stone-700 mb-1">
                Target Pipeline Artifact
              </label>
              <select
                value={targetImage}
                onChange={(e) => setTargetImage(e.target.value as 'restored' | 'original')}
                className="w-full text-xs border border-stone-300 rounded-lg p-2.5 bg-stone-50 focus:ring-2 focus:ring-emerald-700"
              >
                <option value="restored">Restored Folio (U-Net Cleaned)</option>
                <option value="original">Original Degraded Folio</option>
              </select>
            </div>
          </div>
        ) : (
          /* Ingestion Mode 2: Direct File Upload */
          <div className="space-y-3">
            <label className="block text-xs font-medium text-stone-700">
              Upload Manuscript Folio Image
            </label>
            <div
              onClick={() => fileInputRef.current?.click()}
              className="border-2 border-dashed border-stone-300 hover:border-emerald-700 rounded-xl p-5 text-center cursor-pointer transition bg-stone-50/50 hover:bg-emerald-50/20"
            >
              <input
                ref={fileInputRef}
                type="file"
                accept="image/*"
                onChange={handleFileChange}
                className="hidden"
              />
              <div className="flex flex-col items-center justify-center gap-2">
                {filePreview ? (
                  <div className="flex items-center gap-3">
                    <img
                      src={filePreview}
                      alt="Uploaded preview"
                      className="w-16 h-16 object-cover rounded-lg border border-stone-300"
                    />
                    <div className="text-left">
                      <div className="text-xs font-semibold text-stone-800">
                        {selectedFile?.name}
                      </div>
                      <div className="text-[11px] text-stone-500">
                        {selectedFile ? (selectedFile.size / 1024).toFixed(1) : 0} KB • Click to
                        change
                      </div>
                    </div>
                  </div>
                ) : (
                  <>
                    <Upload className="w-8 h-8 text-stone-400" />
                    <div className="text-xs font-semibold text-stone-700">
                      Click to choose manuscript image or drag & drop here
                    </div>
                    <div className="text-[11px] text-stone-400">
                      Supports PNG, JPG, TIFF, BMP (up to 15 MB)
                    </div>
                  </>
                )}
              </div>
            </div>
          </div>
        )}

        {/* Script Selection & Execution Row */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 pt-2">
          <div>
            <label className="block text-xs font-medium text-stone-700 mb-1">
              Target Script / Language Pack
            </label>
            <select
              value={language}
              onChange={(e) => setLanguage(e.target.value)}
              className="w-full text-xs border border-stone-300 rounded-lg p-2.5 bg-stone-50 focus:ring-2 focus:ring-emerald-700"
            >
              <option value="san+hin">Sanskrit + Hindi (Devanagari)</option>
              <option value="san">Vedic Sanskrit (san)</option>
              <option value="hin">Hindi (hin)</option>
              <option value="eng">English Transliteration (eng)</option>
            </select>
          </div>

          <div className="md:col-span-2 flex items-end">
            <button
              type="button"
              onClick={handleExtractOcr}
              disabled={
                isProcessing ||
                (inputMode === 'session' && !sessionId.trim()) ||
                (inputMode === 'direct' && !selectedFile)
              }
              className="inline-flex items-center gap-2 px-6 py-2.5 bg-emerald-800 hover:bg-emerald-700 text-white rounded-lg shadow-sm text-xs font-semibold transition disabled:opacity-50 cursor-pointer w-full sm:w-auto justify-center"
            >
              {isProcessing ? (
                <>
                  <RefreshCw className="w-4 h-4 animate-spin" />
                  <span>Scanning Folio Glyphs...</span>
                </>
              ) : (
                <>
                  <FileText className="w-4 h-4" />
                  <span>
                    {inputMode === 'session'
                      ? 'Execute OCR on Session Folio'
                      : 'Execute Direct OCR on Upload'}
                  </span>
                </>
              )}
            </button>
          </div>
        </div>
      </div>

      {/* OCR Results Display */}
      {ocrResult && (
        <div className="space-y-6">
          {/* Tesseract Unavailable Diagnostic Warning */}
          {ocrResult.status === 'UNAVAILABLE' ? (
            <div className="p-6 rounded-xl bg-amber-50 border border-amber-300 text-amber-950 space-y-3">
              <div className="flex items-center gap-2 font-bold text-sm text-amber-900">
                <AlertTriangle className="w-5 h-5 text-amber-700" />
                <span>Tesseract OCR Engine Not Configured on Server</span>
              </div>
              <p className="text-xs text-amber-900/90 leading-relaxed">
                {ocrResult.error ||
                  'The server environment does not have Tesseract installed or configured in the system PATH. Per academic honesty rules, Past2Pixel does not fabricate OCR text.'}
              </p>
              <div className="bg-amber-100/70 p-4 rounded-lg border border-amber-200 text-xs space-y-2">
                <div className="font-semibold text-amber-900 flex items-center gap-1.5">
                  <Terminal className="w-4 h-4 text-amber-800" />
                  <span>Actionable Setup Steps:</span>
                </div>
                <ol className="list-decimal list-inside space-y-1 text-stone-700 font-mono text-[11px]">
                  <li>Download Tesseract OCR for Windows from UB-Mannheim</li>
                  <li>Include "Additional script data: Devanagari" during installation</li>
                  <li>Add path to your .env: TESSERACT_CMD=C:\Program Files\Tesseract-OCR\tesseract.exe</li>
                </ol>
              </div>
            </div>
          ) : (
            /* Authentic OCR Text Display */
            <div className="bg-white p-6 rounded-xl border border-stone-200 shadow-sm space-y-4">
              <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 border-b border-stone-100 pb-3">
                <div className="flex items-center gap-3">
                  <h2 className="font-serif-scholarly font-bold text-lg text-stone-900">
                    Extracted Manuscript Text
                  </h2>
                  <StatusBadge status={ocrResult.status} size="sm" />
                  {ocrResult.image_source && (
                    <span className="text-[11px] font-mono px-2 py-0.5 rounded bg-stone-100 text-stone-600 border border-stone-200">
                      source: {ocrResult.image_source}
                    </span>
                  )}
                </div>
                <div className="flex items-center gap-4 text-xs text-stone-500">
                  <span>Characters: {ocrResult.character_count}</span>
                  <span>Words: {ocrResult.word_count}</span>
                  {ocrResult.mean_confidence != null && (
                    <span className="font-semibold text-emerald-800">
                      Confidence: {ocrResult.mean_confidence.toFixed(1)}%
                    </span>
                  )}
                  <button
                    onClick={handleCopyText}
                    className="inline-flex items-center gap-1 px-2.5 py-1 rounded bg-stone-100 hover:bg-stone-200 text-stone-700 font-medium transition cursor-pointer"
                  >
                    {copied ? (
                      <Check className="w-3.5 h-3.5 text-emerald-600" />
                    ) : (
                      <Copy className="w-3.5 h-3.5" />
                    )}
                    <span>{copied ? 'Copied' : 'Copy'}</span>
                  </button>
                </div>
              </div>

              {/* Devanagari Typography Paper Container */}
              <div className="p-6 bg-stone-50 rounded-lg border border-stone-200 font-serif-scholarly text-stone-900 text-lg leading-loose tracking-wide whitespace-pre-wrap selection:bg-amber-200">
                {ocrResult.extracted_text || (
                  <span className="text-stone-400 italic text-sm">
                    No characters recognized in this folio region.
                  </span>
                )}
              </div>
            </div>
          )}
        </div>
      )}

      {/* Context-Aware Linguistic Completion (XAI / Preservation) */}
      <div className="bg-white p-6 rounded-xl border border-stone-200 shadow-xs space-y-4">
        <div className="border-b border-stone-100 pb-2">
          <div className="flex items-center gap-2">
            <Sparkles className="w-5 h-5 text-amber-600" />
            <h2 className="font-serif-scholarly font-bold text-base text-stone-900">
              Context-Aware Damaged Text Completion (Linguistic N-Gram)
            </h2>
          </div>
          <p className="text-xs text-stone-500 mt-1">
            Statistically predicts obscured characters using authentic Vedic corpora. Strictly marked as speculative.
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div>
            <label className="block text-xs font-medium text-stone-700 mb-1">
              Left Readable Context (Devanagari)
            </label>
            <input
              type="text"
              value={leftContext}
              onChange={(e) => setLeftContext(e.target.value)}
              className="w-full text-sm font-serif-scholarly border border-stone-300 rounded-lg p-2.5 bg-stone-50"
            />
          </div>

          <div>
            <label className="block text-xs font-medium text-stone-700 mb-1">
              Right Readable Context (Devanagari)
            </label>
            <input
              type="text"
              value={rightContext}
              onChange={(e) => setRightContext(e.target.value)}
              className="w-full text-sm font-serif-scholarly border border-stone-300 rounded-lg p-2.5 bg-stone-50"
            />
          </div>
        </div>

        <button
          type="button"
          onClick={handlePredictMissingText}
          disabled={isPredicting}
          className="inline-flex items-center gap-2 px-4 py-2 bg-stone-800 hover:bg-stone-700 text-stone-100 rounded-lg text-xs font-semibold shadow-xs transition cursor-pointer"
        >
          {isPredicting ? (
            <>
              <RefreshCw className="w-3.5 h-3.5 animate-spin" />
              <span>Querying Linguistic Corpora...</span>
            </>
          ) : (
            <>
              <Sparkles className="w-3.5 h-3.5 text-amber-400" />
              <span>Predict Missing Span</span>
            </>
          )}
        </button>

        {predictionCandidates.length > 0 && (
          <div className="mt-4 p-4 rounded-lg bg-amber-50/60 border border-amber-200 space-y-3">
            <div className="text-xs font-semibold text-stone-800">
              Corpus Completion Candidates (Ranked by Transition Probability):
            </div>
            <div className="flex flex-wrap gap-2">
              {predictionCandidates.map((c, i) => (
                <div
                  key={i}
                  className="px-3 py-1.5 rounded bg-white border border-amber-300 text-xs font-serif-scholarly flex items-center gap-2 shadow-xs"
                >
                  <span className="font-bold text-stone-900">{c.word}</span>
                  <span className="text-[10px] text-stone-400 font-mono">
                    ({(c.score * 100).toFixed(1)}%)
                  </span>
                </div>
              ))}
            </div>

            {predictionDisclaimer && (
              <p className="text-[11px] text-amber-900/80 italic pt-1 border-t border-amber-200">
                {predictionDisclaimer}
              </p>
            )}
          </div>
        )}
      </div>
    </div>
  );
};
