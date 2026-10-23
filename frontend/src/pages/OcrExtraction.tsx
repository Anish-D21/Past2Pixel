import React, { useState, useEffect, useRef } from 'react';
import { useSearchParams, Link } from 'react-router-dom';
import { api } from '../services/api';
import {
  OCRResult,
  OCRStatusInfo,
  TextPredictionCandidate,
  RestorationSession,
  SessionDetailResponse,
} from '../types/api';
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
  Cpu,
  History,
  Download,
  RotateCcw,
  Sliders,
  ChevronDown,
  ChevronUp,
  Layers,
  HelpCircle,
  ArrowRight,
  ExternalLink,
} from 'lucide-react';

export const OcrExtraction: React.FC = () => {
  const [searchParams] = useSearchParams();
  const sessionFromQuery = searchParams.get('session') || '';

  // Mode & Ingestion State
  const [inputMode, setInputMode] = useState<'session' | 'direct'>('session');
  const [sessionId, setSessionId] = useState<string>(sessionFromQuery);
  const [availableSessions, setAvailableSessions] = useState<RestorationSession[]>([]);
  const [sessionDetail, setSessionDetail] = useState<SessionDetailResponse | null>(null);
  const [isLoadingSession, setIsLoadingSession] = useState<boolean>(false);

  // Direct File State
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [filePreview, setFilePreview] = useState<string | null>(null);

  // OCR Configuration State
  const [language, setLanguage] = useState<string>('san+hin');
  const [targetImage, setTargetImage] = useState<
    'restored' | 'original' | 'baseline' | 'model' | 'compare_all'
  >('restored');
  const [preprocessing, setPreprocessing] = useState<
    'none' | 'auto' | 'contrast' | 'illumination' | 'binarize' | 'upscale'
  >('auto');

  // Engine & Processing State
  const [engineStatus, setEngineStatus] = useState<OCRStatusInfo | null>(null);
  const [isLoadingStatus, setIsLoadingStatus] = useState<boolean>(true);
  const [isProcessing, setIsProcessing] = useState<boolean>(false);
  const [ocrResult, setOcrResult] = useState<OCRResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  // Dual-Workspace Text State (Editable vs Immutable Raw)
  const [rawOcrText, setRawOcrText] = useState<string>('');
  const [workingText, setWorkingText] = useState<string>('');
  const [showRawComparison, setShowRawComparison] = useState<boolean>(false);
  const [copiedWorking, setCopiedWorking] = useState<boolean>(false);
  const [copiedRaw, setCopiedRaw] = useState<boolean>(false);
  const [copiedCandidate, setCopiedCandidate] = useState<string | null>(null);

  // Contextual Prediction State (Operates directly on workingText)
  const [selectedWordIndex, setSelectedWordIndex] = useState<number | null>(null);
  const [uncertainSpan, setUncertainSpan] = useState<string>('');
  const [leftContext, setLeftContext] = useState<string>('');
  const [rightContext, setRightContext] = useState<string>('');
  const [isPredicting, setIsPredicting] = useState<boolean>(false);
  const [predictionCandidates, setPredictionCandidates] = useState<TextPredictionCandidate[]>([]);
  const [predictionDisclaimer, setPredictionDisclaimer] = useState<string | null>(null);
  const [acceptedNotice, setAcceptedNotice] = useState<string | null>(null);

  const fileInputRef = useRef<HTMLInputElement>(null);

  // Initialize query param
  useEffect(() => {
    if (sessionFromQuery) {
      setSessionId(sessionFromQuery);
      setInputMode('session');
    }
  }, [sessionFromQuery]);

  // Fetch engine readiness and recent user sessions on mount
  useEffect(() => {
    let isMounted = true;
    const fetchStatusAndSessions = async () => {
      setIsLoadingStatus(true);
      try {
        const [statusRes, sessionsRes] = await Promise.allSettled([
          api.ocr.getStatus(),
          api.sessions.listMySessions(),
        ]);

        if (!isMounted) return;

        if (statusRes.status === 'fulfilled') {
          setEngineStatus(statusRes.value);
        }
        if (sessionsRes.status === 'fulfilled') {
          setAvailableSessions(sessionsRes.value || []);
        }
      } catch (err: any) {
        console.warn('Failed to load initial OCR prerequisites:', err);
      } finally {
        if (isMounted) setIsLoadingStatus(false);
      }
    };

    fetchStatusAndSessions();
    return () => {
      isMounted = false;
    };
  }, []);

  // Fetch session details whenever sessionId changes in session mode
  useEffect(() => {
    if (inputMode !== 'session' || !sessionId.trim()) {
      setSessionDetail(null);
      return;
    }

    let isMounted = true;
    setIsLoadingSession(true);

    api.sessions
      .getSession(sessionId.trim())
      .then((detail) => {
        if (isMounted) {
          setSessionDetail(detail);
        }
      })
      .catch((err) => {
        if (isMounted) {
          setSessionDetail(null);
          console.warn(`Could not load details for session ${sessionId}:`, err);
        }
      })
      .finally(() => {
        if (isMounted) setIsLoadingSession(false);
      });

    return () => {
      isMounted = false;
    };
  }, [sessionId, inputMode]);

  // When OCR completes, initialize both raw and working text
  useEffect(() => {
    if (ocrResult?.extracted_text) {
      setRawOcrText(ocrResult.extracted_text);
      setWorkingText(ocrResult.extracted_text);
      setSelectedWordIndex(null);
      setUncertainSpan('');
      setLeftContext('');
      setRightContext('');
      setPredictionCandidates([]);
      setPredictionDisclaimer(null);
      setAcceptedNotice(null);
    }
  }, [ocrResult]);

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
        setError('Please provide or select a valid restoration session ID.');
        return;
      }

      setIsProcessing(true);
      try {
        const res = await api.ocr.extractText(
          sessionId.trim(),
          language,
          targetImage,
          preprocessing
        );
        setOcrResult(res);
      } catch (err: any) {
        const msg = err.message || '';
        if (msg.includes('404') || msg.toLowerCase().includes('not found')) {
          setError(
            `Session '${sessionId.trim()}' was not found on the server. Please verify the session ID from History, or switch to 'Direct Image Upload'.`
          );
        } else {
          setError(msg || 'Failed to execute OCR text extraction.');
        }
      } finally {
        setIsProcessing(false);
      }
    } else {
      if (!selectedFile) {
        setError('Please select an image file to perform direct OCR text extraction.');
        return;
      }

      setIsProcessing(true);
      try {
        const res = await api.ocr.extractDirect(selectedFile, language, preprocessing);
        setOcrResult(res);
      } catch (err: any) {
        setError(err.message || 'Direct OCR text extraction failed.');
      } finally {
        setIsProcessing(false);
      }
    }
  };

  // Copy helpers
  const handleCopyWorkingText = () => {
    if (!workingText) return;
    navigator.clipboard.writeText(workingText);
    setCopiedWorking(true);
    setTimeout(() => setCopiedWorking(false), 2000);
  };

  const handleCopyRawText = () => {
    if (!rawOcrText) return;
    navigator.clipboard.writeText(rawOcrText);
    setCopiedRaw(true);
    setTimeout(() => setCopiedRaw(false), 2000);
  };

  const handleCopyCandidate = (text: string, key: string) => {
    navigator.clipboard.writeText(text);
    setCopiedCandidate(key);
    setTimeout(() => setCopiedCandidate(null), 2000);
  };

  // UTF-8 Plain Text Download
  const handleDownloadTxt = () => {
    if (!workingText) return;
    const header = [
      '================================================================',
      'PAST2PIXEL MANUSCRIPT OCR & TEXT PRESERVATION REPORT',
      `Timestamp: ${new Date().toISOString()}`,
      `Source Mode: ${inputMode === 'session' ? `Session (${sessionId})` : 'Direct Image Upload'}`,
      `Target Image: ${inputMode === 'session' ? targetImage : (selectedFile?.name || 'Uploaded File')}`,
      `OCR Language: ${language}`,
      `Preprocessing Applied: ${ocrResult?.preprocessing_applied || preprocessing}`,
      `Mean Engine Confidence: ${ocrResult?.mean_confidence != null ? `${ocrResult.mean_confidence.toFixed(1)}%` : 'N/A'}`,
      `Character Count: ${workingText.length}`,
      '================================================================',
      '',
      workingText,
    ].join('\n');

    const blob = new Blob([header], { type: 'text/plain;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `past2pixel_ocr_${sessionId || 'manuscript'}_${Date.now()}.txt`;
    a.click();
    URL.revokeObjectURL(url);
  };

  // Reset working text to immutable raw OCR output
  const handleResetToRaw = () => {
    setWorkingText(rawOcrText);
    setAcceptedNotice('Working text reset to original raw OCR output.');
    setTimeout(() => setAcceptedNotice(null), 3000);
  };

  // Tokenization of workingText for interactive token selector
  const tokens = workingText
    .split(/(\s+|[।॥,;:.!?]+)/)
    .filter((t) => t.length > 0);

  // Extract context around selected token in workingText
  const handleSelectToken = (index: number, word: string) => {
    setSelectedWordIndex(index);
    setUncertainSpan(word);

    // Compute left words and right words
    const beforeTokens = tokens.slice(0, index).filter((t) => /\S/.test(t) && !/^[।॥,;:.!?]+$/.test(t));
    const afterTokens = tokens.slice(index + 1).filter((t) => /\S/.test(t) && !/^[।॥,;:.!?]+$/.test(t));

    const left = beforeTokens.slice(-3).join(' ');
    const right = afterTokens.slice(0, 3).join(' ');

    setLeftContext(left);
    setRightContext(right);
    setPredictionCandidates([]);
    setPredictionDisclaimer(null);
  };

  // Trigger Context-Aware Damaged Text Completion
  const handlePredictMissingText = async () => {
    if (!uncertainSpan.trim() && !leftContext.trim() && !rightContext.trim()) {
      setError('Please select or specify an uncertain span or surrounding context.');
      return;
    }

    setIsPredicting(true);
    setAcceptedNotice(null);
    try {
      const res = await api.prediction.predictText({
        full_text: workingText,
        uncertain_span: uncertainSpan.trim() || undefined,
        span_index: selectedWordIndex != null ? selectedWordIndex : undefined,
        left_context: leftContext.trim() || undefined,
        right_context: rightContext.trim() || undefined,
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

  // Accept a predicted candidate: updates workingText without touching rawOcrText!
  const handleAcceptCandidate = (candidateWord: string) => {
    if (selectedWordIndex != null && selectedWordIndex >= 0 && selectedWordIndex < tokens.length) {
      const originalWord = tokens[selectedWordIndex];
      const newTokens = [...tokens];
      newTokens[selectedWordIndex] = candidateWord;
      const updatedText = newTokens.join('');
      setWorkingText(updatedText);
      setAcceptedNotice(
        `Applied correction: replaced "${originalWord}" with linguistic candidate "${candidateWord}".`
      );
    } else if (uncertainSpan && workingText.includes(uncertainSpan)) {
      // Fallback: replace first occurrence of uncertainSpan
      const updatedText = workingText.replace(uncertainSpan, candidateWord);
      setWorkingText(updatedText);
      setAcceptedNotice(
        `Applied correction: replaced "${uncertainSpan}" with linguistic candidate "${candidateWord}".`
      );
    } else {
      // Append if no span selected
      setWorkingText((prev) => `${prev} ${candidateWord}`.trim());
      setAcceptedNotice(`Inserted candidate "${candidateWord}" into working text.`);
    }

    setUncertainSpan(candidateWord);
    setPredictionCandidates([]);
    setTimeout(() => setAcceptedNotice(null), 4000);
  };

  // Compute text differences indicator
  const isTextModified = rawOcrText !== '' && workingText !== rawOcrText;

  // Artifact check for current session
  const originalImageUrl = sessionDetail?.artifacts?.original;
  const restoredImageUrl = sessionDetail?.artifacts?.restored;
  const isRestoredAvailable = !!restoredImageUrl;

  return (
    <div className="space-y-8 max-w-5xl mx-auto">
      {/* Title */}
      <div className="pb-4 border-b border-stone-200">
        <h1 className="font-serif-scholarly text-3xl font-bold tracking-tight text-stone-900">
          Optical Character Recognition & Text Preservation
        </h1>
        <p className="text-xs text-stone-500 mt-1">
          Evidence-based OCR for Vedic Sanskrit, Hindi, and English manuscripts with context-aware linguistic completion
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
                    {engineStatus.tessdata_dir && (
                      <span className="text-stone-400 text-[10px] block truncate font-mono">
                        tessdata: {engineStatus.tessdata_dir}
                      </span>
                    )}
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
      <div className="bg-white p-6 rounded-xl border border-stone-200 shadow-xs space-y-6">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between border-b border-stone-100 pb-3 gap-2">
          <h2 className="font-serif-scholarly font-bold text-base text-stone-900 flex items-center gap-2">
            <span>Folio Ingestion & Script Parameters</span>
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

        {/* Ingestion Mode 1: Restoration Session ID & Visual Previews */}
        {inputMode === 'session' ? (
          <div className="space-y-4">
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div className="md:col-span-2 space-y-2">
                <div className="flex items-center justify-between">
                  <label className="block text-xs font-medium text-stone-700">
                    Restoration Session Selection
                  </label>
                  <Link
                    to="/history"
                    className="text-[11px] text-emerald-800 hover:underline flex items-center gap-1 font-medium"
                  >
                    <History className="w-3 h-3" />
                    <span>Browse History</span>
                  </Link>
                </div>

                {availableSessions.length > 0 ? (
                  <div className="flex gap-2">
                    <select
                      value={sessionId}
                      onChange={(e) => setSessionId(e.target.value)}
                      className="w-full text-xs font-mono border border-stone-300 rounded-lg p-2.5 bg-stone-50 focus:ring-2 focus:ring-emerald-700"
                    >
                      <option value="">-- Select a Recent Session --</option>
                      {availableSessions.map((s) => (
                        <option key={s.session_id} value={s.session_id}>
                          {s.session_id} ({new Date(s.created_at).toLocaleDateString()}) - {s.status}
                        </option>
                      ))}
                    </select>
                  </div>
                ) : (
                  <input
                    type="text"
                    value={sessionId}
                    onChange={(e) => setSessionId(e.target.value)}
                    placeholder="e.g. p2p_645fd155_1729000000"
                    className="w-full text-xs font-mono border border-stone-300 rounded-lg p-2.5 bg-stone-50 focus:ring-2 focus:ring-emerald-700"
                  />
                )}
              </div>

              <div>
                <label className="block text-xs font-medium text-stone-700 mb-1">
                  Target Pipeline Artifact
                </label>
                <select
                  value={targetImage}
                  onChange={(e) =>
                    setTargetImage(
                      e.target.value as 'restored' | 'original' | 'baseline' | 'model' | 'compare_all'
                    )
                  }
                  className="w-full text-xs border border-stone-300 rounded-lg p-2.5 bg-stone-50 focus:ring-2 focus:ring-emerald-700"
                >
                  <option value="restored">Restored Folio (Text-Aware Selected)</option>
                  <option value="original">Original Degraded Folio</option>
                  <option value="baseline">Traditional Baseline (Illumination Normalized)</option>
                  <option value="model">Neural Model (Raw U-Net Output)</option>
                  <option value="compare_all">⚖️ Compare All Candidates</option>
                </select>
              </div>
            </div>

            {/* Session Visual Folio Previews */}
            {sessionId.trim() && (
              <div className="p-3.5 bg-stone-50 rounded-xl border border-stone-200">
                <div className="flex items-center justify-between mb-2">
                  <span className="text-xs font-semibold text-stone-700 flex items-center gap-1.5">
                    <ImageIcon className="w-4 h-4 text-emerald-800" />
                    <span>Session Folio Assets</span>
                  </span>
                  {isLoadingSession && (
                    <span className="text-[11px] text-stone-400 font-mono animate-pulse">
                      loading session images...
                    </span>
                  )}
                </div>

                {sessionDetail ? (
                  <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                    <div className="p-2.5 bg-white rounded-lg border border-stone-200 space-y-1.5">
                      <div className="flex items-center justify-between text-[11px]">
                        <span className="font-semibold text-stone-700">Original Manuscript</span>
                        <span className="text-[10px] text-stone-400 font-mono">Input</span>
                      </div>
                      {originalImageUrl ? (
                        <div className="h-28 w-full bg-stone-100 rounded overflow-hidden flex items-center justify-center border border-stone-200">
                          <img
                            src={originalImageUrl}
                            alt="Original Folio"
                            className="max-h-full max-w-full object-contain"
                          />
                        </div>
                      ) : (
                        <div className="h-28 w-full bg-stone-100 rounded flex items-center justify-center text-stone-400 text-xs italic">
                          Original image unavailable
                        </div>
                      )}
                    </div>

                    <div className="p-2.5 bg-white rounded-lg border border-stone-200 space-y-1.5">
                      <div className="flex items-center justify-between text-[11px]">
                        <span className="font-semibold text-stone-700">Restored Folio</span>
                        {isRestoredAvailable ? (
                          <span className="text-[10px] bg-emerald-100 text-emerald-800 px-1.5 py-0.2 rounded font-semibold">
                            Restored
                          </span>
                        ) : (
                          <span className="text-[10px] bg-amber-100 text-amber-800 px-1.5 py-0.2 rounded">
                            Pending
                          </span>
                        )}
                      </div>
                      {restoredImageUrl ? (
                        <div className="h-28 w-full bg-stone-100 rounded overflow-hidden flex items-center justify-center border border-stone-200">
                          <img
                            src={restoredImageUrl}
                            alt="Restored Folio"
                            className="max-h-full max-w-full object-contain"
                          />
                        </div>
                      ) : (
                        <div className="h-28 w-full bg-amber-50 rounded flex flex-col items-center justify-center text-amber-800 text-[11px] p-2 text-center border border-amber-200">
                          <AlertTriangle className="w-4 h-4 mb-1 text-amber-600" />
                          <span>Restored image not yet generated for this session.</span>
                          <span className="text-[10px] text-amber-700 mt-0.5">
                            OCR will run on the original image if 'restored' is selected.
                          </span>
                        </div>
                      )}
                    </div>
                  </div>
                ) : (
                  !isLoadingSession && (
                    <div className="text-xs text-stone-500 italic p-2">
                      Enter or select a session above to preview input and restored folios.
                    </div>
                  )
                )}
              </div>
            )}
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
                      className="w-20 h-20 object-contain rounded-lg border border-stone-300 bg-white"
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
                      Supports PNG, JPG, JPEG, TIFF, BMP (up to 15 MB)
                    </div>
                  </>
                )}
              </div>
            </div>
          </div>
        )}

        {/* Script Selection & Preprocessing Settings Row */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 pt-2 border-t border-stone-100">
          <div>
            <label className="block text-xs font-medium text-stone-700 mb-1">
              Target Script / Language Pack
            </label>
            <select
              value={language}
              onChange={(e) => setLanguage(e.target.value)}
              className="w-full text-xs border border-stone-300 rounded-lg p-2.5 bg-stone-50 focus:ring-2 focus:ring-emerald-700 font-medium"
            >
              <option value="san+hin">Sanskrit + Hindi (Devanagari Dual-Pack)</option>
              <option value="san">Vedic Sanskrit (san)</option>
              <option value="hin">Hindi (hin)</option>
              <option value="eng">English Transliteration (eng)</option>
              <option value="hin+eng">Hindi + English (hin+eng)</option>
              <option value="san+hin+eng">Multilingual (san+hin+eng)</option>
            </select>
            <p className="text-[10px] text-stone-400 mt-1">
              Verified against installed traineddata packs.
            </p>
          </div>

          <div>
            <label className="block text-xs font-medium text-stone-700 mb-1 flex items-center gap-1.5">
              <Sliders className="w-3.5 h-3.5 text-stone-500" />
              <span>OCR Preprocessing</span>
            </label>
            <select
              value={preprocessing}
              onChange={(e) =>
                setPreprocessing(
                  e.target.value as 'none' | 'auto' | 'contrast' | 'illumination' | 'binarize' | 'upscale'
                )
              }
              className="w-full text-xs border border-stone-300 rounded-lg p-2.5 bg-stone-50 focus:ring-2 focus:ring-emerald-700"
            >
              <option value="auto">Auto (Evidence-Based Selection)</option>
              <option value="none">None (Raw Input Image)</option>
              <option value="contrast">Conservative Contrast Optimization</option>
              <option value="illumination">Illumination Normalization</option>
              <option value="binarize">Adaptive Binarization (Sauvola)</option>
              <option value="upscale">2× Super-Resolution Upscaling</option>
            </select>
            <p className="text-[10px] text-stone-400 mt-1">
              Isolated OCR enhancement; does not overwrite restoration.
            </p>
          </div>

          <div className="flex items-end">
            <button
              type="button"
              onClick={handleExtractOcr}
              disabled={
                isProcessing ||
                (inputMode === 'session' && !sessionId.trim()) ||
                (inputMode === 'direct' && !selectedFile)
              }
              className="inline-flex items-center gap-2 px-6 py-2.5 bg-emerald-800 hover:bg-emerald-700 text-white rounded-lg shadow-sm text-xs font-semibold transition disabled:opacity-50 cursor-pointer w-full justify-center"
            >
              {isProcessing ? (
                <>
                  <RefreshCw className="w-4 h-4 animate-spin" />
                  <span>Extracting Folio Glyphs...</span>
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
          {ocrResult.status === 'UNAVAILABLE' || ocrResult.status === 'MISSING_LANGUAGE_PACK' ? (
            <div className="p-6 rounded-xl bg-amber-50 border border-amber-300 text-amber-950 space-y-3">
              <div className="flex items-center gap-2 font-bold text-sm text-amber-900">
                <AlertTriangle className="w-5 h-5 text-amber-700" />
                <span>
                  {ocrResult.status === 'MISSING_LANGUAGE_PACK'
                    ? 'Required Tesseract Language Pack Missing'
                    : 'Tesseract OCR Engine Not Configured on Server'}
                </span>
              </div>
              <p className="text-xs text-amber-900/90 leading-relaxed">
                {ocrResult.error ||
                  'The requested OCR configuration could not be executed. Per academic honesty rules, Past2Pixel never substitutes random language packs or fabricates OCR output.'}
              </p>
              <div className="bg-amber-100/70 p-4 rounded-lg border border-amber-200 text-xs space-y-2">
                <div className="font-semibold text-amber-900 flex items-center gap-1.5">
                  <Terminal className="w-4 h-4 text-amber-800" />
                  <span>Actionable Setup Steps:</span>
                </div>
                <ol className="list-decimal list-inside space-y-1 text-stone-700 font-mono text-[11px]">
                  <li>Ensure Tesseract OCR is installed on the host machine.</li>
                  <li>
                    Place missing <code className="bg-amber-200/60 px-1 rounded">.traineddata</code> files (e.g. <code className="bg-amber-200/60 px-1 rounded">san.traineddata</code>, <code className="bg-amber-200/60 px-1 rounded">hin.traineddata</code>) in your tessdata directory.
                  </li>
                  <li>Verify engine status using the diagnostic panel above.</li>
                </ol>
              </div>
            </div>
          ) : (
            <div className="space-y-6">
              {/* Multi-Candidate Comparative OCR Grid if comparison_results present */}
              {ocrResult.comparison_results ? (
                <div className="bg-white p-6 rounded-xl border border-stone-200 shadow-sm space-y-5">
                  <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 border-b border-stone-100 pb-3">
                    <div>
                      <h2 className="font-serif-scholarly font-bold text-lg text-stone-900 flex items-center gap-2">
                        <span>Multi-Candidate Comparative OCR Analysis</span>
                        <StatusBadge status={ocrResult.status} size="sm" />
                      </h2>
                      <p className="text-xs text-stone-500 mt-0.5">
                        Authentic Tesseract extraction executed independently on all pipeline variants
                      </p>
                    </div>
                  </div>

                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    {Object.entries(ocrResult.comparison_results).map(([candKey, candData]) => {
                      const isSelected = candKey === (ocrResult.image_source || 'restored');
                      return (
                        <div
                          key={candKey}
                          className={`p-4 rounded-xl border flex flex-col justify-between space-y-3 ${
                            isSelected
                              ? 'bg-emerald-50/40 border-emerald-300 ring-1 ring-emerald-400'
                              : 'bg-stone-50/60 border-stone-200'
                          }`}
                        >
                          <div className="flex items-center justify-between border-b border-stone-200/60 pb-2">
                            <div className="flex items-center gap-2">
                              <span className="text-xs font-bold uppercase tracking-wider text-stone-800">
                                {candKey.replace('_', ' ')}
                              </span>
                              {isSelected && (
                                <span className="text-[10px] bg-emerald-800 text-white font-bold px-2 py-0.5 rounded-full">
                                  Primary
                                </span>
                              )}
                            </div>
                            <div className="flex items-center gap-2 text-xs text-stone-500 font-mono">
                              <span>Chars: {candData.character_count}</span>
                              {candData.mean_confidence != null && (
                                <span>• Conf: {candData.mean_confidence.toFixed(1)}%</span>
                              )}
                              <button
                                onClick={() => handleCopyCandidate(candData.extracted_text, candKey)}
                                className="p-1 hover:bg-stone-200 rounded text-stone-600 transition cursor-pointer"
                                title="Copy text"
                              >
                                {copiedCandidate === candKey ? (
                                  <Check className="w-3.5 h-3.5 text-emerald-600" />
                                ) : (
                                  <Copy className="w-3.5 h-3.5" />
                                )}
                              </button>
                            </div>
                          </div>

                          <div className="p-3 bg-white rounded-lg border border-stone-200/70 font-serif-scholarly text-stone-900 text-sm leading-relaxed whitespace-pre-wrap min-h-[90px] max-h-[160px] overflow-y-auto">
                            {candData.extracted_text || (
                              <span className="text-stone-400 italic text-xs">
                                No characters recognized
                              </span>
                            )}
                          </div>

                          <div className="pt-1 flex justify-end">
                            <button
                              type="button"
                              onClick={() => {
                                setWorkingText(candData.extracted_text);
                                setAcceptedNotice(`Loaded "${candKey}" OCR text into working editor.`);
                                setTimeout(() => setAcceptedNotice(null), 3000);
                              }}
                              className="text-[11px] text-emerald-800 hover:text-emerald-700 font-semibold cursor-pointer"
                            >
                              Load Into Editor →
                            </button>
                          </div>
                        </div>
                      );
                    })}
                  </div>
                </div>
              ) : null}

              {/* Dual-View Extracted Text Workspace: Editable vs Raw Immutable */}
              <div className="bg-white p-6 rounded-xl border border-stone-200 shadow-sm space-y-4">
                <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 border-b border-stone-100 pb-3">
                  <div className="flex flex-wrap items-center gap-2">
                    <h2 className="font-serif-scholarly font-bold text-lg text-stone-900">
                      Extracted Manuscript Text Workspace
                    </h2>
                    <StatusBadge status={ocrResult.status} size="sm" />
                    {ocrResult.image_source && (
                      <span className="text-[11px] font-mono px-2 py-0.5 rounded bg-stone-100 text-stone-700 border border-stone-200">
                        source: {ocrResult.image_source}
                      </span>
                    )}
                    {ocrResult.preprocessing_applied && (
                      <span className="text-[11px] font-mono px-2 py-0.5 rounded bg-emerald-50 text-emerald-800 border border-emerald-200">
                        prep: {ocrResult.preprocessing_applied}
                      </span>
                    )}
                    {isTextModified && (
                      <span className="text-[10px] bg-amber-100 text-amber-900 font-semibold px-2 py-0.5 rounded border border-amber-300">
                        Edited
                      </span>
                    )}
                  </div>

                  <div className="flex flex-wrap items-center gap-2">
                    <button
                      onClick={handleCopyWorkingText}
                      className="inline-flex items-center gap-1 px-2.5 py-1.5 rounded-lg bg-stone-100 hover:bg-stone-200 text-stone-700 text-xs font-medium transition cursor-pointer"
                      title="Copy working text"
                    >
                      {copiedWorking ? (
                        <Check className="w-3.5 h-3.5 text-emerald-600" />
                      ) : (
                        <Copy className="w-3.5 h-3.5" />
                      )}
                      <span>{copiedWorking ? 'Copied' : 'Copy Text'}</span>
                    </button>

                    <button
                      onClick={handleDownloadTxt}
                      className="inline-flex items-center gap-1 px-2.5 py-1.5 rounded-lg bg-stone-100 hover:bg-stone-200 text-stone-700 text-xs font-medium transition cursor-pointer"
                      title="Download UTF-8 plain text file"
                    >
                      <Download className="w-3.5 h-3.5 text-stone-600" />
                      <span>Download .txt</span>
                    </button>

                    {isTextModified && (
                      <button
                        onClick={handleResetToRaw}
                        className="inline-flex items-center gap-1 px-2.5 py-1.5 rounded-lg bg-amber-50 hover:bg-amber-100 text-amber-800 text-xs font-medium transition cursor-pointer border border-amber-200"
                        title="Revert all edits to raw OCR output"
                      >
                        <RotateCcw className="w-3.5 h-3.5" />
                        <span>Revert to Raw</span>
                      </button>
                    )}

                    <button
                      onClick={() => setShowRawComparison(!showRawComparison)}
                      className="inline-flex items-center gap-1 px-2.5 py-1.5 rounded-lg border border-stone-200 text-stone-600 hover:text-stone-900 text-xs font-medium transition cursor-pointer"
                    >
                      <span>Raw OCR</span>
                      {showRawComparison ? (
                        <ChevronUp className="w-3.5 h-3.5" />
                      ) : (
                        <ChevronDown className="w-3.5 h-3.5" />
                      )}
                    </button>
                  </div>
                </div>

                {/* Status notice */}
                {acceptedNotice && (
                  <div className="p-3 bg-emerald-50 border border-emerald-200 text-emerald-900 rounded-lg text-xs flex items-center justify-between">
                    <span>{acceptedNotice}</span>
                    <button
                      onClick={() => setAcceptedNotice(null)}
                      className="text-emerald-700 font-bold hover:text-emerald-900"
                    >
                      ✕
                    </button>
                  </div>
                )}

                {/* Primary Editable Workspace */}
                <div className="space-y-2">
                  <div className="flex items-center justify-between text-xs text-stone-500">
                    <span className="font-semibold text-stone-700">
                      Editable Working Transcription (Unicode UTF-8)
                    </span>
                    <div className="flex items-center gap-3 font-mono text-[11px]">
                      <span>Characters: {workingText.length}</span>
                      <span>Words: {workingText.trim() ? workingText.trim().split(/\s+/).length : 0}</span>
                      {ocrResult.mean_confidence != null && (
                        <span className="text-emerald-800 font-semibold">
                          Confidence: {ocrResult.mean_confidence.toFixed(1)}%
                        </span>
                      )}
                    </div>
                  </div>

                  <textarea
                    value={workingText}
                    onChange={(e) => setWorkingText(e.target.value)}
                    rows={6}
                    placeholder="Extracted manuscript text will appear here. You can directly edit and correct OCR glyphs."
                    className="w-full p-4 bg-stone-50/50 hover:bg-stone-50 focus:bg-white rounded-lg border border-stone-300 font-serif-scholarly text-stone-900 text-lg leading-relaxed focus:ring-2 focus:ring-emerald-700 transition"
                  />
                  <p className="text-[11px] text-stone-400">
                    Tip: You can correct misspelled characters above, or click individual words below for context-aware completion.
                  </p>
                </div>

                {/* Collapsible Raw Immutable OCR Output */}
                {showRawComparison && (
                  <div className="p-4 bg-stone-100/70 rounded-lg border border-stone-300 space-y-2">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-2">
                        <span className="text-xs font-bold text-stone-700 uppercase tracking-wider">
                          Immutable Raw Tesseract Output
                        </span>
                        <span className="text-[10px] bg-stone-200 text-stone-600 px-2 py-0.5 rounded font-mono">
                          Read-Only Audit Trail
                        </span>
                      </div>
                      <button
                        onClick={handleCopyRawText}
                        className="text-xs text-stone-600 hover:text-stone-900 flex items-center gap-1 font-medium"
                      >
                        {copiedRaw ? (
                          <Check className="w-3.5 h-3.5 text-emerald-600" />
                        ) : (
                          <Copy className="w-3.5 h-3.5" />
                        )}
                        <span>{copiedRaw ? 'Copied' : 'Copy Raw'}</span>
                      </button>
                    </div>
                    <pre className="p-3 bg-white rounded border border-stone-200 font-serif-scholarly text-stone-800 text-sm whitespace-pre-wrap max-h-48 overflow-y-auto">
                      {rawOcrText || '(Empty raw OCR output)'}
                    </pre>
                  </div>
                )}

                {/* Interactive Token Selector for Damaged Text Completion */}
                {tokens.filter((t) => /\S/.test(t) && !/^[।॥,;:.!?]+$/.test(t)).length > 0 && (
                  <div className="pt-3 border-t border-stone-100 space-y-2">
                    <span className="text-xs font-semibold text-stone-700 block">
                      Interactive Folio Tokens (Click a word or damaged glyph to predict):
                    </span>
                    <div className="flex flex-wrap gap-1.5 max-h-36 overflow-y-auto p-2 bg-stone-50 rounded-lg border border-stone-200">
                      {tokens.map((token, idx) => {
                        const isWord = /\S/.test(token) && !/^[।॥,;:.!?]+$/.test(token);
                        if (!isWord) {
                          return (
                            <span key={idx} className="text-stone-400 font-serif-scholarly self-center">
                              {token}
                            </span>
                          );
                        }
                        const isSelected = selectedWordIndex === idx;
                        return (
                          <button
                            key={idx}
                            type="button"
                            onClick={() => handleSelectToken(idx, token)}
                            className={`px-2.5 py-1 rounded text-xs font-serif-scholarly transition cursor-pointer ${
                              isSelected
                                ? 'bg-amber-600 text-white font-bold shadow-xs'
                                : 'bg-white hover:bg-amber-100 text-stone-800 border border-stone-200'
                            }`}
                          >
                            {token}
                          </button>
                        );
                      })}
                    </div>
                  </div>
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
            Statistically predicts obscured characters using authentic Vedic & Hindi corpora (VedoOCR and VQA datasets). Operates directly on the extracted manuscript text above.
          </p>
        </div>

        {/* Context Input Grid */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div>
            <label className="block text-xs font-medium text-stone-700 mb-1">
              Left Context (Preceding Words)
            </label>
            <input
              type="text"
              value={leftContext}
              onChange={(e) => setLeftContext(e.target.value)}
              placeholder="e.g. अग्निमीळे पुरोहितं"
              className="w-full text-sm font-serif-scholarly border border-stone-300 rounded-lg p-2.5 bg-stone-50"
            />
          </div>

          <div>
            <label className="block text-xs font-medium text-stone-700 mb-1">
              Uncertain / Damaged Span
            </label>
            <input
              type="text"
              value={uncertainSpan}
              onChange={(e) => setUncertainSpan(e.target.value)}
              placeholder="e.g. य[?]स्य or uncertain word"
              className="w-full text-sm font-serif-scholarly border border-amber-300 rounded-lg p-2.5 bg-amber-50/50 font-bold text-stone-900"
            />
          </div>

          <div>
            <label className="block text-xs font-medium text-stone-700 mb-1">
              Right Context (Succeeding Words)
            </label>
            <input
              type="text"
              value={rightContext}
              onChange={(e) => setRightContext(e.target.value)}
              placeholder="e.g. रत्नधातमम्"
              className="w-full text-sm font-serif-scholarly border border-stone-300 rounded-lg p-2.5 bg-stone-50"
            />
          </div>
        </div>

        <div className="flex items-center gap-3">
          <button
            type="button"
            onClick={handlePredictMissingText}
            disabled={isPredicting || (!uncertainSpan.trim() && !leftContext.trim() && !rightContext.trim())}
            className="inline-flex items-center gap-2 px-4 py-2 bg-stone-800 hover:bg-stone-700 text-stone-100 rounded-lg text-xs font-semibold shadow-xs transition cursor-pointer disabled:opacity-50"
          >
            {isPredicting ? (
              <>
                <RefreshCw className="w-3.5 h-3.5 animate-spin" />
                <span>Querying Linguistic Corpora...</span>
              </>
            ) : (
              <>
                <Sparkles className="w-3.5 h-3.5 text-amber-400" />
                <span>Predict Missing / Damaged Span</span>
              </>
            )}
          </button>

          {uncertainSpan && (
            <span className="text-xs text-stone-500">
              Target: <strong className="font-serif-scholarly text-stone-800">"{uncertainSpan}"</strong>
            </span>
          )}
        </div>

        {/* Prediction Results & Accept Mechanism */}
        {predictionCandidates.length > 0 && (
          <div className="mt-4 p-5 rounded-xl bg-amber-50/70 border border-amber-200 space-y-4">
            <div className="flex items-center justify-between">
              <span className="text-xs font-bold text-stone-900 uppercase tracking-wider">
                Corpus Completion Candidates (Ranked by Transition Probability)
              </span>
              <span className="text-[11px] text-amber-800 font-mono">
                {predictionCandidates.length} suggestions
              </span>
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-4 gap-3">
              {predictionCandidates.map((c, i) => (
                <div
                  key={i}
                  className="p-3 rounded-lg bg-white border border-amber-300 shadow-xs flex flex-col justify-between space-y-2.5"
                >
                  <div className="flex items-center justify-between">
                    <span className="text-base font-bold font-serif-scholarly text-stone-900">
                      {c.word}
                    </span>
                    <span className="text-xs font-bold text-emerald-800 font-mono">
                      {(c.score * 100).toFixed(1)}%
                    </span>
                  </div>

                  {c.evidence && (
                    <p className="text-[10px] text-stone-500 italic line-clamp-2">
                      {c.evidence}
                    </p>
                  )}

                  <button
                    type="button"
                    onClick={() => handleAcceptCandidate(c.word)}
                    className="w-full py-1.5 px-2 bg-emerald-800 hover:bg-emerald-700 text-white rounded text-[11px] font-semibold transition cursor-pointer flex items-center justify-center gap-1 shadow-xs"
                  >
                    <Check className="w-3 h-3" />
                    <span>Accept & Apply</span>
                  </button>
                </div>
              ))}
            </div>

            {predictionDisclaimer && (
              <p className="text-[11px] text-amber-900/90 italic pt-2 border-t border-amber-200/80 leading-relaxed">
                ⚠️ <strong>Scholarly Disclaimer:</strong> {predictionDisclaimer}
              </p>
            )}
          </div>
        )}
      </div>
    </div>
  );
};
