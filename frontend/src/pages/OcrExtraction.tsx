import React, { useState, useEffect, useRef } from 'react';
import { useSearchParams, Link } from 'react-router-dom';
import { api } from '../services/api';
import {
  OCRResult,
  OCRStatusInfo,
  TextPredictionCandidate,
  TextPredictionResponse,
  RestorationSession,
  SessionDetailResponse,
  OCRLineDetail,
  OCRWordDetail,
  ImageGroundedCorrection,
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
  Zap,
  ListFilter,
  BarChart2,
  HelpCircle,
  Eye,
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

  // Maximum Recovery Multi-Pass State
  const [extractionMode, setExtractionMode] = useState<'multi-pass' | 'single-pass'>('multi-pass');
  const [selectedPsm, setSelectedPsm] = useState<number>(6);

  // Active Workspace Sub-Tab ('transcription' | 'raw' | 'lines' | 'passes')
  const [activeTab, setActiveTab] = useState<'transcription' | 'raw' | 'lines' | 'passes'>('transcription');
  const [inspectedWord, setInspectedWord] = useState<OCRWordDetail | null>(null);

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
  const [inputSentence, setInputSentence] = useState<string>('');
  const [uncertainSpan, setUncertainSpan] = useState<string>('');
  const [leftContext, setLeftContext] = useState<string>('');
  const [rightContext, setRightContext] = useState<string>('');
  const [isPredicting, setIsPredicting] = useState<boolean>(false);
  const [predictionResponse, setPredictionResponse] = useState<TextPredictionResponse | null>(null);
  const [predictionCandidates, setPredictionCandidates] = useState<TextPredictionCandidate[]>([]);
  const [selectedCandidateIndex, setSelectedCandidateIndex] = useState<number>(0);
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
      setRawOcrText(ocrResult.raw_extracted_text || ocrResult.extracted_text);
      setWorkingText(ocrResult.corrected_text || ocrResult.extracted_text);
      setSelectedWordIndex(null);
      setInspectedWord(null);
      setUncertainSpan('');
      setLeftContext('');
      setRightContext('');
      setPredictionCandidates([]);
      setPredictionDisclaimer(null);
      setAcceptedNotice(null);
      setActiveTab('transcription');
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
          preprocessing,
          extractionMode,
          extractionMode === 'single-pass' ? selectedPsm : undefined
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
        const res = await api.ocr.extractDirect(
          selectedFile,
          language,
          preprocessing,
          extractionMode,
          extractionMode === 'single-pass' ? selectedPsm : undefined
        );
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

  // UTF-8 Plain Text Download with Comprehensive Provenance Header
  const handleDownloadTxt = () => {
    if (!workingText) return;
    const header = [
      '================================================================',
      'PAST2PIXEL MANUSCRIPT OCR & TEXT PRESERVATION REPORT',
      `Timestamp: ${new Date().toISOString()}`,
      `Source Mode: ${inputMode === 'session' ? `Session (${sessionId})` : 'Direct Image Upload'}`,
      `Target Image: ${inputMode === 'session' ? targetImage : (selectedFile?.name || 'Uploaded File')}`,
      `OCR Language: ${language}`,
      `Extraction Mode: ${ocrResult?.is_multi_pass ? 'Multi-Pass Reconciled (Maximum Text Recovery)' : 'Single-Pass'}`,
      `Preprocessing Applied: ${ocrResult?.preprocessing_applied || preprocessing}`,
      `Total Lines Extracted: ${ocrResult?.lines_count || workingText.split('\n').filter(Boolean).length}`,
      `Total Characters: ${workingText.length}`,
      `Mean Engine Confidence: ${ocrResult?.mean_confidence != null ? `${ocrResult.mean_confidence.toFixed(1)}%` : 'N/A'}`,
      `Uncertain Lines Flagged: ${ocrResult?.uncertain_lines_count || 0}`,
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

    const beforeTokens = tokens.slice(0, index).filter((t) => /\S/.test(t) && !/^[।॥,;:.!?]+$/.test(t));
    const afterTokens = tokens.slice(index + 1).filter((t) => /\S/.test(t) && !/^[।॥,;:.!?]+$/.test(t));

    const left = beforeTokens.slice(-3).join(' ');
    const right = afterTokens.slice(0, 3).join(' ');

    setLeftContext(left);
    setRightContext(right);

    // Build a sentence window around the selected token
    const sentenceWindow = [left, word, right].filter(Boolean).join(' ');
    setInputSentence(sentenceWindow);

    // Look up word token from image-grounded words_detail
    const matchedWord = ocrResult?.words_detail?.find(
      (w) => w.token_index === index || w.word === word || w.original_word === word
    );
    setInspectedWord(matchedWord || null);

    setPredictionCandidates([]);
    setPredictionResponse(null);
    setSelectedCandidateIndex(0);
    setPredictionDisclaimer(null);
  };

  // Replace a word directly from Word Inspector alternative readings
  const handleApplyWordAlternative = (targetOriginalWord: string, replacementWord: string) => {
    if (workingText.includes(targetOriginalWord)) {
      const updated = workingText.replace(targetOriginalWord, replacementWord);
      setWorkingText(updated);
      setAcceptedNotice(`Applied image-grounded reading: replaced "${targetOriginalWord}" with "${replacementWord}".`);
    } else {
      setWorkingText((prev) => `${prev} ${replacementWord}`.trim());
      setAcceptedNotice(`Inserted candidate "${replacementWord}" into working text.`);
    }
    setInspectedWord(null);
    setTimeout(() => setAcceptedNotice(null), 4000);
  };

  // Trigger Context-Aware Damaged Text Completion
  const handlePredictMissingText = async () => {
    if (!inputSentence.trim() && !uncertainSpan.trim() && !leftContext.trim() && !rightContext.trim()) {
      setError('Please enter a sentence or specify an uncertain span or context.');
      return;
    }

    setIsPredicting(true);
    setAcceptedNotice(null);
    setError(null);
    try {
      const res = await api.prediction.predictText({
        sentence: inputSentence.trim() || undefined,
        full_text: workingText,
        uncertain_span: uncertainSpan.trim() || undefined,
        span_index: selectedWordIndex != null ? selectedWordIndex : undefined,
        left_context: leftContext.trim() || undefined,
        right_context: rightContext.trim() || undefined,
        allow_multi_word: true,
        max_candidates: 5,
      });

      setPredictionResponse(res);
      setPredictionCandidates(res.candidates || []);
      setSelectedCandidateIndex(0);
      setPredictionDisclaimer(res.disclaimer);
    } catch (err: any) {
      setError(err.message || 'Linguistic completion prediction failed.');
    } finally {
      setIsPredicting(false);
    }
  };

  // Apply reconstructed sentence or candidate word to workingText
  const handleApplyReconstruction = (chosenCandidateIndex?: number) => {
    const idx = chosenCandidateIndex !== undefined ? chosenCandidateIndex : selectedCandidateIndex;
    const candidate = predictionCandidates[idx] || predictionResponse?.candidates?.[0];
    const candidateWord = candidate?.word || '';
    const reconstructed = candidate?.reconstructed_sentence || predictionResponse?.reconstructed_sentence;
    const original = predictionResponse?.original_sentence || inputSentence;

    if (reconstructed && original && workingText.includes(original)) {
      const updated = workingText.replace(original, reconstructed);
      setWorkingText(updated);
      setAcceptedNotice(`Applied reconstructed sentence: "${reconstructed}".`);
    } else if (reconstructed && workingText.trim() === '') {
      setWorkingText(reconstructed);
      setAcceptedNotice(`Applied reconstructed sentence: "${reconstructed}".`);
    } else if (uncertainSpan && workingText.includes(uncertainSpan) && candidateWord) {
      const updated = workingText.replace(uncertainSpan, candidateWord);
      setWorkingText(updated);
      setAcceptedNotice(`Applied correction: replaced "${uncertainSpan}" with "${candidateWord}".`);
    } else if (selectedWordIndex != null && selectedWordIndex >= 0 && selectedWordIndex < tokens.length && candidateWord) {
      const originalWord = tokens[selectedWordIndex];
      const newTokens = [...tokens];
      newTokens[selectedWordIndex] = candidateWord;
      const updatedText = newTokens.join('');
      setWorkingText(updatedText);
      setAcceptedNotice(`Applied correction: replaced "${originalWord}" with "${candidateWord}".`);
    } else if (reconstructed) {
      setWorkingText((prev) => (prev ? `${prev}\n${reconstructed}` : reconstructed));
      setAcceptedNotice(`Added reconstructed sentence: "${reconstructed}".`);
    } else if (candidateWord) {
      setWorkingText((prev) => `${prev} ${candidateWord}`.trim());
      setAcceptedNotice(`Inserted candidate "${candidateWord}" into working text.`);
    }

    setTimeout(() => setAcceptedNotice(null), 5000);
  };

  // Accept a predicted candidate word directly
  const handleAcceptCandidate = (candidateWord: string, candidateIdx: number) => {
    setSelectedCandidateIndex(candidateIdx);
    handleApplyReconstruction(candidateIdx);
  };

  // Send a line from Line Inspector into Completion
  const handleInspectLineInPredictor = (line: OCRLineDetail) => {
    setInputSentence(line.text);
    setUncertainSpan('[MISSING]');
    const words = line.text.split(/\s+/).filter(Boolean);
    if (words.length > 2) {
      setLeftContext(words.slice(0, 2).join(' '));
      setRightContext(words.slice(2).join(' '));
    } else {
      setLeftContext(line.text);
      setRightContext('');
    }
    setPredictionResponse(null);
    setPredictionCandidates([]);
    setSelectedCandidateIndex(0);
    setAcceptedNotice(`Loaded line #${line.line_num} into Context-Aware Damaged Text Completion below.`);
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
          Evidence-based Maximum Text Recovery (Multi-Pass Reconciled OCR) for Vedic Sanskrit, Hindi, and English manuscripts
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
            <span>Folio Ingestion & Extraction Parameters</span>
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
                  className="w-full text-xs border border-stone-300 rounded-lg p-2.5 bg-stone-50 focus:ring-2 focus:ring-emerald-700 font-medium"
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

        {/* Strategy & Script Selection Row */}
        <div className={`grid grid-cols-1 ${extractionMode === 'single-pass' ? 'md:grid-cols-5' : 'md:grid-cols-4'} gap-4 pt-2 border-t border-stone-100`}>
          {/* 1. Language Pack */}
          <div>
            <label className="block text-xs font-medium text-stone-700 mb-1">
              Target Script / Pack
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

          {/* 2. Extraction Strategy Mode */}
          <div>
            <label className="block text-xs font-medium text-stone-700 mb-1 flex items-center gap-1">
              <Zap className="w-3.5 h-3.5 text-amber-500" />
              <span>Extraction Strategy</span>
            </label>
            <select
              value={extractionMode}
              onChange={(e) => setExtractionMode(e.target.value as 'multi-pass' | 'single-pass')}
              className="w-full text-xs border border-stone-300 rounded-lg p-2.5 bg-amber-50/50 font-semibold text-amber-950 focus:ring-2 focus:ring-amber-600"
            >
              <option value="multi-pass">⚡ Maximum Text Recovery (Multi-Pass)</option>
              <option value="single-pass">Single-Pass (Targeted PSM)</option>
            </select>
            <p className="text-[10px] text-stone-400 mt-1">
              {extractionMode === 'multi-pass'
                ? 'Runs 4 complementary passes & reconciles lines.'
                : 'Runs one targeted pass with chosen PSM.'}
            </p>
          </div>

          {/* 3. Preprocessing (Available for both Multi-Pass and Single-Pass) */}
          <div>
            <label className="block text-xs font-medium text-stone-700 mb-1 flex items-center gap-1.5">
              <Sliders className="w-3.5 h-3.5 text-stone-500" />
              <span>Image Preprocessing</span>
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
              {preprocessing === 'upscale'
                ? 'High-fidelity 2× Lanczos resampling with stroke preservation.'
                : 'Adaptive enhancement for historical manuscript scans.'}
            </p>
          </div>

          {/* 4. Page Segmentation (PSM) for Single-Pass */}
          {extractionMode === 'single-pass' && (
            <div>
              <label className="block text-xs font-medium text-stone-700 mb-1 flex items-center gap-1.5">
                <Sliders className="w-3.5 h-3.5 text-stone-500" />
                <span>Page Segmentation (PSM)</span>
              </label>
              <select
                value={selectedPsm}
                onChange={(e) => setSelectedPsm(Number(e.target.value))}
                className="w-full text-xs border border-stone-300 rounded-lg p-2.5 bg-stone-50 focus:ring-2 focus:ring-emerald-700 font-mono"
              >
                <option value={6}>PSM 6 — Single Uniform Text Block (Default)</option>
                <option value={3}>PSM 3 — Fully Automatic Page Layout</option>
                <option value={11}>PSM 11 — Sparse Text & Marginalia</option>
                <option value={4}>PSM 4 — Single Column of Variable Sizes</option>
              </select>
              <p className="text-[10px] text-stone-400 mt-1">
                Tesseract layout segmentation policy.
              </p>
            </div>
          )}

          {/* 4. Action Button */}
          <div className="flex items-end">
            <button
              type="button"
              onClick={handleExtractOcr}
              disabled={
                isProcessing ||
                (inputMode === 'session' && !sessionId.trim()) ||
                (inputMode === 'direct' && !selectedFile)
              }
              className="inline-flex items-center gap-2 px-5 py-2.5 bg-emerald-800 hover:bg-emerald-700 text-white rounded-lg shadow-sm text-xs font-semibold transition disabled:opacity-50 cursor-pointer w-full justify-center"
            >
              {isProcessing ? (
                <>
                  <RefreshCw className="w-4 h-4 animate-spin" />
                  <span>
                    {extractionMode === 'multi-pass'
                      ? 'Reconciling Multi-Pass Glyphs...'
                      : 'Scanning Folio Glyphs...'}
                  </span>
                </>
              ) : (
                <>
                  {extractionMode === 'multi-pass' ? (
                    <Zap className="w-4 h-4 text-amber-300" />
                  ) : (
                    <FileText className="w-4 h-4" />
                  )}
                  <span>
                    {extractionMode === 'multi-pass'
                      ? 'Execute Maximum Text Recovery'
                      : 'Execute Single-Pass OCR'}
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

              {/* Main Workspace Card */}
              <div className="bg-white p-6 rounded-xl border border-stone-200 shadow-sm space-y-4">
                {/* Header with Badges & Action Buttons */}
                <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 border-b border-stone-100 pb-3">
                  <div className="flex flex-wrap items-center gap-2">
                    <h2 className="font-serif-scholarly font-bold text-lg text-stone-900">
                      Manuscript Text Workspace
                    </h2>
                    <StatusBadge status={ocrResult.status} size="sm" />

                    {ocrResult.is_multi_pass && (
                      <span className="text-[11px] font-semibold px-2 py-0.5 rounded bg-amber-100 text-amber-900 border border-amber-300 flex items-center gap-1">
                        <Zap className="w-3 h-3 text-amber-600" />
                        <span>Multi-Pass Reconciled</span>
                      </span>
                    )}

                    {ocrResult.image_source && (
                      <span className="text-[11px] font-mono px-2 py-0.5 rounded bg-stone-100 text-stone-700 border border-stone-200">
                        source: {ocrResult.image_source}
                      </span>
                    )}

                    {(ocrResult.super_resolution_applied || ocrResult.preprocessing_applied === 'upscale') && (
                      <span className="text-[11px] font-semibold px-2 py-0.5 rounded bg-emerald-100 text-emerald-900 border border-emerald-300 flex items-center gap-1">
                        <Sparkles className="w-3 h-3 text-emerald-700" />
                        <span>2× Super-Resolution</span>
                        {ocrResult.original_dimensions && ocrResult.processed_dimensions && (
                          <span className="font-mono text-[10px] text-emerald-800">
                            ({ocrResult.original_dimensions[0]}×{ocrResult.original_dimensions[1]} → {ocrResult.processed_dimensions[0]}×{ocrResult.processed_dimensions[1]})
                          </span>
                        )}
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

                {/* Sub-Tabs: Transcription vs Raw Audit vs Line Inspector vs Multi-Pass Diagnostic */}
                <div className="flex flex-wrap items-center gap-2 border-b border-stone-100 pb-2">
                  <button
                    type="button"
                    onClick={() => setActiveTab('transcription')}
                    className={`px-3 py-1.5 rounded-lg text-xs font-semibold flex items-center gap-1.5 cursor-pointer transition ${
                      activeTab === 'transcription'
                        ? 'bg-emerald-800 text-white shadow-xs'
                        : 'text-stone-600 hover:bg-stone-100'
                    }`}
                  >
                    <FileText className="w-3.5 h-3.5" />
                    <span>Corrected Transcription (Image-Supported)</span>
                    {ocrResult.corrections_count ? (
                      <span className="text-[10px] bg-emerald-600 text-white px-1.5 py-0.2 rounded-full font-bold">
                        +{ocrResult.corrections_count}
                      </span>
                    ) : null}
                  </button>

                  <button
                    type="button"
                    onClick={() => setActiveTab('raw')}
                    className={`px-3 py-1.5 rounded-lg text-xs font-semibold flex items-center gap-1.5 cursor-pointer transition ${
                      activeTab === 'raw'
                        ? 'bg-emerald-800 text-white shadow-xs'
                        : 'text-stone-600 hover:bg-stone-100'
                    }`}
                  >
                    <Terminal className="w-3.5 h-3.5" />
                    <span>Raw OCR Audit Trail</span>
                  </button>

                  <button
                    type="button"
                    onClick={() => setActiveTab('lines')}
                    className={`px-3 py-1.5 rounded-lg text-xs font-semibold flex items-center gap-1.5 cursor-pointer transition ${
                      activeTab === 'lines'
                        ? 'bg-emerald-800 text-white shadow-xs'
                        : 'text-stone-600 hover:bg-stone-100'
                    }`}
                  >
                    <ListFilter className="w-3.5 h-3.5" />
                    <span>Line Inspector ({ocrResult.lines_count || ocrResult.lines_detail?.length || 0})</span>
                    {ocrResult.uncertain_lines_count ? (
                      <span className="text-[10px] bg-amber-500 text-white px-1.5 py-0.2 rounded-full font-bold">
                        {ocrResult.uncertain_lines_count}
                      </span>
                    ) : null}
                  </button>

                  {ocrResult.pass_summaries && (
                    <button
                      type="button"
                      onClick={() => setActiveTab('passes')}
                      className={`px-3 py-1.5 rounded-lg text-xs font-semibold flex items-center gap-1.5 cursor-pointer transition ${
                        activeTab === 'passes'
                          ? 'bg-emerald-800 text-white shadow-xs'
                          : 'text-stone-600 hover:bg-stone-100'
                      }`}
                    >
                      <BarChart2 className="w-3.5 h-3.5" />
                      <span>Diagnostics & Quality</span>
                    </button>
                  )}
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

                {/* TAB 1: CORRECTED IMAGE-SUPPORTED TRANSCRIPTION */}
                {activeTab === 'transcription' && (
                  <div className="space-y-4">
                    <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 text-xs text-stone-500">
                      <div className="flex flex-wrap items-center gap-2">
                        <span className="font-semibold text-stone-800">
                          Image-Supported Transcription (Editable UTF-8)
                        </span>
                        {ocrResult.corrections_count ? (
                          <span className="text-[10px] bg-emerald-100 text-emerald-900 font-semibold px-2 py-0.5 rounded border border-emerald-300">
                            ✨ {ocrResult.corrections_count} Image-Grounded Corrections
                          </span>
                        ) : null}
                        {ocrResult.uncertain_spans_count ? (
                          <span className="text-[10px] bg-amber-100 text-amber-900 font-semibold px-2 py-0.5 rounded border border-amber-300">
                            ⚠️ {ocrResult.uncertain_spans_count} Flagged for Review
                          </span>
                        ) : null}
                        {ocrResult.deskew_applied ? (
                          <span className="text-[10px] bg-stone-100 text-stone-700 font-mono px-2 py-0.5 rounded border border-stone-200">
                            📐 Skew Corrected
                          </span>
                        ) : null}
                      </div>

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
                      rows={8}
                      placeholder="Extracted manuscript text will appear here. You can directly edit and correct OCR glyphs."
                      className="w-full p-4 bg-stone-50/50 hover:bg-stone-50 focus:bg-white rounded-lg border border-stone-300 font-serif-scholarly text-stone-900 text-lg leading-relaxed focus:ring-2 focus:ring-emerald-700 transition"
                    />

                    {/* Interactive Folio Word Tokens */}
                    {tokens.filter((t) => /\S/.test(t) && !/^[।॥,;:.!?]+$/.test(t)).length > 0 && (
                      <div className="pt-2 border-t border-stone-100 space-y-2">
                        <div className="flex items-center justify-between">
                          <span className="text-xs font-semibold text-stone-700 block">
                            Interactive Folio Tokens (Click any token to inspect visual crop & confidence):
                          </span>
                          <span className="text-[10px] text-stone-400">
                            Green = Verified • Amber = Uncertain • Emerald = Corrected
                          </span>
                        </div>
                        <div className="flex flex-wrap gap-1.5 max-h-40 overflow-y-auto p-2.5 bg-stone-50 rounded-lg border border-stone-200">
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
                            const wordDetail = ocrResult.words_detail?.find(
                              (w) => w.token_index === idx || w.word === token || w.original_word === token
                            );
                            const isCorrected = wordDetail?.status === 'IMAGE_SUPPORTED_CORRECTION';
                            const isUncertain = wordDetail?.status === 'UNCERTAIN_REVIEW_NEEDED' || (wordDetail?.confidence ?? 100) < 60;
                            const isDamaged = wordDetail?.status === 'DAMAGED_UNREADABLE' || token === '[DAMAGED]';

                            let badgeStyle = 'bg-white hover:bg-stone-100 text-stone-800 border-stone-200';
                            if (isSelected) {
                              badgeStyle = 'bg-amber-600 text-white font-bold shadow-xs';
                            } else if (isDamaged) {
                              badgeStyle = 'bg-rose-100 text-rose-900 border-rose-300 font-bold';
                            } else if (isCorrected) {
                              badgeStyle = 'bg-emerald-100 text-emerald-950 border-emerald-300 font-semibold';
                            } else if (isUncertain) {
                              badgeStyle = 'bg-amber-100 text-amber-950 border-amber-300 font-semibold';
                            }

                            return (
                              <button
                                key={idx}
                                type="button"
                                onClick={() => handleSelectToken(idx, token)}
                                className={`px-2 py-0.5 rounded text-xs font-serif-scholarly transition cursor-pointer border flex items-center gap-1 ${badgeStyle}`}
                              >
                                {isCorrected && !isSelected && <span className="text-[10px]">✨</span>}
                                {isUncertain && !isSelected && <span className="text-[10px]">⚠️</span>}
                                {isDamaged && !isSelected && <span className="text-[10px]">❌</span>}
                                <span>{token}</span>
                              </button>
                            );
                          })}
                        </div>
                      </div>
                    )}

                    {/* MANUSCRIPT IMAGE PATCH INSPECTOR CARD */}
                    {inspectedWord && (
                      <div className="p-4 bg-gradient-to-r from-amber-50/80 via-white to-stone-50 rounded-xl border border-amber-300 shadow-xs space-y-3">
                        <div className="flex items-center justify-between border-b border-amber-200/80 pb-2">
                          <div className="flex items-center gap-2">
                            <Eye className="w-4 h-4 text-amber-700" />
                            <span className="text-xs font-bold text-stone-900 uppercase tracking-wider">
                              Manuscript Physical Patch Evidence
                            </span>
                            <span
                              className={`text-[10px] font-bold px-2 py-0.5 rounded ${
                                inspectedWord.status === 'IMAGE_SUPPORTED_CORRECTION'
                                  ? 'bg-emerald-200 text-emerald-950'
                                  : inspectedWord.status === 'DAMAGED_UNREADABLE'
                                  ? 'bg-rose-200 text-rose-950'
                                  : inspectedWord.is_uncertain
                                  ? 'bg-amber-200 text-amber-950'
                                  : 'bg-stone-200 text-stone-800'
                              }`}
                            >
                              {inspectedWord.status.replace(/_/g, ' ')}
                            </span>
                          </div>

                          <button
                            type="button"
                            onClick={() => setInspectedWord(null)}
                            className="text-stone-400 hover:text-stone-700 text-xs font-bold p-1 cursor-pointer"
                          >
                            ✕
                          </button>
                        </div>

                        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 items-center">
                          {/* 1. Visual Crop */}
                          <div className="flex flex-col items-center justify-center p-2 bg-stone-100 rounded-lg border border-stone-200 min-h-[70px]">
                            {inspectedWord.patch_metrics?.patch_data_uri ? (
                              <img
                                src={inspectedWord.patch_metrics.patch_data_uri}
                                alt={`Patch for ${inspectedWord.word}`}
                                className="h-14 max-w-full object-contain rounded bg-white p-1 border border-stone-300"
                              />
                            ) : (
                              <span className="text-[11px] text-stone-400 italic">No image crop available</span>
                            )}
                            <div className="flex items-center gap-2 text-[10px] text-stone-500 font-mono mt-1">
                              <span>
                                Stroke: {((inspectedWord.patch_metrics?.stroke_density || 0) * 100).toFixed(1)}%
                              </span>
                              <span>•</span>
                              <span>
                                Contrast: {((inspectedWord.patch_metrics?.contrast_ratio || 0) * 100).toFixed(0)}%
                              </span>
                            </div>
                          </div>

                          {/* 2. Reading & Evidence */}
                          <div className="space-y-1">
                            <div className="flex items-center gap-2">
                              <span className="text-base font-bold font-serif-scholarly text-stone-900">
                                "{inspectedWord.word}"
                              </span>
                              <span className="text-xs font-mono font-bold text-emerald-800">
                                {inspectedWord.confidence.toFixed(1)}% conf
                              </span>
                            </div>
                            {inspectedWord.original_word !== inspectedWord.word && (
                              <p className="text-[11px] text-stone-500">
                                Original raw OCR: <del>{inspectedWord.original_word}</del>
                              </p>
                            )}
                            <p className="text-[11px] text-stone-600 italic">
                              {inspectedWord.evidence || 'Analyzed against manuscript stroke geometry.'}
                            </p>
                          </div>

                          {/* 3. Alternative Readings Actions */}
                          <div className="space-y-2">
                            {inspectedWord.alternatives && inspectedWord.alternatives.length > 0 ? (
                              <div className="space-y-1">
                                <span className="text-[10px] font-bold text-stone-500 uppercase tracking-wider block">
                                  Alternative Multi-Pass Readings:
                                </span>
                                <div className="flex flex-wrap gap-1.5">
                                  {inspectedWord.alternatives.map((alt, aIdx) => (
                                    <button
                                      key={aIdx}
                                      type="button"
                                      onClick={() => handleApplyWordAlternative(inspectedWord.word, alt)}
                                      className="px-2.5 py-1 bg-white hover:bg-emerald-50 text-emerald-900 border border-emerald-300 rounded text-xs font-serif-scholarly transition cursor-pointer shadow-2xs font-semibold"
                                    >
                                      Use "{alt}"
                                    </button>
                                  ))}
                                </div>
                              </div>
                            ) : null}

                            <div className="flex items-center gap-2 pt-1">
                              <button
                                type="button"
                                onClick={() => handleApplyWordAlternative(inspectedWord.word, '[DAMAGED]')}
                                className="text-[10px] font-semibold text-rose-800 hover:text-rose-950 bg-rose-50 hover:bg-rose-100 px-2 py-1 rounded border border-rose-200 transition cursor-pointer"
                              >
                                Mark as [DAMAGED]
                              </button>
                            </div>
                          </div>
                        </div>
                      </div>
                    )}
                  </div>
                )}

                {/* TAB 2: RAW OCR AUDIT TRAIL */}
                {activeTab === 'raw' && (
                  <div className="space-y-4">
                    <div className="flex items-center justify-between text-xs text-stone-500">
                      <div>
                        <span className="font-semibold text-stone-700">
                          Immutable Raw Tesseract Extraction Output
                        </span>
                        <span className="ml-2 text-[10px] bg-stone-200 text-stone-600 px-2 py-0.5 rounded font-mono">
                          Read-Only Audit Trail
                        </span>
                      </div>

                      <button
                        type="button"
                        onClick={handleCopyRawText}
                        className="text-xs text-stone-600 hover:text-stone-900 flex items-center gap-1 font-medium bg-stone-100 hover:bg-stone-200 px-2.5 py-1 rounded transition cursor-pointer"
                      >
                        {copiedRaw ? (
                          <Check className="w-3.5 h-3.5 text-emerald-600" />
                        ) : (
                          <Copy className="w-3.5 h-3.5" />
                        )}
                        <span>{copiedRaw ? 'Copied' : 'Copy Raw Output'}</span>
                      </button>
                    </div>

                    <pre className="p-4 bg-stone-50 rounded-lg border border-stone-300 font-serif-scholarly text-stone-900 text-base leading-relaxed whitespace-pre-wrap max-h-96 overflow-y-auto">
                      {rawOcrText || '(Empty raw OCR output)'}
                    </pre>

                    <div className="p-3 bg-stone-50 rounded-lg border border-stone-200 text-xs text-stone-600 flex items-center justify-between">
                      <span>
                        Preserves original Tesseract character extraction without image-grounded corrections or contextual edits.
                      </span>
                      <button
                        type="button"
                        onClick={handleResetToRaw}
                        className="text-amber-800 font-semibold hover:underline cursor-pointer"
                      >
                        Revert Working Text to Raw →
                      </button>
                    </div>
                  </div>
                )}

                {/* TAB 3: LINE-BY-LINE INSPECTOR & PROVENANCE */}
                {activeTab === 'lines' && (
                  <div className="space-y-3">
                    <div className="flex items-center justify-between text-xs text-stone-600 pb-1">
                      <span>
                        Structured Line-Level Breakdown with Vertical Alignment & Provenance Tracking
                      </span>
                      {ocrResult.uncertain_lines_count ? (
                        <span className="text-amber-800 font-medium">
                          ⚠️ {ocrResult.uncertain_lines_count} lines flagged for manual verification
                        </span>
                      ) : null}
                    </div>

                    <div className="space-y-2 max-h-[460px] overflow-y-auto pr-1">
                      {ocrResult.lines_detail && ocrResult.lines_detail.length > 0 ? (
                        ocrResult.lines_detail.map((line, idx) => (
                          <div
                            key={idx}
                            className={`p-3 rounded-lg border flex flex-col sm:flex-row sm:items-center justify-between gap-3 ${
                              line.is_uncertain
                                ? 'bg-amber-50/60 border-amber-300'
                                : 'bg-stone-50/50 border-stone-200'
                            }`}
                          >
                            <div className="space-y-1">
                              <div className="flex items-center gap-2 text-[11px]">
                                <span className="font-mono font-bold text-stone-700">
                                  #{line.line_num}
                                </span>
                                <span className="font-mono text-stone-400">
                                  {line.pass_name}
                                </span>
                                {line.mean_confidence != null && (
                                  <span
                                    className={`font-semibold font-mono ${
                                      line.mean_confidence >= 60
                                        ? 'text-emerald-800'
                                        : 'text-amber-700'
                                    }`}
                                  >
                                    {line.mean_confidence.toFixed(1)}% conf
                                  </span>
                                )}
                                {line.is_uncertain && (
                                  <span className="text-[10px] bg-amber-200 text-amber-900 font-bold px-1.5 rounded">
                                    Uncertain
                                  </span>
                                )}
                              </div>
                              <div className="font-serif-scholarly text-stone-900 text-base leading-snug">
                                {line.text}
                              </div>
                            </div>

                            <button
                              type="button"
                              onClick={() => handleInspectLineInPredictor(line)}
                              className="px-2.5 py-1.5 bg-stone-800 hover:bg-stone-700 text-white rounded text-[11px] font-semibold shrink-0 cursor-pointer self-start sm:self-center transition"
                            >
                              Predict Line →
                            </button>
                          </div>
                        ))
                      ) : (
                        <div className="text-xs text-stone-400 italic p-4 text-center">
                          No line-level metadata available.
                        </div>
                      )}
                    </div>
                  </div>
                )}

                {/* TAB 4: PASS BREAKDOWN & DIAGNOSTICS */}
                {activeTab === 'passes' && (
                  <div className="space-y-4">
                    {/* Resolution & Super Resolution Telemetry */}
                    {(ocrResult.original_dimensions || ocrResult.processed_dimensions) && (
                      <div className="p-3 bg-emerald-50/60 rounded-xl border border-emerald-200 flex flex-col sm:flex-row sm:items-center justify-between gap-2 text-xs">
                        <div className="flex flex-wrap items-center gap-2">
                          <span className="font-semibold text-emerald-950">Resolution & Upscaling Provenance:</span>
                          <span className="font-mono text-emerald-800">
                            Input: {ocrResult.original_dimensions ? `${ocrResult.original_dimensions[0]}×${ocrResult.original_dimensions[1]} px` : 'N/A'}
                          </span>
                          <span>→</span>
                          <span className="font-mono font-bold text-emerald-900">
                            OCR Stage: {ocrResult.processed_dimensions ? `${ocrResult.processed_dimensions[0]}×${ocrResult.processed_dimensions[1]} px` : 'N/A'}
                          </span>
                        </div>
                        {ocrResult.super_resolution_applied && (
                          <span className="px-2 py-0.5 rounded bg-emerald-200 text-emerald-900 font-bold text-[10px] font-mono self-start sm:self-auto">
                            2.0× Super-Resolution Active
                          </span>
                        )}
                      </div>
                    )}

                    {/* Image Quality Diagnostics */}
                    {ocrResult.image_quality && (
                      <div className="p-4 bg-stone-50 rounded-xl border border-stone-200 space-y-2">
                        <span className="text-xs font-bold text-stone-800 uppercase tracking-wider block">
                          Manuscript Physical Image Quality Diagnostics
                        </span>
                        <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 text-center text-xs">
                          <div className="p-2.5 bg-white rounded border border-stone-200">
                            <div className="text-stone-400 text-[10px]">Contrast Ratio</div>
                            <div className="font-bold font-mono text-stone-800">
                              {((ocrResult.image_quality.contrast_ratio || 0) * 100).toFixed(1)}%
                            </div>
                          </div>
                          <div className="p-2.5 bg-white rounded border border-stone-200">
                            <div className="text-stone-400 text-[10px]">Blur Score (Laplacian)</div>
                            <div className="font-bold font-mono text-stone-800">
                              {ocrResult.image_quality.blur_score ?? 'N/A'}
                            </div>
                          </div>
                          <div className="p-2.5 bg-white rounded border border-stone-200">
                            <div className="text-stone-400 text-[10px]">Detected Skew</div>
                            <div className="font-bold font-mono text-stone-800">
                              {ocrResult.image_quality.skew_angle_deg != null
                                ? `${ocrResult.image_quality.skew_angle_deg.toFixed(1)}°`
                                : '0.0°'}
                            </div>
                          </div>
                          <div className="p-2.5 bg-white rounded border border-stone-200">
                            <div className="text-stone-400 text-[10px]">Recommended Mode</div>
                            <div className="font-bold text-emerald-800 uppercase">
                              {ocrResult.image_quality.recommended_mode || 'Auto'}
                            </div>
                          </div>
                        </div>
                      </div>
                    )}

                    {/* Secondary Model Status */}
                    {ocrResult.secondary_model_status && (
                      <div className="p-3.5 bg-emerald-50/60 rounded-xl border border-emerald-200 text-xs space-y-1">
                        <div className="flex items-center justify-between">
                          <span className="font-bold text-emerald-950">
                            Secondary Recognition Engine: {ocrResult.secondary_model_status.name}
                          </span>
                          <span className="text-[10px] bg-emerald-200 text-emerald-900 font-bold px-2 py-0.5 rounded font-mono">
                            {ocrResult.secondary_model_status.status}
                          </span>
                        </div>
                        <p className="text-[11px] text-emerald-900/80">
                          {ocrResult.secondary_model_status.notes}
                        </p>
                      </div>
                    )}

                    {/* Pass summaries */}
                    {ocrResult.pass_summaries && (
                      <div className="space-y-2">
                        <div className="text-xs text-stone-600">
                          Multi-pass extraction runs complementary Tesseract Page Segmentation Modes (PSM) and reconciles output into a unified reading order.
                        </div>

                        <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                          {Object.entries(ocrResult.pass_summaries).map(([pKey, pSumm]) => (
                            <div
                              key={pKey}
                              className="p-3.5 bg-stone-50 rounded-xl border border-stone-200 space-y-2"
                            >
                              <div className="flex items-center justify-between">
                                <span className="text-xs font-bold text-stone-800">
                                  {pSumm.name}
                                </span>
                                <span className="text-[11px] font-mono text-stone-500">
                                  {pSumm.processing_time_ms} ms
                                </span>
                              </div>

                              <div className="grid grid-cols-3 gap-2 text-center text-[11px]">
                                <div className="p-2 bg-white rounded border border-stone-200">
                                  <div className="text-stone-400">Lines</div>
                                  <div className="font-bold text-stone-800">{pSumm.lines_count}</div>
                                </div>
                                <div className="p-2 bg-white rounded border border-stone-200">
                                  <div className="text-stone-400">Chars</div>
                                  <div className="font-bold text-stone-800">{pSumm.character_count}</div>
                                </div>
                                <div className="p-2 bg-white rounded border border-stone-200">
                                  <div className="text-stone-400">Conf</div>
                                  <div className="font-bold text-emerald-800">
                                    {pSumm.mean_confidence != null ? `${pSumm.mean_confidence}%` : 'N/A'}
                                  </div>
                                </div>
                              </div>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>
                )}
              </div>
            </div>
          )}
        </div>
      )}

      {/* Context-Aware Linguistic Completion (XAI / Preservation) */}
      <div className="bg-white p-6 rounded-xl border border-stone-200 shadow-xs space-y-5">
        <div className="border-b border-stone-100 pb-3 flex flex-col sm:flex-row sm:items-center justify-between gap-2">
          <div>
            <div className="flex items-center gap-2">
              <Sparkles className="w-5 h-5 text-amber-600" />
              <h2 className="font-serif-scholarly font-bold text-base text-stone-900">
                Context-Aware Damaged Text Completion (Linguistic N-Gram)
              </h2>
            </div>
            <p className="text-xs text-stone-500 mt-0.5">
              Statistically predicts damaged or obscured characters using authentic Vedic Sanskrit, Classical Hindi, and Historical English corpora.
            </p>
          </div>

          {workingText.trim() && (
            <button
              type="button"
              onClick={() => {
                const firstSentence = workingText.split(/[।\.\n]/).find((s) => s.trim().length > 0) || workingText;
                setInputSentence(firstSentence.trim());
                setUncertainSpan('[MISSING]');
              }}
              className="text-[11px] font-semibold text-emerald-800 hover:text-emerald-950 bg-emerald-50 hover:bg-emerald-100 px-3 py-1.5 rounded-lg border border-emerald-200 transition cursor-pointer self-start sm:self-center"
            >
              📥 Load from Extracted Text
            </button>
          )}
        </div>

        {/* Sentence Input with Placeholder Shortcuts */}
        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <label className="text-xs font-semibold text-stone-800">
              Target Sentence (Contains missing, damaged, or uncertain text):
            </label>
            <div className="flex items-center gap-1.5">
              <span className="text-[11px] text-stone-400">Quick Placeholders:</span>
              <button
                type="button"
                onClick={() => {
                  setInputSentence((prev) => (prev ? `${prev} [MISSING]` : '[MISSING]'));
                  setUncertainSpan('[MISSING]');
                }}
                className="text-[10px] bg-amber-100 hover:bg-amber-200 text-amber-900 font-mono font-bold px-2 py-0.5 rounded cursor-pointer transition"
              >
                + [MISSING]
              </button>
              <button
                type="button"
                onClick={() => {
                  setInputSentence((prev) => (prev ? `${prev} [DAMAGED]` : '[DAMAGED]'));
                  setUncertainSpan('[DAMAGED]');
                }}
                className="text-[10px] bg-amber-100 hover:bg-amber-200 text-amber-900 font-mono font-bold px-2 py-0.5 rounded cursor-pointer transition"
              >
                + [DAMAGED]
              </button>
              <button
                type="button"
                onClick={() => {
                  setInputSentence((prev) => (prev ? `${prev} ___` : '___'));
                  setUncertainSpan('___');
                }}
                className="text-[10px] bg-stone-200 hover:bg-stone-300 text-stone-800 font-mono font-bold px-2 py-0.5 rounded cursor-pointer transition"
              >
                + ___
              </button>
            </div>
          </div>

          <textarea
            value={inputSentence}
            onChange={(e) => setInputSentence(e.target.value)}
            rows={2}
            placeholder="e.g. The king [MISSING] the temple. OR The king visited the market. OR धर्मक्षेत्रे [MISSING] समवेता युयुत्सवः"
            className="w-full text-base font-serif-scholarly border border-stone-300 rounded-lg p-3 bg-stone-50/50 hover:bg-stone-50 focus:bg-white text-stone-900 focus:ring-2 focus:ring-amber-500 transition"
          />
        </div>

        {/* Granular Context Input Grid */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div>
            <label className="block text-xs font-medium text-stone-700 mb-1">
              Left Context (Preceding Words)
            </label>
            <input
              type="text"
              value={leftContext}
              onChange={(e) => setLeftContext(e.target.value)}
              placeholder="e.g. The king or अग्निमीळे"
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
              placeholder="e.g. [MISSING], [DAMAGED], visited"
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
              placeholder="e.g. the temple or पुरोहितं"
              className="w-full text-sm font-serif-scholarly border border-stone-300 rounded-lg p-2.5 bg-stone-50"
            />
          </div>
        </div>

        {/* Action Button */}
        <div className="flex flex-wrap items-center gap-3">
          <button
            type="button"
            onClick={handlePredictMissingText}
            disabled={
              isPredicting ||
              (!inputSentence.trim() && !uncertainSpan.trim() && !leftContext.trim() && !rightContext.trim())
            }
            className="inline-flex items-center gap-2 px-5 py-2.5 bg-stone-800 hover:bg-stone-700 text-white rounded-lg text-xs font-semibold shadow-xs transition cursor-pointer disabled:opacity-50"
          >
            {isPredicting ? (
              <>
                <RefreshCw className="w-3.5 h-3.5 animate-spin" />
                <span>Predicting Linguistic Transitions...</span>
              </>
            ) : (
              <>
                <Sparkles className="w-3.5 h-3.5 text-amber-400" />
                <span>Predict Missing / Damaged Span</span>
              </>
            )}
          </button>

          {(inputSentence || uncertainSpan) && (
            <span className="text-xs text-stone-500">
              Active Target:{' '}
              <strong className="font-serif-scholarly text-stone-800">
                "{uncertainSpan || inputSentence}"
              </strong>
            </span>
          )}
        </div>

        {/* Reconstructed Sentence & Candidate Review Section */}
        {(predictionResponse || predictionCandidates.length > 0) && (
          <div className="mt-4 p-5 rounded-xl bg-gradient-to-br from-amber-50/70 via-stone-50 to-emerald-50/40 border border-amber-200 space-y-4">
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 border-b border-amber-200/70 pb-3">
              <div className="flex items-center gap-2">
                <CheckCircle2 className="w-4 h-4 text-emerald-800" />
                <span className="text-xs font-bold text-stone-900 uppercase tracking-wider">
                  Linguistic N-Gram Reconstruction Result
                </span>
                {predictionResponse?.detected_script && (
                  <span className="text-[10px] bg-stone-200 text-stone-700 px-2 py-0.5 rounded font-mono font-medium">
                    Script: {predictionResponse.detected_script}
                  </span>
                )}
              </div>

              {predictionResponse?.confidence_score != null ? (
                <div className="flex items-center gap-2">
                  <span className="text-xs text-stone-500">Confidence:</span>
                  <span
                    className={`text-xs font-bold font-mono px-2 py-0.5 rounded ${
                      predictionResponse.confidence_score >= 0.5
                        ? 'bg-emerald-100 text-emerald-900'
                        : 'bg-amber-100 text-amber-900'
                    }`}
                  >
                    {(predictionResponse.confidence_score * 100).toFixed(1)}%
                  </span>
                </div>
              ) : (
                <span className="text-[11px] text-stone-400 italic">Confidence: N/A</span>
              )}
            </div>

            {/* Display: Original, Predicted Replacement, Reconstructed Sentence */}
            <div className="grid grid-cols-1 gap-3">
              {/* 1. Original Sentence */}
              <div className="p-3 bg-white rounded-lg border border-stone-200">
                <span className="text-[11px] font-bold text-stone-500 uppercase tracking-wider block mb-1">
                  Original Sentence / Context:
                </span>
                <p className="font-serif-scholarly text-stone-800 text-base">
                  {predictionResponse?.original_sentence || inputSentence || (
                    <span className="text-stone-400 italic">No sentence provided</span>
                  )}
                </p>
              </div>

              {/* 2. Predicted Missing / Corrected Text */}
              <div className="p-3 bg-amber-50/80 rounded-lg border border-amber-300">
                <span className="text-[11px] font-bold text-amber-900 uppercase tracking-wider block mb-1">
                  Predicted Missing / Corrected Text:
                </span>
                <div className="flex items-center gap-3">
                  <span className="font-serif-scholarly font-bold text-amber-950 text-xl">
                    {predictionCandidates[selectedCandidateIndex]?.word ||
                      predictionResponse?.candidates?.[0]?.word ||
                      'No prediction available'}
                  </span>
                  {predictionCandidates[selectedCandidateIndex]?.score != null && (
                    <span className="text-xs font-mono font-bold text-emerald-800 bg-emerald-100/80 px-2 py-0.5 rounded">
                      Prob: {(predictionCandidates[selectedCandidateIndex].score * 100).toFixed(1)}%
                    </span>
                  )}
                  {predictionCandidates[selectedCandidateIndex]?.evidence && (
                    <span className="text-xs text-stone-500 italic">
                      ({predictionCandidates[selectedCandidateIndex].evidence})
                    </span>
                  )}
                </div>
              </div>

              {/* 3. Complete Reconstructed Sentence */}
              <div className="p-3.5 bg-emerald-50/90 rounded-lg border border-emerald-300">
                <span className="text-[11px] font-bold text-emerald-900 uppercase tracking-wider block mb-1">
                  Complete Reconstructed Sentence:
                </span>
                <p className="font-serif-scholarly font-semibold text-emerald-950 text-lg leading-relaxed">
                  {predictionCandidates[selectedCandidateIndex]?.reconstructed_sentence ||
                    predictionResponse?.reconstructed_sentence ||
                    'Reconstruction not generated'}
                </p>
              </div>
            </div>

            {/* Candidate Alternatives Switcher */}
            {predictionCandidates.length > 1 && (
              <div className="space-y-1.5 pt-1">
                <span className="text-[11px] font-bold text-stone-600 block">
                  Alternative Candidate Replacements (Click to preview and select):
                </span>
                <div className="flex flex-wrap gap-2">
                  {predictionCandidates.map((c, i) => (
                    <button
                      key={i}
                      type="button"
                      onClick={() => setSelectedCandidateIndex(i)}
                      className={`px-3 py-1.5 rounded-lg text-xs font-serif-scholarly transition cursor-pointer flex items-center gap-2 ${
                        selectedCandidateIndex === i
                          ? 'bg-amber-600 text-white font-bold shadow-xs'
                          : 'bg-white hover:bg-stone-100 text-stone-800 border border-stone-300'
                      }`}
                    >
                      <span>{c.word}</span>
                      <span
                        className={`text-[10px] font-mono ${
                          selectedCandidateIndex === i ? 'text-amber-100' : 'text-stone-400'
                        }`}
                      >
                        {(c.score * 100).toFixed(0)}%
                      </span>
                    </button>
                  ))}
                </div>
              </div>
            )}

            {/* Apply Correction Button */}
            <div className="pt-2 flex flex-col sm:flex-row sm:items-center justify-between gap-3 border-t border-amber-200/60">
              <button
                type="button"
                onClick={() => handleApplyReconstruction()}
                className="inline-flex items-center justify-center gap-2 px-5 py-2.5 bg-emerald-800 hover:bg-emerald-700 text-white rounded-lg text-xs font-bold shadow-sm transition cursor-pointer"
              >
                <Check className="w-4 h-4" />
                <span>Apply Reconstructed Sentence to Extracted Text</span>
              </button>

              <span className="text-[11px] text-stone-500 italic">
                Updates working text while preserving original raw OCR audit trail.
              </span>
            </div>

            {/* Scholarly Disclaimer */}
            {predictionDisclaimer && (
              <p className="text-[11px] text-amber-900/90 italic pt-2 border-t border-amber-200/80 leading-relaxed">
                ⚠️ <strong>Scholarly Note:</strong> {predictionDisclaimer}
              </p>
            )}
          </div>
        )}
      </div>
    </div>
  );
};
