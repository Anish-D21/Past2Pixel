import React, { useState, useEffect } from 'react';
import { useSearchParams } from 'react-router-dom';
import { api } from '../services/api';
import { OCRResult, TextPredictionCandidate } from '../types/api';
import { StatusBadge } from '../components/StatusBadge';
import {
  FileText,
  Copy,
  Check,
  AlertTriangle,
  Sparkles,
  RefreshCw,
  HelpCircle,
  Terminal,
} from 'lucide-react';

export const OcrExtraction: React.FC = () => {
  const [searchParams] = useSearchParams();
  const sessionFromQuery = searchParams.get('session') || '';

  const [sessionId, setSessionId] = useState<string>(sessionFromQuery);
  const [language, setLanguage] = useState<string>('san+hin');
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

  useEffect(() => {
    if (sessionFromQuery) {
      setSessionId(sessionFromQuery);
    }
  }, [sessionFromQuery]);

  const handleExtractOcr = async () => {
    if (!sessionId.trim()) {
      setError('Please provide a valid session ID containing an uploaded manuscript folio.');
      return;
    }

    setIsProcessing(true);
    setError(null);
    setOcrResult(null);

    try {
      const res = await api.ocr.extractText(sessionId.trim(), language);
      setOcrResult(res);
    } catch (err: any) {
      setError(err.message || 'Failed to execute OCR text extraction.');
    } finally {
      setIsProcessing(false);
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

      {error && (
        <div className="p-4 rounded-lg bg-rose-50 border border-rose-200 text-rose-900 text-xs flex items-start gap-3">
          <AlertTriangle className="w-5 h-5 text-rose-600 shrink-0 mt-0.5" />
          <div>
            <div className="font-semibold">OCR Pipeline Notice</div>
            <div>{error}</div>
          </div>
        </div>
      )}

      {/* Control Card */}
      <div className="bg-white p-6 rounded-xl border border-stone-200 shadow-xs space-y-4">
        <h2 className="font-serif-scholarly font-bold text-base text-stone-900 border-b border-stone-100 pb-2">
          Folio Ingestion & Language Parameters
        </h2>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="md:col-span-2">
            <label className="block text-xs font-medium text-stone-700 mb-1">
              Restoration Session ID
            </label>
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
              Target Script / Language Pack
            </label>
            <select
              value={language}
              onChange={(e) => setLanguage(e.target.value)}
              className="w-full text-xs border border-stone-300 rounded-lg p-2.5 bg-stone-50 focus:ring-2 focus:ring-emerald-700"
            >
              <option value="san+hin">Sanskrit + Hindi (Devanagari)</option>
              <option value="san">Vedic Sanskrit Only (san)</option>
              <option value="hin">Classical Hindi (hin)</option>
              <option value="eng">English Transliteration (eng)</option>
            </select>
          </div>
        </div>

        <button
          type="button"
          onClick={handleExtractOcr}
          disabled={!sessionId.trim() || isProcessing}
          className="inline-flex items-center gap-2 px-5 py-2.5 bg-emerald-800 hover:bg-emerald-700 text-white rounded-lg shadow-sm text-xs font-semibold transition disabled:opacity-50 cursor-pointer"
        >
          {isProcessing ? (
            <>
              <RefreshCw className="w-4 h-4 animate-spin" />
              <span>Scanning Folio Glyphs...</span>
            </>
          ) : (
            <>
              <FileText className="w-4 h-4" />
              <span>Execute Character Recognition</span>
            </>
          )}
        </button>
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
                    {copied ? <Check className="w-3.5 h-3.5 text-emerald-600" /> : <Copy className="w-3.5 h-3.5" />}
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
