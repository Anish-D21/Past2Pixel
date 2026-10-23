import React from 'react';
import { Link } from 'react-router-dom';
import { Scroll, ArrowLeft } from 'lucide-react';

export const NotFound: React.FC = () => {
  return (
    <div className="min-h-screen flex items-center justify-center bg-archival-paper p-6 text-center">
      <div className="max-w-md w-full bg-white p-8 rounded-xl border border-stone-300 shadow-xl space-y-4">
        <div className="w-12 h-12 mx-auto rounded-full bg-stone-100 text-stone-500 flex items-center justify-center">
          <Scroll className="w-6 h-6" />
        </div>
        <h1 className="font-serif-scholarly text-3xl font-bold text-stone-900">
          404 — Folio Not Found
        </h1>
        <p className="text-xs text-stone-500 leading-relaxed">
          The manuscript page or laboratory route you requested does not exist in the Past2Pixel archival catalog.
        </p>
        <div className="pt-2">
          <Link
            to="/dashboard"
            className="inline-flex items-center gap-2 px-4 py-2 bg-emerald-800 hover:bg-emerald-700 text-white rounded-lg text-xs font-semibold shadow-xs transition"
          >
            <ArrowLeft className="w-4 h-4" />
            <span>Return to Laboratory Dashboard</span>
          </Link>
        </div>
      </div>
    </div>
  );
};
