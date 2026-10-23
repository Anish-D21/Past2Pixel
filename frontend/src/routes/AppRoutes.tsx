import React from 'react';
import { Routes, Route, Navigate } from 'react-router-dom';
import { ProtectedRoute } from '../components/ProtectedRoute';
import { AppLayout } from '../layouts/AppLayout';
import { Login } from '../pages/Login';
import { Register } from '../pages/Register';
import { Dashboard } from '../pages/Dashboard';
import { RestoreManuscript } from '../pages/RestoreManuscript';
import { OcrExtraction } from '../pages/OcrExtraction';
import { ResultsAnalysis } from '../pages/ResultsAnalysis';
import { RestorationHistory } from '../pages/RestorationHistory';
import { PipelineExplanation } from '../pages/PipelineExplanation';
import { UserProfile } from '../pages/UserProfile';
import { NotFound } from '../pages/NotFound';

export const AppRoutes: React.FC = () => {
  return (
    <Routes>
      {/* Public Authentication Routes */}
      <Route path="/login" element={<Login />} />
      <Route path="/register" element={<Register />} />

      {/* Protected Research Workspace */}
      <Route
        path="/"
        element={
          <ProtectedRoute>
            <AppLayout />
          </ProtectedRoute>
        }
      >
        <Route index element={<Navigate to="/dashboard" replace />} />
        <Route path="dashboard" element={<Dashboard />} />
        <Route path="restore" element={<RestoreManuscript />} />
        <Route path="ocr" element={<OcrExtraction />} />
        <Route path="results/:sessionId" element={<ResultsAnalysis />} />
        <Route path="results" element={<ResultsAnalysis />} />
        <Route path="history" element={<RestorationHistory />} />
        <Route path="explainable-ai" element={<PipelineExplanation />} />
        <Route path="profile" element={<UserProfile />} />
      </Route>

      {/* 404 Fallback */}
      <Route path="*" element={<NotFound />} />
    </Routes>
  );
};
