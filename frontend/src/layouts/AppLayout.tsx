import React from 'react';
import { Outlet } from 'react-router-dom';
import { Navbar } from '../components/Navbar';
import { Sidebar } from '../components/Sidebar';

export const AppLayout: React.FC = () => {
  return (
    <div className="min-h-screen flex flex-col bg-archival-paper text-stone-900 dark:text-stone-100 transition-colors duration-200">
      {/* Top Header */}
      <Navbar />

      {/* Body with Sidebar and Main Content */}
      <div className="flex-1 flex overflow-hidden">
        {/* Left Navigation */}
        <Sidebar className="hidden md:flex" />

        {/* Dynamic Page Content */}
        <main className="flex-1 overflow-y-auto p-4 sm:p-6 lg:p-8">
          <div className="max-w-7xl mx-auto">
            <Outlet />
          </div>
        </main>
      </div>
    </div>
  );
};
