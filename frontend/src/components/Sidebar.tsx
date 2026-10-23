import React from 'react';
import { NavLink } from 'react-router-dom';
import {
  LayoutDashboard,
  Wand2,
  FileText,
  History,
  GitBranch,
  User,
  Activity,
  Layers,
} from 'lucide-react';

interface SidebarProps {
  className?: string;
}

export const Sidebar: React.FC<SidebarProps> = ({ className = '' }) => {
  const navItems = [
    {
      to: '/dashboard',
      label: 'Laboratory Dashboard',
      icon: LayoutDashboard,
      description: 'System telemetry & model status',
    },
    {
      to: '/restore',
      label: 'Manuscript Restoration',
      icon: Wand2,
      description: 'Upload & neural restoration',
    },
    {
      to: '/ocr',
      label: 'OCR & Text Extraction',
      icon: FileText,
      description: 'Sanskrit & Hindi character recognition',
    },
    {
      to: '/history',
      label: 'Restoration History',
      icon: History,
      description: 'Archived folio sessions',
    },
    {
      to: '/explainable-ai',
      label: 'Explainable AI & Report',
      icon: GitBranch,
      description: 'Pipeline audit & architecture',
    },
    {
      to: '/profile',
      label: 'Scholar Profile',
      icon: User,
      description: 'Account settings & keys',
    },
  ];

  return (
    <aside
      className={`w-64 bg-stone-900 border-r border-stone-800 flex flex-col shrink-0 text-stone-300 ${className}`}
    >
      {/* Section Header */}
      <div className="p-4 border-b border-stone-800">
        <div className="flex items-center gap-2 text-xs font-semibold text-stone-400 uppercase tracking-wider">
          <Layers className="w-3.5 h-3.5 text-emerald-400" />
          <span>Research Modules</span>
        </div>
      </div>

      {/* Nav List */}
      <nav className="flex-1 p-3 space-y-1 overflow-y-auto">
        {navItems.map((item) => {
          const Icon = item.icon;
          return (
            <NavLink
              key={item.to}
              to={item.to}
              className={({ isActive }) =>
                `flex items-start gap-3 px-3 py-2.5 rounded-lg text-sm transition group ${
                  isActive
                    ? 'bg-emerald-900/60 text-emerald-200 border border-emerald-700/50 shadow-xs'
                    : 'text-stone-400 hover:bg-stone-800/80 hover:text-stone-200'
                }`
              }
            >
              <Icon className="w-4 h-4 mt-0.5 shrink-0 text-stone-400 group-hover:text-amber-300 transition" />
              <div>
                <div className="font-medium">{item.label}</div>
                <div className="text-[11px] text-stone-500 line-clamp-1">
                  {item.description}
                </div>
              </div>
            </NavLink>
          );
        })}
      </nav>

      {/* Footer Info */}
      <div className="p-4 border-t border-stone-800 text-[11px] text-stone-500 space-y-1">
        <div className="flex items-center justify-between">
          <span>Architecture:</span>
          <span className="font-mono text-stone-400">U-Net + CNN</span>
        </div>
        <div className="flex items-center justify-between">
          <span>Datasets:</span>
          <span className="font-mono text-stone-400">VedoOCR / Sanskrit</span>
        </div>
      </div>
    </aside>
  );
};
