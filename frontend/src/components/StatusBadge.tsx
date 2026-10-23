import React from 'react';

interface StatusBadgeProps {
  status: string;
  size?: 'sm' | 'md' | 'lg';
}

export const StatusBadge: React.FC<StatusBadgeProps> = ({ status, size = 'md' }) => {
  let bg = 'bg-stone-100 text-stone-700 border-stone-200 dark:bg-[#1E2B25] dark:text-stone-300 dark:border-[#27362E]';
  let dot = 'bg-stone-400 dark:bg-stone-500';
  let label = status;

  switch (status?.toUpperCase()) {
    case 'READY_FOR_INFERENCE':
    case 'READY':
    case 'ONLINE':
    case 'SUCCESS':
    case 'RESTORED':
    case 'CONNECTED':
      bg = 'bg-emerald-50 text-emerald-800 border-emerald-300 dark:bg-emerald-950/70 dark:text-emerald-300 dark:border-emerald-800/80';
      dot = 'bg-emerald-600 dark:bg-emerald-400';
      label = status === 'READY_FOR_INFERENCE' ? 'READY (TRAINED)' : status;
      break;
    case 'UNTRAINED':
    case 'NOT_INITIALIZED':
    case 'UNAVAILABLE':
    case 'FAILED':
      bg = 'bg-rose-50 text-rose-800 border-rose-300 dark:bg-rose-950/70 dark:text-rose-300 dark:border-rose-800/80';
      dot = 'bg-rose-500 dark:bg-rose-400';
      break;
    case 'TRAINING_STARTED':
    case 'PENDING':
    case 'ANALYZED':
    case 'UPLOADED':
      bg = 'bg-amber-50 text-amber-800 border-amber-300 dark:bg-amber-950/70 dark:text-amber-300 dark:border-amber-800/80';
      dot = 'bg-amber-500 dark:bg-amber-400';
      break;
    case 'DEGRADED':
    case 'LOCAL_PERSISTENCE':
      bg = 'bg-blue-50 text-blue-800 border-blue-300 dark:bg-blue-950/70 dark:text-blue-300 dark:border-blue-800/80';
      dot = 'bg-blue-500 dark:bg-blue-400';
      break;
    default:
      break;
  }

  const sizeClasses = {
    sm: 'text-xs px-2 py-0.5',
    md: 'text-xs px-2.5 py-1',
    lg: 'text-sm px-3 py-1.5',
  }[size];

  return (
    <span
      className={`inline-flex items-center gap-1.5 font-medium rounded-full border shadow-xs ${bg} ${sizeClasses}`}
    >
      <span className={`w-1.5 h-1.5 rounded-full ${dot} animate-pulse`} />
      {label}
    </span>
  );
};
