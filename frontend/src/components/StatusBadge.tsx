import React from 'react';

interface StatusBadgeProps {
  status: string;
  size?: 'sm' | 'md' | 'lg';
}

export const StatusBadge: React.FC<StatusBadgeProps> = ({ status, size = 'md' }) => {
  let bg = 'bg-stone-100 text-stone-700 border-stone-200';
  let dot = 'bg-stone-400';
  let label = status;

  switch (status?.toUpperCase()) {
    case 'READY_FOR_INFERENCE':
    case 'READY':
    case 'ONLINE':
    case 'SUCCESS':
    case 'RESTORED':
    case 'CONNECTED':
      bg = 'bg-emerald-50 text-emerald-800 border-emerald-300';
      dot = 'bg-emerald-600';
      label = status === 'READY_FOR_INFERENCE' ? 'READY (TRAINED)' : status;
      break;
    case 'UNTRAINED':
    case 'NOT_INITIALIZED':
    case 'UNAVAILABLE':
    case 'FAILED':
      bg = 'bg-rose-50 text-rose-800 border-rose-300';
      dot = 'bg-rose-500';
      break;
    case 'TRAINING_STARTED':
    case 'PENDING':
    case 'ANALYZED':
    case 'UPLOADED':
      bg = 'bg-amber-50 text-amber-800 border-amber-300';
      dot = 'bg-amber-500';
      break;
    case 'DEGRADED':
    case 'LOCAL_PERSISTENCE':
      bg = 'bg-blue-50 text-blue-800 border-blue-300';
      dot = 'bg-blue-500';
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
