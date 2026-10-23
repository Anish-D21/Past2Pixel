import React, { useState } from 'react';
import { ImageOff, Loader2 } from 'lucide-react';

interface ArtifactImageProps {
  src?: string | null;
  alt: string;
  className?: string;
  containerClassName?: string;
  fallbackLabel?: string;
}

export const ArtifactImage: React.FC<ArtifactImageProps> = ({
  src,
  alt,
  className = 'h-44 w-full object-contain rounded bg-stone-50 border border-stone-100',
  containerClassName = 'relative w-full',
  fallbackLabel = 'Artifact Unavailable',
}) => {
  const [hasError, setHasError] = useState(false);
  const [isLoading, setIsLoading] = useState(Boolean(src));

  if (!src || hasError) {
    return (
      <div
        className={`flex flex-col items-center justify-center p-4 text-center rounded bg-stone-100/70 border border-dashed border-stone-300 text-stone-500 h-44 w-full select-none`}
      >
        <ImageOff className="w-6 h-6 mb-1.5 text-stone-400" />
        <span className="text-[11px] font-medium text-stone-600">{fallbackLabel}</span>
        <span className="text-[9px] text-stone-400 mt-0.5">Asset not found or failed to load</span>
      </div>
    );
  }

  // Normalize path ensuring leading slash
  const resolvedSrc = src.startsWith('http://') || src.startsWith('https://')
    ? src
    : src.startsWith('/')
    ? src
    : `/${src}`;

  return (
    <div className={containerClassName}>
      {isLoading && (
        <div className="absolute inset-0 flex items-center justify-center bg-stone-50/80 rounded border border-stone-100 z-10">
          <Loader2 className="w-5 h-5 text-stone-400 animate-spin" />
        </div>
      )}
      <img
        src={resolvedSrc}
        alt={alt}
        className={className}
        loading="lazy"
        onLoad={() => setIsLoading(false)}
        onError={() => {
          setIsLoading(false);
          setHasError(true);
        }}
      />
    </div>
  );
};
