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
  className = 'h-44 w-full object-contain rounded bg-stone-50 dark:bg-[#141A17] border border-stone-100 dark:border-[#27362E]',
  containerClassName = 'relative w-full',
  fallbackLabel = 'Artifact Unavailable',
}) => {
  const [hasError, setHasError] = useState(false);
  const [isLoading, setIsLoading] = useState(Boolean(src));

  if (!src || hasError) {
    return (
      <div
        className={`flex flex-col items-center justify-center p-4 text-center rounded bg-stone-100/70 dark:bg-[#141A17] border border-dashed border-stone-300 dark:border-[#27362E] text-stone-500 dark:text-stone-400 h-44 w-full select-none`}
      >
        <ImageOff className="w-6 h-6 mb-1.5 text-stone-400 dark:text-stone-500" />
        <span className="text-[11px] font-medium text-stone-600 dark:text-stone-300">{fallbackLabel}</span>
        <span className="text-[9px] text-stone-400 dark:text-stone-500 mt-0.5">Asset not found or failed to load</span>
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
        <div className="absolute inset-0 flex items-center justify-center bg-stone-50/80 dark:bg-[#141A17]/80 rounded border border-stone-100 dark:border-[#27362E] z-10">
          <Loader2 className="w-5 h-5 text-stone-400 dark:text-stone-500 animate-spin" />
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
