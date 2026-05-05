import { AlertTriangle, RotateCcw } from "lucide-react";

interface ErrorDisplayProps {
  error: {
    message: string;
    details: string;
    raw?: unknown;
  };
  onRetry: () => void;
  onReset: () => void;
}

export function ErrorDisplay({ error, onRetry, onReset }: ErrorDisplayProps) {
  return (
    <div className="max-w-2xl mx-auto mt-10 p-6 border border-red-500 rounded-xl bg-red-500/10">
      <div className="flex items-center gap-3 mb-4">
        <AlertTriangle className="h-5 w-5 text-red-400" />
        <h2 className="text-xl font-semibold text-red-400">{error.message}</h2>
      </div>

      <p className="text-sm text-red-300 mb-4 wrap-break-word">{error.details}</p>

      {error.raw !== undefined && (
        <details className="text-xs text-gray-400">
          <summary className="cursor-pointer hover:text-white">Show technical details</summary>
          <pre className="mt-2 p-3 bg-black rounded overflow-auto">
            {JSON.stringify(error.raw, null, 2)}
          </pre>
        </details>
      )}

      <div className="flex items-center gap-2 mt-4">
        <button
          onClick={onRetry}
          className="inline-flex items-center gap-2 px-4 py-2 bg-red-600 rounded hover:bg-red-700 text-sm font-medium"
        >
          <RotateCcw className="h-4 w-4" />
          Try again
        </button>
        <button
          onClick={onReset}
          className="px-4 py-2 border border-red-500/40 rounded text-sm font-medium text-red-200 hover:bg-red-500/20"
        >
          Start over
        </button>
      </div>
    </div>
  );
}
