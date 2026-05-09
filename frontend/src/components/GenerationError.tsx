import { AlertTriangle, RotateCcw, ChevronDown, ChevronRight } from "lucide-react";
import { useState } from "react";

export interface NormalizedError {
  title: string;
  message: string;
  details: string;
}

interface GenerationErrorProps {
  error: NormalizedError;
  onRetry: () => void;
  onReset: () => void;
}

export function GenerationError({ error, onRetry, onReset }: GenerationErrorProps) {
  const [isExpanded, setIsExpanded] = useState(false);

  return (
    <div className="max-w-3xl mx-auto mt-10 p-6 border border-red-500/50 rounded-xl bg-zinc-950 shadow-xl overflow-hidden flex flex-col items-center sm:items-start text-left">
      <div className="flex items-center gap-3 mb-2 w-full">
        <AlertTriangle className="h-6 w-6 text-red-500 shrink-0" />
        <h2 className="text-xl font-bold text-red-100 tracking-tight">{error.title}</h2>
      </div>

      <div className="w-full pl-0 sm:pl-9">
        <p className="text-base text-zinc-300 mb-6 font-medium leading-relaxed">
          {error.message}
        </p>

        {error.details && error.details.trim() !== "Error" && error.details.trim() !== "" && (
          <div className="mb-6 w-full overflow-hidden border border-zinc-800 rounded-md bg-zinc-900/50">
            <button
              onClick={() => setIsExpanded(!isExpanded)}
              className="flex items-center gap-2 w-full px-4 py-3 text-sm font-medium text-zinc-400 hover:text-zinc-200 hover:bg-zinc-800/50 transition-colors"
            >
              {isExpanded ? (
                <ChevronDown className="h-4 w-4 shrink-0" />
              ) : (
                <ChevronRight className="h-4 w-4 shrink-0" />
              )}
              {isExpanded ? "Hide Technical Details" : "Show Technical Details"}
            </button>
            
            {isExpanded && (
              <div className="p-4 pt-0 border-t border-zinc-800 max-h-[400px] overflow-auto custom-scrollbar">
                <pre className="text-[13px] text-zinc-400 font-mono whitespace-pre-wrap break-words leading-relaxed">
                  {error.details}
                </pre>
              </div>
            )}
          </div>
        )}

        <div className="flex items-center gap-3">
          <button
            onClick={onRetry}
            className="inline-flex items-center gap-2 px-5 py-2.5 bg-red-500/10 border border-red-500/30 text-red-400 rounded-lg hover:bg-red-500/20 hover:text-red-300 hover:border-red-500/50 text-sm font-medium transition-all"
          >
            <RotateCcw className="h-4 w-4" />
            Retry
          </button>
          <button
            onClick={onReset}
            className="px-5 py-2.5 border border-zinc-700 rounded-lg text-sm font-medium text-zinc-300 hover:bg-zinc-800 hover:text-zinc-100 transition-all"
          >
            Start Over
          </button>
        </div>
      </div>
      
      <style dangerouslySetInnerHTML={{__html:`
        .custom-scrollbar::-webkit-scrollbar {
          width: 8px;
          height: 8px;
        }
        .custom-scrollbar::-webkit-scrollbar-track {
          background: rgba(0, 0, 0, 0.2); 
          border-radius: 4px;
        }
        .custom-scrollbar::-webkit-scrollbar-thumb {
          background: rgba(255, 255, 255, 0.1); 
          border-radius: 4px;
        }
        .custom-scrollbar::-webkit-scrollbar-thumb:hover {
          background: rgba(255, 255, 255, 0.2); 
        }
      `}} />
    </div>
  );
}
