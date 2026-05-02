import { useEffect, useState } from "react";
import { Check, Loader2 } from "lucide-react";

const STEPS = [
  { id: "plan", label: "Planning architecture", duration: 800 },
  { id: "arch", label: "Designing file structure", duration: 700 },
  { id: "code", label: "Generating files", duration: 1200 },
  { id: "polish", label: "Finalizing project", duration: 500 },
];

interface LoadingStateProps {
  prompt: string;
}

export function LoadingState({ prompt }: LoadingStateProps) {
  const [activeIdx, setActiveIdx] = useState(0);

  useEffect(() => {
    let cancelled = false;
    let i = 0;
    const tick = () => {
      if (cancelled || i >= STEPS.length) return;
      setActiveIdx(i);
      const t = setTimeout(() => {
        i++;
        tick();
      }, STEPS[i].duration);
      return () => clearTimeout(t);
    };
    tick();
    return () => {
      cancelled = true;
    };
  }, []);

  return (
    <div className="w-full max-w-2xl mx-auto">
      <div className="rounded-2xl border bg-card shadow-elegant overflow-hidden">
        <div className="relative h-1 bg-muted overflow-hidden">
          <div className="absolute inset-0 animate-shimmer" />
        </div>

        <div className="p-6">
          <div className="flex items-start gap-3 mb-6 pb-6 border-b">
            <div className="mt-1 flex gap-1">
              <span className="h-1.5 w-1.5 rounded-full bg-primary animate-pulse-dot" style={{ animationDelay: "0ms" }} />
              <span className="h-1.5 w-1.5 rounded-full bg-primary animate-pulse-dot" style={{ animationDelay: "200ms" }} />
              <span className="h-1.5 w-1.5 rounded-full bg-primary animate-pulse-dot" style={{ animationDelay: "400ms" }} />
            </div>
            <div className="flex-1 min-w-0">
              <p className="text-xs uppercase tracking-wider text-muted-foreground mb-1">Generating</p>
              <p className="text-sm text-foreground line-clamp-2">{prompt}</p>
            </div>
          </div>

          <ul className="space-y-3">
            {STEPS.map((step, i) => {
              const done = i < activeIdx;
              const active = i === activeIdx;
              return (
                <li key={step.id} className="flex items-center gap-3 transition-smooth">
                  <div
                    className={`h-6 w-6 rounded-full flex items-center justify-center transition-smooth ${
                      done
                        ? "bg-primary text-primary-foreground"
                        : active
                        ? "bg-primary/20 text-primary"
                        : "bg-muted text-muted-foreground"
                    }`}
                  >
                    {done ? (
                      <Check className="h-3.5 w-3.5" />
                    ) : active ? (
                      <Loader2 className="h-3.5 w-3.5 animate-spin" />
                    ) : (
                      <span className="h-1.5 w-1.5 rounded-full bg-current opacity-50" />
                    )}
                  </div>
                  <span
                    className={`text-sm transition-smooth ${
                      done ? "text-muted-foreground line-through" : active ? "text-foreground" : "text-muted-foreground"
                    }`}
                  >
                    {step.label}
                  </span>
                </li>
              );
            })}
          </ul>
        </div>
      </div>
    </div>
  );
}
