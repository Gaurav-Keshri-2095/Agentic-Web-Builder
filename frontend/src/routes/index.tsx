import { createFileRoute } from "@tanstack/react-router";
import { useCallback, useEffect, useRef, useState } from "react";
import { Code2 } from "lucide-react";
import { InputPanel } from "@/components/InputPanel";
import { LoadingState } from "@/components/LoadingState";
import { ErrorDisplay } from "@/components/ErrorDisplay";
import { ResultView } from "@/components/ResultView";
import { generateProject, ApiError } from "@/lib/api";
import type { GeneratedFile } from "@/lib/types";

export const Route = createFileRoute("/")({
  component: Index,
});

type Status = "idle" | "loading" | "success" | "error";
type Step = { id: string; label: string; duration: number };

type ErrorState = {
  message: string;
  details: string;
  raw?: unknown;
};

const STEPS: Step[] = [
  { id: "plan", label: "Planning architecture", duration: 800 },
  { id: "arch", label: "Designing file structure", duration: 700 },
  { id: "code", label: "Generating files", duration: 1200 },
  { id: "polish", label: "Finalizing project", duration: 500 },
];

function Index() {
  const [status, setStatus] = useState<Status>("idle");
  const [prompt, setPrompt] = useState("");
  const [files, setFiles] = useState<GeneratedFile[]>([]);
  const [selectedFile, setSelectedFile] = useState<string>("");
  const [loadingStep, setLoadingStep] = useState(0);
  const [error, setError] = useState<ErrorState | null>(null);
  const inFlight = useRef<AbortController | null>(null);
  const timers = useRef<number[]>([]);

  useEffect(() => {
    timers.current.forEach((id) => window.clearTimeout(id));
    timers.current = [];
    if (status !== "loading") return;

    let idx = 0;
    const schedule = (stepIdx: number) => {
      if (stepIdx >= STEPS.length) return;
      setLoadingStep(stepIdx);
      const id = window.setTimeout(() => {
        idx += 1;
        schedule(idx);
      }, STEPS[stepIdx].duration);
      timers.current.push(id);
    };

    schedule(idx);
    return () => {
      timers.current.forEach((id) => window.clearTimeout(id));
      timers.current = [];
    };
  }, [status]);

  const submit = useCallback(async (p: string) => {
    if (status === "loading") return; // prevent duplicates
    setPrompt(p);
    setStatus("loading");
    setError(null);
    setFiles([]);
    setSelectedFile("");
    setLoadingStep(0);

    inFlight.current?.abort();
    const ctrl = new AbortController();
    inFlight.current = ctrl;

    try {
      const data = await generateProject(p, ctrl.signal);
      if (ctrl.signal.aborted) return;
      setFiles(data.files);
      setSelectedFile(data.files[0]?.path ?? "");
      setStatus("success");
    } catch (err) {
      if (ctrl.signal.aborted) return;
      const fallback: ErrorState = {
        message: "Generation failed",
        details: "Unable to connect to server",
      };

      let parsed: ErrorState = fallback;

      if (err instanceof ApiError) {
        parsed = {
          message: "Generation failed",
          details: err.message || "Something went wrong",
        };
        try {
          const asJson = JSON.parse(err.message);
          const detail = (asJson as { detail?: unknown }).detail;
          if (typeof detail === "string") {
            parsed.details = detail.split("{")[0].trim() || "Invalid request";
            parsed.raw = detail;
          } else if (detail) {
            parsed.details = "Invalid request";
            parsed.raw = detail;
          }
        } catch {}
      } else if (err instanceof Error) {
        parsed = {
          message: "Generation failed",
          details: err.message || "Something went wrong",
        };
      }

      setError(parsed);
      setStatus("error");
    }
  }, [status]);

  const reset = useCallback(() => {
    inFlight.current?.abort();
    setStatus("idle");
    setFiles([]);
    setSelectedFile("");
    setError(null);
    setPrompt("");
  }, []);

  return (
    <div className="min-h-screen flex flex-col">
      <header className="border-b bg-background/80 backdrop-blur sticky top-0 z-10">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 h-14 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <div className="h-7 w-7 rounded-lg bg-gradient-primary flex items-center justify-center">
              <Code2 className="h-4 w-4 text-primary-foreground" />
            </div>
            <span className="font-semibold tracking-tight">AI WEB BUILDER</span>
          </div>
          <div className="text-xs text-muted-foreground hidden sm:flex items-center gap-2">
            <span className="h-1.5 w-1.5 rounded-full bg-primary animate-pulse-dot" />
            Agent pipeline ready
          </div>
        </div>
      </header>

      <main className="flex-1 w-full max-w-7xl mx-auto px-4 sm:px-6 py-10">
        {status === "idle" && <InputPanel onSubmit={submit} disabled={false} />}
        {status === "loading" && <LoadingState prompt={prompt} activeIdx={loadingStep} />}
        {status === "error" && error && (
          <ErrorDisplay error={error} onRetry={() => submit(prompt)} onReset={reset} />
        )}
        {status === "success" && files.length > 0 && (
          <ResultView
            files={files}
            activePath={selectedFile}
            prompt={prompt}
            onNew={reset}
            onSelect={setSelectedFile}
          />
        )}
      </main>
    </div>
  );
}
