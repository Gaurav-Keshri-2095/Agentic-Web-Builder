import { createFileRoute, useNavigate } from "@tanstack/react-router";
import { useCallback, useEffect, useRef, useState } from "react";
import { Code2, LogOut } from "lucide-react";
import { InputPanel } from "@/components/InputPanel";
import { LoadingState } from "@/components/LoadingState";
import { GenerationError } from "@/components/GenerationError";
import { ResultView } from "@/components/ResultView";
import { generateProject, ApiError } from "@/lib/api";
import { parseApiError, type NormalizedError } from "@/lib/utils";
import type { GeneratedFile } from "@/lib/types";
import { useAuth } from "@/hooks/useAuth";
import { AuthPage } from "@/components/auth/AuthPage";
import { Button } from "@/components/ui/button";

export const Route = createFileRoute("/")({
  component: Index,
});

type Status = "idle" | "loading" | "success" | "error";
type Step = { id: string; label: string; duration: number };

const STEPS: Step[] = [
  { id: "plan", label: "Planning architecture", duration: 800 },
  { id: "arch", label: "Designing file structure", duration: 700 },
  { id: "code", label: "Generating files", duration: 1200 },
  { id: "polish", label: "Finalizing project", duration: 500 },
];

function Index() {
  const { isAuthenticated, isRecoveryMode, isLoading, logout } = useAuth();
  const navigate = useNavigate();
  const [status, setStatus] = useState<Status>("idle");
  const [prompt, setPrompt] = useState("");
  const [files, setFiles] = useState<GeneratedFile[]>([]);
  const [selectedFile, setSelectedFile] = useState<string>("");
  const [loadingStep, setLoadingStep] = useState(0);
  const [error, setError] = useState<NormalizedError | null>(null);
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

  useEffect(() => {
    if (isRecoveryMode) {
      navigate({ to: "/reset-password", replace: true });
    }
  }, [isRecoveryMode, navigate]);

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
      const generatedFiles = data.files ?? [];
      setFiles(generatedFiles);
      setSelectedFile(generatedFiles[0]?.path ?? "");
      setStatus("success");
    } catch (err) {
      if (ctrl.signal.aborted) return;
      setError(parseApiError(err));
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

  if (isLoading) return null;

  if (isRecoveryMode) return null;

  if (!isAuthenticated) return <AuthPage onLogin={() => {}} />;

  return (
    <div className="min-h-screen flex flex-col bg-background">
      <header className="bg-background/80 backdrop-blur top-0 border-b border-border/40 z-10 flex-none sticky">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 h-14 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <div className="h-7 w-7 rounded-lg bg-gradient-primary flex items-center justify-center shadow-[0_0_15px_rgba(var(--primary),0.3)]">
              <Code2 className="h-4 w-4 text-primary-foreground" />
            </div>
            <span className="font-semibold tracking-tight text-foreground">AI WEB BUILDER</span>
          </div>
          <div className="flex items-center gap-4">
            <div className="text-xs text-muted-foreground hidden sm:flex items-center gap-2">
              <span className="h-1.5 w-1.5 rounded-full bg-primary animate-pulse-dot" />
              Agent pipeline ready
            </div>
            <Button variant="ghost" size="sm" onClick={logout} className="h-8 group hover:bg-destructive/10 hover:text-destructive transition-colors">
              <LogOut className="h-3.5 w-3.5 mr-2 group-hover:scale-110 transition-transform" />
              Sign Out
            </Button>
          </div>
        </div>
      </header>

      <main className="flex-1 w-full max-w-7xl mx-auto px-4 sm:px-6 py-6 md:py-10 flex flex-col">
        {status === "idle" && <InputPanel onSubmit={submit} disabled={false} />}
        {status === "loading" && <LoadingState prompt={prompt} />}
        {status === "error" && error && (
          <GenerationError error={error} onRetry={() => submit(prompt)} onReset={reset} />
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
