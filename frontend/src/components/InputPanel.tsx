import { useState, FormEvent, KeyboardEvent } from "react";
import { Sparkles, ArrowUp } from "lucide-react";

interface InputPanelProps {
  onSubmit: (prompt: string) => void;
  disabled: boolean;
}

const SUGGESTIONS = [
  "A REST API for a todo app with auth",
  "A Python CLI that scrapes Hacker News",
  "A React dashboard with charts and filters",
  "A Discord bot that summarizes links",
];

export function InputPanel({ onSubmit, disabled }: InputPanelProps) {
  const [value, setValue] = useState("");

  const handleSubmit = (e?: FormEvent) => {
    e?.preventDefault();
    const trimmed = value.trim();
    if (!trimmed || disabled) return;
    onSubmit(trimmed);
  };

  const handleKey = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  return (
    <div className="relative w-full max-w-3xl mx-auto mt-8">
      <div className="absolute -inset-x-32 -inset-y-32 bg-gradient-glow pointer-events-none -z-10 opacity-70" />

      <div className="text-center mb-10">
        <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-surface-elevated border text-xs text-muted-foreground mb-6">
          <Sparkles className="h-3 w-3 text-primary" />
          AI Code Generation
        </div>
        <h1 className="text-4xl md:text-5xl font-semibold tracking-tight mb-3">
          Describe what you want to{" "}
          <span className="bg-gradient-primary bg-clip-text text-transparent">build</span>
        </h1>
        <p className="text-muted-foreground text-base max-w-xl mx-auto">
          Our agent pipeline plans, architects, and writes a full codebase from a single prompt.
        </p>
      </div>

      <form onSubmit={handleSubmit}>
        <div className="relative rounded-2xl border bg-card shadow-elegant transition-smooth focus-within:border-primary/50 focus-within:shadow-glow">
          <textarea
            value={value}
            onChange={(e) => setValue(e.target.value)}
            onKeyDown={handleKey}
            disabled={disabled}
            rows={5}
            placeholder="Build me a..."
            className="w-full resize-none bg-transparent px-5 py-4 text-base placeholder:text-muted-foreground focus:outline-none disabled:opacity-50 scrollbar-thin"
          />
          <div className="flex items-center justify-between px-3 pb-3 pt-1">
            <span className="text-xs text-muted-foreground px-2">
              <kbd className="px-1.5 py-0.5 rounded bg-muted text-[10px]">↵</kbd> to send,{" "}
              <kbd className="px-1.5 py-0.5 rounded bg-muted text-[10px]">⇧</kbd> +{" "}
              <kbd className="px-1.5 py-0.5 rounded bg-muted text-[10px]">↵</kbd> for new line
            </span>
            <button
              type="submit"
              disabled={disabled || !value.trim()}
              className="inline-flex items-center gap-2 rounded-lg bg-gradient-primary text-primary-foreground px-4 py-2 text-sm font-medium transition-smooth hover:opacity-90 disabled:opacity-40 disabled:cursor-not-allowed"
            >
              Generate
              <ArrowUp className="h-4 w-4" />
            </button>
          </div>
        </div>
      </form>

      <div className="mt-6 flex flex-wrap gap-2 justify-center">
        {SUGGESTIONS.map((s) => (
          <button
            key={s}
            type="button"
            disabled={disabled}
            onClick={() => setValue(s)}
            className="text-xs text-muted-foreground hover:text-foreground border border-border hover:border-primary/40 rounded-full px-3 py-1.5 transition-smooth disabled:opacity-40"
          >
            {s}
          </button>
        ))}
      </div>
    </div>
  );
}
