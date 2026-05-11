import { useState } from "react";
import { Prism as SyntaxHighlighter } from "react-syntax-highlighter";
import { oneDark } from "react-syntax-highlighter/dist/esm/styles/prism";
import { Check, Copy } from "lucide-react";
import type { GeneratedFile } from "@/lib/types";

interface CodeViewerProps {
  file: GeneratedFile | null;
}

export function CodeViewer({ file }: CodeViewerProps) {
  const [copied, setCopied] = useState(false);

  if (!file) {
    return (
      <div className="h-full flex items-center justify-center text-muted-foreground text-sm">
        Select a file to view
      </div>
    );
  }

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(file.content);
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch {}
  };

  return (
    <div className="flex flex-col h-full bg-zinc-950">
      <div className="flex items-center justify-between border-b px-4 py-2.5 bg-surface sticky top-0 z-10">
        <div className="flex items-center gap-2 min-w-0">
          <span className="text-xs text-muted-foreground truncate font-mono">{file.path}</span>
          {file.language && (
            <span className="text-[10px] uppercase tracking-wider text-muted-foreground bg-muted rounded px-1.5 py-0.5">
              {file.language}
            </span>
          )}
        </div>
        <button
          onClick={handleCopy}
          className="inline-flex items-center gap-1.5 text-xs text-muted-foreground hover:text-foreground transition-smooth rounded-md px-2 py-1 hover:bg-muted"
        >
          {copied ? (
            <>
              <Check className="h-3.5 w-3.5 text-primary" />
              Copied
            </>
          ) : (
            <>
              <Copy className="h-3.5 w-3.5" />
              Copy
            </>
          )}
        </button>
      </div>
      <div className="flex-1 w-full bg-zinc-950">
        <SyntaxHighlighter
          language={file.language ?? "text"}
          style={oneDark}
          showLineNumbers
          wrapLongLines={false}
          customStyle={{
            margin: 0,
            padding: "1rem",
            background: "transparent",
            fontSize: "13px",
            lineHeight: "1.6",
            minHeight: "100%",
          }}
          lineNumberStyle={{ color: "oklch(0.45 0.015 260)", minWidth: "2.5em" }}
        >
          {file.content}
        </SyntaxHighlighter>
      </div>
    </div>
  );
}
