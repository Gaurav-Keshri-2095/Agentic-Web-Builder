import { useState } from "react";
import { Plus } from "lucide-react";
import { FileExplorer } from "./FileExplorer";
import { CodeViewer } from "./CodeViewer";
import type { GeneratedFile } from "@/lib/types";
import { downloadZip } from "@/lib/downloadZip";

interface ResultViewProps {
  files: GeneratedFile[];
  activePath: string;
  prompt: string;
  onNew: () => void;
  onSelect: (path: string) => void;
}

export function ResultView({ files, activePath, prompt, onNew, onSelect }: ResultViewProps) {
  const activeFile = files.find((f) => f.path === activePath) ?? files[0] ?? null;
  const [menuOpen, setMenuOpen] = useState(false);
  const canDownload = files.length > 0;

  const handleDownload = async () => {
    if (!canDownload) return;
    setMenuOpen(false);
    await downloadZip(files);
  };

  return (
    <div className="w-full h-[calc(100vh-5rem)] flex flex-col">
      <div className="flex items-start justify-between gap-4 mb-4 px-1">
        <div className="min-w-0">
          <p className="text-[10px] uppercase tracking-wider text-muted-foreground mb-1">Prompt</p>
          <p className="text-sm text-foreground line-clamp-2 max-w-3xl">{prompt}</p>
        </div>
        <div className="relative shrink-0 flex items-center gap-2">
          <button
            onClick={onNew}
            className="inline-flex items-center gap-1.5 rounded-lg border px-3 py-2 text-sm font-medium hover:bg-muted transition-smooth"
          >
            <Plus className="h-4 w-4" />
            New
          </button>
          <button
            type="button"
            disabled={!canDownload}
            onClick={() => canDownload && setMenuOpen((open) => !open)}
            className="h-9 w-9 rounded-lg border text-lg leading-none flex items-center justify-center transition-smooth hover:bg-muted disabled:opacity-50 disabled:hover:bg-transparent"
            aria-label="Open file actions"
            aria-haspopup="menu"
            aria-expanded={menuOpen}
          >
            ...
          </button>
          {menuOpen && (
            <div
              role="menu"
              className="absolute right-0 top-11 z-10 w-40 rounded-lg border bg-popover text-popover-foreground shadow-elegant p-1"
            >
              <button
                type="button"
                onClick={handleDownload}
                className="w-full text-left px-3 py-2 text-sm rounded-md hover:bg-muted transition-smooth"
                role="menuitem"
              >
                Download ZIP
              </button>
            </div>
          )}
        </div>
      </div>

      <div className="flex-1 min-h-0 grid grid-cols-1 md:grid-cols-[260px_1fr] rounded-xl border bg-card shadow-elegant overflow-hidden">
        <aside className="border-b md:border-b-0 md:border-r bg-surface min-h-0 max-h-[40vh] md:max-h-none">
          <FileExplorer files={files} activePath={activePath} onSelect={onSelect} />
        </aside>
        <main className="min-h-0 min-w-0">
          <CodeViewer file={activeFile} />
        </main>
      </div>
    </div>
  );
}
