import { useMemo, useState } from "react";
import { ChevronRight, File as FileIcon, Folder, FolderOpen } from "lucide-react";
import type { GeneratedFile } from "@/lib/types";

interface FileExplorerProps {
  files: GeneratedFile[];
  activePath: string;
  onSelect: (path: string) => void;
}

interface TreeNode {
  name: string;
  path: string;
  isDir: boolean;
  children: Map<string, TreeNode>;
}

function buildTree(files: GeneratedFile[]): TreeNode {
  const root: TreeNode = { name: "", path: "", isDir: true, children: new Map() };
  for (const f of files) {
    const parts = f.path.split("/").filter(Boolean);
    let node = root;
    parts.forEach((part, i) => {
      const isFile = i === parts.length - 1;
      if (!node.children.has(part)) {
        node.children.set(part, {
          name: part,
          path: parts.slice(0, i + 1).join("/"),
          isDir: !isFile,
          children: new Map(),
        });
      }
      node = node.children.get(part)!;
    });
  }
  return root;
}

function TreeItem({
  node,
  depth,
  activePath,
  onSelect,
}: {
  node: TreeNode;
  depth: number;
  activePath: string;
  onSelect: (path: string) => void;
}) {
  const [open, setOpen] = useState(true);
  const children = Array.from(node.children.values()).sort((a, b) => {
    if (a.isDir !== b.isDir) return a.isDir ? -1 : 1;
    return a.name.localeCompare(b.name);
  });

  if (!node.isDir) {
    const active = node.path === activePath;
    return (
      <button
        onClick={() => onSelect(node.path)}
        style={{ paddingLeft: `${depth * 12 + 8}px` }}
        className={`w-full flex items-center gap-2 py-1.5 pr-2 text-sm text-left transition-smooth rounded-md ${
          active
            ? "bg-primary/15 text-foreground"
            : "text-muted-foreground hover:text-foreground hover:bg-muted/50"
        }`}
      >
        <FileIcon className="h-3.5 w-3.5 shrink-0" />
        <span className="truncate">{node.name}</span>
      </button>
    );
  }

  return (
    <div>
      {node.name && (
        <button
          onClick={() => setOpen((o) => !o)}
          style={{ paddingLeft: `${depth * 12 + 4}px` }}
          className="w-full flex items-center gap-1 py-1.5 pr-2 text-sm text-foreground hover:bg-muted/50 rounded-md transition-smooth"
        >
          <ChevronRight
            className={`h-3 w-3 shrink-0 transition-transform ${open ? "rotate-90" : ""}`}
          />
          {open ? (
            <FolderOpen className="h-3.5 w-3.5 shrink-0 text-primary" />
          ) : (
            <Folder className="h-3.5 w-3.5 shrink-0 text-primary" />
          )}
          <span className="truncate font-medium">{node.name}</span>
        </button>
      )}
      {open && (
        <div>
          {children.map((c) => (
            <TreeItem
              key={c.path}
              node={c}
              depth={node.name ? depth + 1 : depth}
              activePath={activePath}
              onSelect={onSelect}
            />
          ))}
        </div>
      )}
    </div>
  );
}

export function FileExplorer({ files, activePath, onSelect }: FileExplorerProps) {
  const tree = useMemo(() => buildTree(files), [files]);
  return (
    <div className="h-full overflow-y-auto scrollbar-thin py-2 px-1">
      <div className="px-3 py-2 text-[10px] uppercase tracking-wider text-muted-foreground font-semibold">
        Files ({files.length})
      </div>
      <TreeItem node={tree} depth={0} activePath={activePath} onSelect={onSelect} />
    </div>
  );
}
