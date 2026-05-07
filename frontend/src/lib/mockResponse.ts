export interface GeneratedFile {
  path: string;
  content: string;
  language?: string;
}

export interface GenerateResponse {
  summary?: string;
  files: GeneratedFile[];
}

const SAMPLE_FILES: GeneratedFile[] = [
  {
    path: "README.md",
    language: "markdown",
    content: "# Generated Project\n\nThis is a mocked response to verify the UI and routing.",
  },
  {
    path: "src/main.ts",
    language: "typescript",
    content: "export function hello() {\n  console.log('Hello from the mock project');\n}\n",
  },
  {
    path: "src/components/App.tsx",
    language: "tsx",
    content: "export function App() {\n  return <div>Hello from App</div>;\n}\n",
  },
];

export function getMockResponse(prompt: string): GenerateResponse {
  return {
    summary: `Mocked output for: ${prompt}`,
    files: SAMPLE_FILES,
  };
}
