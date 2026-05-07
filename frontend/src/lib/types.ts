export interface GeneratedFile {
  path: string;
  content: string;
  language?: string;
}

export interface GenerateResponse {
  success: boolean;
  files?: GeneratedFile[];
  error?: string;
  details?: string;
}
