import type { GenerateResponse } from "./types";

export class ApiError extends Error {
  status?: number;

  constructor(message: string, status?: number) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

export async function generateProject(
  prompt: string,
  signal: AbortSignal,
): Promise<GenerateResponse> {
  const apiBase = import.meta.env.VITE_API_BASE ?? "http://localhost:8000";
  const url = `${apiBase.replace(/\/+$/, "")}/generate`;
  let response: Response;

  try {
    response = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ prompt }),
      signal,
    });
  } catch (err) {
    if (signal.aborted) throw err;
    throw new ApiError("Failed to reach API");
  }

  if (!response.ok) {
    const text = await response.text().catch(() => "");
    const message = text || `Request failed with status ${response.status}`;
    throw new ApiError(message, response.status);
  }

  const data = (await response.json()) as GenerateResponse;
  if (!data?.files || !Array.isArray(data.files)) {
    throw new ApiError("Invalid response from API");
  }
  if (data.files.length === 0) {
    throw new ApiError("No files returned from API");
  }
  return data;
}
