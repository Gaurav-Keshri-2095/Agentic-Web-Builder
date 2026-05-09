import { clsx, type ClassValue } from "clsx"
import { twMerge } from "tailwind-merge"

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}


export interface NormalizedError {
  title: string;
  message: string;
  details: string;
}

export function parseApiError(err: unknown): NormalizedError {
  const result: NormalizedError = {
    title: "Generation Failed",
    message: "The backend failed while generating the project.",
    details: ""
  };

  let rawContent = "";

  if (err instanceof Error) {
    rawContent = err.message;
  } else if (typeof err === "string") {
    rawContent = err;
  } else if (typeof err === "object" && err !== null) {
    rawContent = JSON.stringify(err);
  } else {
    rawContent = String(err);
  }

  let parsedJson: any = null;
  try {
    parsedJson = JSON.parse(rawContent);
  } catch {
    const match = rawContent.match(/^(.*?) - (\{.*})$/s);
    if (match) {
      try {
        const embedded = JSON.parse(match[2]);
        parsedJson = { message: match[1], details: embedded };
      } catch {}
    }
  }

  if (parsedJson && typeof parsedJson === "object") {
    if (parsedJson.error) {
       result.message = typeof parsedJson.error === "string" ? parsedJson.error : JSON.stringify(parsedJson.error);
    }
    if (parsedJson.message && !parsedJson.error) {
       result.message = parsedJson.message;
    }
    if (parsedJson.detail) {
       result.details = typeof parsedJson.detail === "string" ? parsedJson.detail : JSON.stringify(parsedJson.detail, null, 2);
    } else if (parsedJson.details) {
       result.details = typeof parsedJson.details === "string" ? parsedJson.details : JSON.stringify(parsedJson.details, null, 2);
    } else {
       result.details = JSON.stringify(parsedJson, null, 2);
    }
  } else {
    result.details = rawContent;
  }
  
  try {
     if (typeof result.details === "string" && (result.details.startsWith("{") || result.details.startsWith("["))) {
       result.details = JSON.stringify(JSON.parse(result.details), null, 2);
     }
  } catch {}

  if (typeof result.details === "string") {
    const jsonStartIdx = result.details.indexOf(" - {");
    if (jsonStartIdx !== -1) {
       try {
          const prefix = result.details.substring(0, jsonStartIdx);
          const jsonStr = result.details.substring(jsonStartIdx + 3);
          const parsed = JSON.parse(jsonStr);
          result.details = prefix + " -\n" + JSON.stringify(parsed, null, 2);
       } catch {}
    }
  }
  
  if (result.details.includes("json_validate_failed")) {
     result.message = "Failed to generate JSON. The model produced malformed output.";
  } else if (result.details.includes("rate_limit_exceeded")) {
     result.message = "Rate limit exceeded. Please try again later.";
  }

  return result;
}
