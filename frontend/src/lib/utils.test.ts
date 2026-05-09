import { describe, it, expect } from "vitest";
import { parseApiError } from "@/lib/utils";

describe("parseApiError", () => {
  it("should parse an Axios 400 validation error appropriately", () => {
    // Simulate what standard HTTP validation 400 errors look like
    const mockAxiosError = {
      response: {
        data: {
          success: false,
          error: "Code generation failed",
          details: "missing prompt param"
        }
      }
    };

    const parsed = parseApiError(mockAxiosError);
    expect(parsed.title).toBe("Generation Failed");
    expect(parsed.details).toContain("missing prompt param");
  });

  it("should handle raw generic strings", () => {
    const parsed = parseApiError("Something blew up");
    expect(parsed.title).toBe("Generation Failed");
    expect(parsed.details).toBe("Something blew up");
  });

  it("should handle native Error instances", () => {
    const error = new Error("Network timeout");
    const parsed = parseApiError(error);
    expect(parsed.title).toBe("Generation Failed");
    expect(parsed.details).toBe("Network timeout");
  });
});
