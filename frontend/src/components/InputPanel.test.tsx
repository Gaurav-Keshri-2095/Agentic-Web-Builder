import { render, screen, fireEvent } from "@testing-library/react";
import { describe, it, expect, vi } from "vitest";
import { InputPanel } from "@/components/InputPanel";

describe("InputPanel", () => {
  it("submits the text inside input area successfully when button pressed", () => {
    const submitMock = vi.fn();
    render(<InputPanel onSubmit={submitMock} disabled={false} />);

    // Grab elements
    const textarea = screen.getByPlaceholderText(/Build me a.../i);
    const button = screen.getByRole("button", { name: /^Generate$/i });

    // Simulate interactions
    fireEvent.change(textarea, { target: { value: "Build a chat app" } });
    fireEvent.click(button);

    expect(submitMock).toHaveBeenCalledWith("Build a chat app");
    expect(submitMock).toHaveBeenCalledTimes(1);
  });

  it("submits the text via Enter without Shift", () => {
    const submitMock = vi.fn();
    render(<InputPanel onSubmit={submitMock} disabled={false} />);

    const textarea = screen.getByPlaceholderText(/Build me a.../i);
    fireEvent.change(textarea, { target: { value: "Build a chat app" } });
    
    // Simulate Enter without shift
    fireEvent.keyDown(textarea, { key: "Enter", shiftKey: false });

    // It should submit
    expect(submitMock).toHaveBeenCalledWith("Build a chat app");
  });

  it("does not submit data on Shift+Enter (intended for new line)", () => {
    const submitMock = vi.fn();
    render(<InputPanel onSubmit={submitMock} disabled={false} />);

    const textarea = screen.getByPlaceholderText(/Build me a.../i);
    fireEvent.change(textarea, { target: { value: "Build a chat app" } });
    
    // Simulate Shift+Enter
    fireEvent.keyDown(textarea, { key: "Enter", shiftKey: true });

    expect(submitMock).not.toHaveBeenCalled();
  });
});
