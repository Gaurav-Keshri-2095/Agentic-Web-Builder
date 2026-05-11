import React, { useState } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { OAuthButtons } from "./OAuthButtons";

interface SignupFormProps {
  onSuccess: () => void;
  switchToLogin: () => void;
}

export function SignupForm({ onSuccess, switchToLogin }: SignupFormProps) {
  const [name, setName] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");
  const [error, setError] = useState("");
  const [success, setSuccess] = useState(false);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    setError("");
    setSuccess(false);

    if (!name || !email || !password || !confirmPassword) {
      setError("Please fill in all fields.");
      return;
    }

    if (password !== confirmPassword) {
      setError("Passwords do not match.");
      return;
    }

    setSuccess(true);
    setTimeout(() => {
      switchToLogin();
    }, 1500);
  };

  return (
    <div className="space-y-4">
      <form onSubmit={handleSubmit} className="space-y-3">
        {error && (
          <div className="p-2.5 text-xs text-destructive bg-destructive/10 rounded-md">
            {error}
          </div>
        )}
        {success && (
          <div className="p-2.5 text-xs text-green-500 bg-green-500/10 rounded-md">
            Account created! Redirecting to login...
          </div>
        )}
        <div className="space-y-1.5">
          <Label htmlFor="name">Name</Label>
          <Input
            id="name"
            placeholder="John Doe"
            value={name}
            onChange={(e) => setName(e.target.value)}
            className="h-9 text-sm"
          />
        </div>
        <div className="space-y-1.5">
          <Label htmlFor="email-signup">Email</Label>
          <Input
            id="email-signup"
            type="email"
            placeholder="m@example.com"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            className="h-9 text-sm"
          />
        </div>
        <div className="space-y-1.5">
          <Label htmlFor="password-signup">Password</Label>
          <Input
            id="password-signup"
            type="password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            className="h-9 text-sm"
          />
        </div>
        <div className="space-y-1.5">
          <Label htmlFor="confirm-password">Confirm Password</Label>
          <Input
            id="confirm-password"
            type="password"
            value={confirmPassword}
            onChange={(e) => setConfirmPassword(e.target.value)}
            className="h-9 text-sm"
          />
        </div>
        <Button className="w-full h-9" type="submit" disabled={success}>
          Sign Up
        </Button>
      </form>
      <OAuthButtons onSuccess={onSuccess} />
    </div>
  );
}