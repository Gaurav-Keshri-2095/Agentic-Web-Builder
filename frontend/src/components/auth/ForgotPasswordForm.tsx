import React, { useState, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { ArrowLeft, MailCheck } from "lucide-react";
import { supabase } from "../../lib/supabase";

interface ForgotPasswordFormProps {
  onBack: () => void;
}

export function ForgotPasswordForm({ onBack }: ForgotPasswordFormProps) {
  const [email, setEmail] = useState("");
  const [error, setError] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [isSuccess, setIsSuccess] = useState(false);
  const [countdown, setCountdown] = useState(0);

  useEffect(() => {
    let timer: number;
    if (countdown > 0) {
      timer = window.setInterval(() => {
        setCountdown((prev) => prev - 1);
      }, 1000);
    }
    return () => {
      if (timer) clearInterval(timer);
    };
  }, [countdown]);

  const handleSendReset = async (e?: React.FormEvent) => {
    if (e) e.preventDefault();
    setError("");

    if (!email) {
      setError("Please enter your email address.");
      return;
    }

    setIsLoading(true);
    const { error: resetError } = await supabase.auth.resetPasswordForEmail(email, {
      redirectTo: `${window.location.origin}/reset-password`,
    });
    setIsLoading(false);

    if (resetError) {
      setError(resetError.message);
      return;
    }

    setIsSuccess(true);
    setCountdown(60);
  };

  if (isSuccess) {
    return (
      <div className="space-y-6 text-center animate-in fade-in slide-in-from-bottom-2 duration-300">
        <div className="flex justify-center">
          <div className="h-12 w-12 rounded-full bg-primary/10 flex items-center justify-center">
            <MailCheck className="h-6 w-6 text-primary" />
          </div>
        </div>
        <div className="space-y-2">
          <h3 className="text-lg font-medium">Password reset link sent</h3>
          <p className="text-sm text-muted-foreground text-balance">
            Check your email for the password reset link.
          </p>
        </div>
        <div className="pt-2">
          <Button
            variant="outline"
            className="w-full text-xs font-normal"
            onClick={() => handleSendReset()}
            disabled={countdown > 0 || isLoading}
          >
            {isLoading ? "Resending..." : countdown > 0 ? `Resend in ${countdown}s` : "Didn't receive the email? Resend"}
          </Button>
          <Button
            variant="ghost"
            className="w-full mt-2 text-xs font-normal"
            onClick={onBack}
          >
            <ArrowLeft className="mr-2 h-3 w-3" />
            Back to Sign In
          </Button>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-4 animate-in fade-in slide-in-from-bottom-2 duration-300">
      <div className="mb-2">
        <h3 className="text-lg font-medium">Reset your password</h3>
        <p className="text-sm text-muted-foreground">
          Enter your registered email address and we'll send you a password reset link.
        </p>
      </div>

      <form onSubmit={handleSendReset} className="space-y-3">
        {error && (
          <div className="p-2.5 text-xs text-destructive bg-destructive/10 rounded-md">
            {error}
          </div>
        )}
        <div className="space-y-1.5">
          <Label htmlFor="reset-email">Email</Label>
          <Input
            id="reset-email"
            type="email"
            placeholder="m@example.com"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            className="h-9"
          />
        </div>
        <Button className="w-full h-9" type="submit" disabled={isLoading}>
          {isLoading ? "Sending Link..." : "Send Reset Link"}
        </Button>
      </form>
      
      <Button
        variant="ghost"
        className="w-full text-xs font-normal"
        onClick={onBack}
        disabled={isLoading}
      >
        <ArrowLeft className="mr-2 h-3 w-3" />
        Back to Sign In
      </Button>
    </div>
  );
}