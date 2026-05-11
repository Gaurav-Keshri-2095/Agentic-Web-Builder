import React, { useState } from "react";
import { Code2 } from "lucide-react";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { LoginForm } from "./LoginForm";
import { SignupForm } from "./SignupForm";

interface AuthPageProps {
  onLogin: () => void;
}

export function AuthPage({ onLogin }: AuthPageProps) {
  const [activeTab, setActiveTab] = useState("signin");

  return (
    <div className="min-h-screen w-full flex items-center justify-center bg-background p-4 sm:p-6 dark">
      {/* Background gradients */}
      <div className="fixed inset-0 overflow-hidden pointer-events-none">
        <div className="absolute -top-[20%] -left-[10%] w-[50%] h-[50%] rounded-full bg-primary/10 blur-[120px]" />
        <div className="absolute top-[60%] -right-[10%] w-[50%] h-[50%] rounded-full bg-primary/10 blur-[120px]" />
      </div>

      <div className="w-full max-w-md z-10 relative">
        <div className="flex flex-col items-center mb-6 text-center">
          <div className="h-10 w-10 rounded-xl bg-gradient-primary flex items-center justify-center mb-3 shadow-lg shadow-primary/20">
            <Code2 className="h-5 w-5 text-primary-foreground" />
          </div>
          <h1 className="text-2xl sm:text-3xl font-bold tracking-tight">AI Web Builder</h1>
          <p className="text-sm text-muted-foreground mt-1.5">Generate web apps with AI</p>
        </div>

        <Card className="border-border/50 bg-card/50 backdrop-blur-xl shadow-2xl">
          <CardHeader className="pb-3 text-center">
            <CardTitle>Welcome back</CardTitle>
            <CardDescription className="text-xs">Sign in to your account</CardDescription>
          </CardHeader>
          <CardContent>
            <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
              <TabsList className="grid w-full grid-cols-2 mb-4 h-9">
                <TabsTrigger value="signin" className="text-xs">Sign In</TabsTrigger>
                <TabsTrigger value="signup" className="text-xs">Sign Up</TabsTrigger>
              </TabsList>
              <TabsContent value="signin" className="mt-0">
                <LoginForm onSuccess={onLogin} />
              </TabsContent>
              <TabsContent value="signup" className="mt-0">
                <SignupForm onSuccess={onLogin} switchToLogin={() => setActiveTab("signin")} />
              </TabsContent>
            </Tabs>
          </CardContent>
        </Card>
        
        <p className="text-center text-xs text-muted-foreground mt-6">
          By clicking continue, you agree to our Terms of Service and Privacy Policy.
        </p>
      </div>
    </div>
  );
}