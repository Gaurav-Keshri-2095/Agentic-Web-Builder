import React, { useState, useEffect } from 'react'
import { createFileRoute, useNavigate } from '@tanstack/react-router'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Code2, Eye, EyeOff, KeyRound } from 'lucide-react'
import { supabase } from '../lib/supabase'

export const Route = createFileRoute('/reset-password')({
  component: ResetPasswordPage,
})

function ResetPasswordPage() {
  const [password, setPassword] = useState('')
  const [confirmPassword, setConfirmPassword] = useState('')
  const [showPassword, setShowPassword] = useState(false)
  const [showConfirmPassword, setShowConfirmPassword] = useState(false)
  const [error, setError] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [isSuccess, setIsSuccess] = useState(false)
  const [isInitializing, setIsInitializing] = useState(true)
  const [hasRecoverySession, setHasRecoverySession] = useState(false)
  const navigate = useNavigate()

  useEffect(() => {
    let isMounted = true

    const checkRecoverySession = async () => {
      try {
        await new Promise(resolve => setTimeout(resolve, 100))

        const { data: { session } } = await supabase.auth.getSession()
        const hasRecoveryHash = window.location.hash.includes('type=recovery')

        if (isMounted) {
          if (session) {
            setHasRecoverySession(true)
            setError('')
          } else {
            if (hasRecoveryHash) {
              setError('Recovery token is invalid or has expired. Please request a new password reset.')
            } else {
              setError('Please use the password reset link from your email.')
            }
            setHasRecoverySession(false)
          }
          setIsInitializing(false)
        }
      } catch (err) {
        if (isMounted) {
          setError('An error occurred while verifying your reset link.')
          setIsInitializing(false)
        }
        console.error('Recovery session check error:', err)
      }
    }

    checkRecoverySession()

    return () => {
      isMounted = false
    }
  }, [])

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setError('')

    if (!password || !confirmPassword) {
      setError('Please fill in all fields.')
      return
    }

    if (password.length < 6) {
      setError('Password must be at least 6 characters long.')
      return
    }

    if (password !== confirmPassword) {
      setError('Passwords do not match.')
      return
    }

    setIsLoading(true)
    try {
      const { error: updateError } = await supabase.auth.updateUser({
        password: password,
      })
      setIsLoading(false)

      if (updateError) {
        setError(updateError.message)
        return
      }

      await supabase.auth.signOut()
      
      setIsSuccess(true)
      setTimeout(() => {
        navigate({ to: '/' })
      }, 2500)
    } catch (err) {
      setIsLoading(false)
      setError('An unexpected error occurred. Please try again.')
      console.error('Password update error:', err)
    }
  }

  if (isInitializing) {
    return (
      <div className="min-h-screen w-full flex items-center justify-center bg-background p-4 sm:p-6 dark">
        <div className="fixed inset-0 overflow-hidden pointer-events-none">
          <div className="absolute -top-[20%] -left-[10%] w-[50%] h-[50%] rounded-full bg-primary/10 blur-[120px]" />
          <div className="absolute top-[60%] -right-[10%] w-[50%] h-[50%] rounded-full bg-primary/10 blur-[120px]" />
        </div>
        <div className="w-full max-w-md z-10 relative">
          <Card className="border-border/50 bg-card/50 backdrop-blur-xl shadow-2xl">
            <CardContent className="pt-6 text-center">
              <p className="text-sm text-muted-foreground">Verifying reset link...</p>
            </CardContent>
          </Card>
        </div>
      </div>
    )
  }

  if (!hasRecoverySession) {
    return (
      <div className="min-h-screen w-full flex items-center justify-center bg-background p-4 sm:p-6 dark">
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
          </div>
          <Card className="border-border/50 bg-card/50 backdrop-blur-xl shadow-2xl">
            <CardContent className="pt-6">
              <div className="space-y-4 text-center">
                <div className="p-2.5 text-xs text-destructive bg-destructive/10 rounded-md">
                  {error}
                </div>
                <Button 
                  onClick={() => navigate({ to: '/' })}
                  className="w-full"
                >
                  Back to Sign In
                </Button>
              </div>
            </CardContent>
          </Card>
        </div>
      </div>
    )
  }

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
        </div>

        <Card className="border-border/50 bg-card/50 backdrop-blur-xl shadow-2xl">
          <CardHeader className="pb-3 text-center">
            <CardTitle>Set New Password</CardTitle>
            <CardDescription className="text-xs">
              Please enter your new password below
            </CardDescription>
          </CardHeader>
          <CardContent>
            {isSuccess ? (
              <div className="space-y-4 text-center animate-in fade-in slide-in-from-bottom-2 duration-300 py-4">
                <div className="flex justify-center">
                  <div className="h-12 w-12 rounded-full bg-primary/10 flex items-center justify-center">
                    <KeyRound className="h-6 w-6 text-primary" />
                  </div>
                </div>
                <div className="space-y-1">
                  <h3 className="text-lg font-medium text-green-500">Password updated successfully</h3>
                  <p className="text-sm text-muted-foreground">
                    Please sign in again using your new password. Redirecting you...
                  </p>
                </div>
              </div>
            ) : (
              <form onSubmit={handleSubmit} className="space-y-4 animate-in fade-in slide-in-from-bottom-2 duration-300">
                {error && (
                  <div className="p-2.5 text-xs text-destructive bg-destructive/10 rounded-md">
                    {error}
                  </div>
                )}
                <div className="space-y-1.5">
                  <Label htmlFor="new-password">New Password</Label>
                  <div className="relative">
                    <Input
                      id="new-password"
                      type={showPassword ? 'text' : 'password'}
                      value={password}
                      onChange={(e) => setPassword(e.target.value)}
                      className="h-9 text-sm pr-10"
                    />
                    <Button
                      type="button"
                      variant="ghost"
                      size="icon"
                      onClick={() => setShowPassword((prev) => !prev)}
                      className="absolute right-1 top-1/2 h-7 w-7 -translate-y-1/2 text-muted-foreground hover:text-foreground"
                      aria-label={showPassword ? 'Hide password' : 'Show password'}
                    >
                      {showPassword ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
                    </Button>
                  </div>
                </div>
                <div className="space-y-1.5">
                  <Label htmlFor="confirm-password">Confirm Password</Label>
                  <div className="relative">
                    <Input
                      id="confirm-password"
                      type={showConfirmPassword ? 'text' : 'password'}
                      value={confirmPassword}
                      onChange={(e) => setConfirmPassword(e.target.value)}
                      className="h-9 text-sm pr-10"
                    />
                    <Button
                      type="button"
                      variant="ghost"
                      size="icon"
                      onClick={() => setShowConfirmPassword((prev) => !prev)}
                      className="absolute right-1 top-1/2 h-7 w-7 -translate-y-1/2 text-muted-foreground hover:text-foreground"
                      aria-label={showConfirmPassword ? 'Hide password' : 'Show password'}
                    >
                      {showConfirmPassword ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
                    </Button>
                  </div>
                </div>
                <Button className="w-full h-9 mt-2" type="submit" disabled={isLoading}>
                  {isLoading ? 'Updating...' : 'Update Password'}
                </Button>
              </form>
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  )
}
