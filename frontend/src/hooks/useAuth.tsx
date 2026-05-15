import { useState, useEffect } from 'react';
import { supabase } from '../lib/supabase';
import { Session } from '@supabase/supabase-js';

export function useAuth() {
  const [session, setSession] = useState<Session | null>(null);
  const [isAuthenticated, setIsAuthenticated] = useState<boolean>(false);
  const [isRecoveryMode, setIsRecoveryMode] = useState<boolean>(false);
  const [isLoading, setIsLoading] = useState<boolean>(true);

  useEffect(() => {
    const hasRecoveryHash = window.location.hash.includes('type=recovery');

    if (hasRecoveryHash) {
      setIsRecoveryMode(true);
    }

    // Get initial session
    supabase.auth.getSession().then(({ data: { session } }) => {
      setSession(session);
      setIsAuthenticated(!!session);
      setIsLoading(false);
    });

    // Listen for auth changes
    const { data: { subscription } } = supabase.auth.onAuthStateChange((event, session) => {
      setSession(session);

      if (event === 'PASSWORD_RECOVERY') {
        setIsRecoveryMode(true);
        setIsAuthenticated(false);
        return;
      }

      if (event === 'SIGNED_OUT') {
        setIsRecoveryMode(false);
        setIsAuthenticated(false);
        return;
      }

      if (event === 'SIGNED_IN') {
        setIsAuthenticated(true);
        if (!window.location.hash.includes('type=recovery')) {
          setIsRecoveryMode(false);
        }
        return;
      }

      setIsAuthenticated(!!session);
    });

    return () => subscription.unsubscribe();
  }, []);

  const logout = async () => {
    await supabase.auth.signOut();
  };

  return { session, isAuthenticated, isRecoveryMode, isLoading, logout };
}