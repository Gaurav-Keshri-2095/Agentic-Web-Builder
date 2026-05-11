import { useState, useEffect } from 'react';

export function useAuth() {
  const [isAuthenticated, setIsAuthenticated] = useState<boolean>(false);
  const [isLoading, setIsLoading] = useState<boolean>(true);

  useEffect(() => {
    const isAuth = localStorage.getItem('authenticated');
    if (isAuth === 'true') {
      setIsAuthenticated(true);
    }
    setIsLoading(false);
  }, []);

  const login = () => {
    localStorage.setItem('authenticated', 'true');
    setIsAuthenticated(true);
  };

  const logout = () => {
    localStorage.removeItem('authenticated');
    setIsAuthenticated(false);
  };

  return { isAuthenticated, isLoading, login, logout };
}