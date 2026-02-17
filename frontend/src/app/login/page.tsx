'use client';

import { Suspense, useEffect, useState } from 'react';
import { useRouter, useSearchParams } from 'next/navigation';
import { api } from '@/lib/api';
import { useStore } from '@/lib/store';

function LoginContent() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const setUser = useStore((s) => s.setUser);
  const authBypass = process.env.NEXT_PUBLIC_AUTH_BYPASS_LOCAL === 'true';
  const [status, setStatus] = useState<'idle' | 'loading' | 'callback' | 'totp' | 'error'>('idle');
  const [error, setError] = useState('');
  const [totpCode, setTotpCode] = useState('');
  const [totpUri, setTotpUri] = useState('');
  const [callbackData, setCallbackData] = useState<any>(null);

  // Handle OAuth redirect: backend redirects here with ?token=...&requires_totp=...
  useEffect(() => {
    if (authBypass) {
      api
        .getMe()
        .then((data) => {
          setUser(data);
          router.push('/dashboard');
        })
        .catch(() => {
          // allow manual local entry button below
        });
      return;
    }

    const token = searchParams.get('token');
    const errorParam = searchParams.get('error');
    const totpSetupParam = searchParams.get('totp_setup_uri');
    const requiresTotpParam = searchParams.get('requires_totp');

    if (errorParam) {
      setError(decodeURIComponent(errorParam));
      setStatus('error');
      return;
    }

    if (token) {
      setStatus('callback');
      api.setToken(token);

      const requiresTotp = requiresTotpParam === 'true';

      if (requiresTotp) {
        setStatus('totp');
      } else if (totpSetupParam) {
        setTotpUri(decodeURIComponent(totpSetupParam));
        setStatus('totp');
      } else {
        // Token received, TOTP not required — fetch user and go to dashboard
        api
          .getMe()
          .then((data) => {
            setUser(data);
            router.push('/dashboard');
          })
          .catch((err) => {
            setError(err.message || 'Login failed');
            setStatus('error');
          });
      }
    }
  }, [searchParams, setUser, router, authBypass]);

  // Check if already logged in (existing token in localStorage)
  useEffect(() => {
    if (authBypass) return;
    const existingToken = api.getToken();
    if (existingToken && !searchParams.get('token') && !searchParams.get('error')) {
      api
        .getMe()
        .then((data) => {
          setUser(data);
          router.push('/dashboard');
        })
        .catch(() => {
          api.clearToken();
        });
    }
  }, [setUser, router, searchParams, authBypass]);

  const handleLogin = async () => {
    setStatus('loading');
    setError('');
    try {
      const { auth_url } = await api.getLoginUrl();
      window.location.href = auth_url;
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : 'Failed to initiate login');
      setStatus('error');
    }
  };

  const handleTotpVerify = async () => {
    if (totpCode.length !== 6) return;
    try {
      const result = await api.verifyTotp(totpCode);
      if (result.totp_verified || result.verified) {
        if (result.access_token) {
          api.setToken(result.access_token);
        }
        // Fetch user data and redirect to dashboard
        const userData = await api.getMe();
        setUser(userData);
        router.push('/dashboard');
      } else {
        setError('Invalid TOTP code. Try again.');
      }
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : 'TOTP verification failed');
    }
  };

  return (
    <div className="min-h-screen bg-bg-primary flex items-center justify-center p-4">
      <div className="w-full max-w-sm space-y-6">
        {/* Logo / Title */}
        <div className="text-center space-y-2">
          <h1 className="text-2xl font-bold text-text-primary tracking-tight">
            QuantDesk Pro
          </h1>
          <p className="text-sm text-text-muted">
            Professional Trading Analytics
          </p>
        </div>

        {/* Card */}
        <div className="panel p-6 space-y-4">
          {status === 'totp' ? (
            <>
              <div className="space-y-2">
                <h2 className="text-sm font-semibold text-text-primary">
                  {totpUri ? 'Set Up Two-Factor Authentication' : 'Enter TOTP Code'}
                </h2>
                {totpUri && (
                  <div className="space-y-2">
                    <p className="text-xs text-text-muted">
                      Scan this URI with Google Authenticator or any TOTP app:
                    </p>
                    <div className="bg-bg-tertiary rounded p-2 text-xs font-mono text-text-secondary break-all">
                      {totpUri}
                    </div>
                  </div>
                )}
                <p className="text-xs text-text-muted">
                  Enter the 6-digit code from your authenticator app.
                </p>
              </div>
              <input
                type="text"
                inputMode="numeric"
                maxLength={6}
                placeholder="000000"
                value={totpCode}
                onChange={(e) => setTotpCode(e.target.value.replace(/\D/g, ''))}
                className="input text-center text-lg font-mono tracking-[0.5em]"
                autoFocus
              />
              <button
                onClick={handleTotpVerify}
                disabled={totpCode.length !== 6}
                className="w-full btn btn-primary py-2.5 text-sm disabled:opacity-50"
              >
                Verify
              </button>
            </>
          ) : (
            <>
              {authBypass ? (
                <>
                  <p className="text-sm text-text-secondary text-center">
                    Local auth bypass is enabled. Google OAuth is disabled for localhost.
                  </p>
                  <button
                    onClick={() => router.push('/dashboard')}
                    className="w-full btn btn-primary py-2.5 text-sm"
                  >
                    Enter Local Dashboard
                  </button>
                </>
              ) : (
                <>
                  <p className="text-sm text-text-secondary text-center">
                    Sign in with your authorized Google account to access the dashboard.
                  </p>
                  <button
                    onClick={handleLogin}
                    disabled={status === 'loading' || status === 'callback'}
                    className="w-full flex items-center justify-center gap-2 bg-white text-gray-800
                               font-medium py-2.5 px-4 rounded-md text-sm
                               hover:bg-gray-100 disabled:opacity-50 transition-colors"
                  >
                    {(status === 'loading' || status === 'callback') && (
                      <div className="w-4 h-4 border-2 border-gray-400 border-t-gray-800 rounded-full animate-spin" />
                    )}
                    {status === 'callback'
                      ? 'Authenticating...'
                      : status === 'loading'
                      ? 'Redirecting...'
                      : 'Sign in with Google'}
                  </button>
                </>
              )}
            </>
          )}

          {error && (
            <div className="bg-loss/10 border border-loss/20 rounded-md p-3 text-xs text-loss">
              {error}
            </div>
          )}
        </div>

        <p className="text-center text-xs text-text-muted">
          {authBypass
            ? 'Local mode: authentication bypass enabled.'
            : 'Single-user platform. Only authorized accounts can sign in.'}
        </p>
      </div>
    </div>
  );
}

export default function LoginPage() {
  return (
    <Suspense
      fallback={
        <div className="min-h-screen bg-bg-primary flex items-center justify-center">
          <div className="w-5 h-5 border-2 border-accent-blue/30 border-t-accent-blue rounded-full animate-spin" />
        </div>
      }
    >
      <LoginContent />
    </Suspense>
  );
}
