'use client';

import { useEffect } from 'react';
import { useRouter } from 'next/navigation';
import DashboardLayout from '@/components/layout/DashboardLayout';
import StocksPanel from '@/components/dashboard/StocksPanel';
import { api } from '@/lib/api';

export default function StocksPage() {
  const router = useRouter();
  const authBypass = process.env.NEXT_PUBLIC_AUTH_BYPASS_LOCAL === 'true';

  useEffect(() => {
    const token = api.getToken();
    if (!token && !authBypass) router.push('/login');
  }, [authBypass, router]);

  return (
    <DashboardLayout>
      <StocksPanel />
    </DashboardLayout>
  );
}
