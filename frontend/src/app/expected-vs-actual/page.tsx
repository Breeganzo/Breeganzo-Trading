'use client';

import { useEffect } from 'react';
import { useRouter } from 'next/navigation';
import DashboardLayout from '@/components/layout/DashboardLayout';
import ExpectedActualPanel from '@/components/dashboard/ExpectedActualPanel';
import { api } from '@/lib/api';

export default function ExpectedVsActualPage() {
  const router = useRouter();
  const authBypass = process.env.NEXT_PUBLIC_AUTH_BYPASS_LOCAL === 'true';

  useEffect(() => {
    const token = api.getToken();
    if (!token && !authBypass) router.push('/login');
  }, [authBypass, router]);

  return (
    <DashboardLayout>
      <ExpectedActualPanel />
    </DashboardLayout>
  );
}
