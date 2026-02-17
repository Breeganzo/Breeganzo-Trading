import { redirect } from 'next/navigation';

export default function RootPage() {
  if (process.env.NEXT_PUBLIC_AUTH_BYPASS_LOCAL === 'true') {
    redirect('/dashboard');
  }
  redirect('/login');
}
