'use client';

import { ReactNode } from 'react';
import TopBar from './TopBar';
import Sidebar from './Sidebar';
import { useStore } from '@/lib/store';

interface DashboardLayoutProps {
  children: ReactNode;
}

export default function DashboardLayout({ children }: DashboardLayoutProps) {
  const isMobileMenuOpen = useStore((s) => s.isMobileMenuOpen);

  return (
    <div className="min-h-screen bg-bg-primary text-text-primary">
      {/* ── Fixed top bar ── */}
      <TopBar />

      {/* ── Sidebar ── */}
      <Sidebar />

      {/* ── Main content area ── */}
      {/*
        Desktop: offset by sidebar width (48px collapsed) and top bar (48px).
        The sidebar expand/collapse is handled internally by the Sidebar component,
        but the main content uses the collapsed width as the baseline margin so it
        never overlaps the icon rail. When the sidebar expands it overlays on top.
        Mobile: full width, no left margin.
      */}
      <main
        className={`
          pt-12
          lg:ml-12
          min-h-screen
          transition-all duration-200
        `}
      >
        <div className="p-4 lg:p-6">
          {children}
        </div>
      </main>

      {/* ── Mobile menu open body scroll lock ── */}
      {isMobileMenuOpen && (
        <style jsx global>{`
          body {
            overflow: hidden;
          }
        `}</style>
      )}
    </div>
  );
}
