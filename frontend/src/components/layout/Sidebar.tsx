'use client';

import { useState } from 'react';
import {
  BarChart3,
  Shield,
  TrendingUp,
  BookOpen,
  ArrowLeftRight,
  Settings,
  ChevronLeft,
  ChevronRight,
} from 'lucide-react';
import { useStore } from '@/lib/store';
import { cn } from '@/lib/utils';

// ── Navigation items ──
const NAV_ITEMS = [
  { id: 'portfolio', label: 'Portfolio', icon: BarChart3 },
  { id: 'risk', label: 'Risk Analytics', icon: Shield },
  { id: 'rankings', label: 'Rankings', icon: TrendingUp },
  { id: 'orders', label: 'Order Book', icon: BookOpen },
  { id: 'trades', label: 'Trades', icon: ArrowLeftRight },
  { id: 'system', label: 'System', icon: Settings },
] as const;

export default function Sidebar() {
  const [expanded, setExpanded] = useState(false);
  const activePanel = useStore((s) => s.activePanel);
  const setActivePanel = useStore((s) => s.setActivePanel);
  const isMobileMenuOpen = useStore((s) => s.isMobileMenuOpen);
  const setMobileMenuOpen = useStore((s) => s.setMobileMenuOpen);

  const handleNavClick = (panelId: string) => {
    setActivePanel(panelId);
    // Close mobile overlay on selection
    if (isMobileMenuOpen) {
      setMobileMenuOpen(false);
    }
  };

  const sidebarContent = (
    <div
      className={cn(
        'h-full flex flex-col bg-bg-secondary border-r border-border transition-all duration-200 ease-in-out',
        expanded ? 'w-[200px]' : 'w-12'
      )}
    >
      {/* ── Toggle button (top) ── */}
      <div className="h-12 flex items-center justify-center border-b border-border shrink-0">
        <button
          onClick={() => setExpanded(!expanded)}
          className="p-1.5 rounded hover:bg-bg-hover text-text-muted hover:text-text-primary transition-colors"
          aria-label={expanded ? 'Collapse sidebar' : 'Expand sidebar'}
        >
          {expanded ? <ChevronLeft size={16} /> : <ChevronRight size={16} />}
        </button>
      </div>

      {/* ── Navigation items ── */}
      <nav className="flex-1 py-2 flex flex-col gap-0.5 overflow-y-auto">
        {NAV_ITEMS.map((item) => {
          const isActive = activePanel === item.id;
          const Icon = item.icon;

          return (
            <button
              key={item.id}
              onClick={() => handleNavClick(item.id)}
              className={cn(
                'relative flex items-center h-10 mx-1 rounded transition-colors duration-150 group',
                isActive
                  ? 'bg-bg-hover text-accent-blue'
                  : 'text-text-muted hover:bg-bg-hover hover:text-text-primary'
              )}
              title={!expanded ? item.label : undefined}
            >
              {/* Active indicator - left accent border */}
              {isActive && (
                <span className="absolute left-0 top-1.5 bottom-1.5 w-0.5 rounded-r bg-accent-blue" />
              )}

              {/* Icon */}
              <span
                className={cn(
                  'shrink-0 flex items-center justify-center',
                  expanded ? 'w-10' : 'w-full'
                )}
              >
                <Icon size={18} strokeWidth={isActive ? 2 : 1.5} />
              </span>

              {/* Label (visible when expanded) */}
              {expanded && (
                <span className="text-xs font-medium truncate pr-2">
                  {item.label}
                </span>
              )}
            </button>
          );
        })}
      </nav>

      {/* ── Bottom: collapse/expand toggle ── */}
      <div className="border-t border-border py-2 shrink-0">
        <button
          onClick={() => setExpanded(!expanded)}
          className={cn(
            'flex items-center h-9 mx-1 rounded text-text-muted hover:bg-bg-hover hover:text-text-primary transition-colors w-[calc(100%-0.5rem)]'
          )}
        >
          <span
            className={cn(
              'shrink-0 flex items-center justify-center',
              expanded ? 'w-10' : 'w-full'
            )}
          >
            {expanded ? <ChevronLeft size={16} /> : <ChevronRight size={16} />}
          </span>
          {expanded && (
            <span className="text-label truncate">Collapse</span>
          )}
        </button>
      </div>
    </div>
  );

  return (
    <>
      {/* ── Desktop sidebar ── */}
      <aside className="hidden lg:block fixed top-12 left-0 bottom-0 z-40">
        {sidebarContent}
      </aside>

      {/* ── Mobile overlay ── */}
      {isMobileMenuOpen && (
        <>
          {/* Backdrop */}
          <div
            className="lg:hidden fixed inset-0 z-40 bg-black/50 backdrop-blur-sm"
            onClick={() => setMobileMenuOpen(false)}
          />
          {/* Sidebar overlay */}
          <aside className="lg:hidden fixed top-12 left-0 bottom-0 z-50">
            {/* Force expanded on mobile for usability */}
            <div className="h-full flex flex-col bg-bg-secondary border-r border-border w-[200px]">
              {/* ── Navigation items (mobile) ── */}
              <nav className="flex-1 py-2 flex flex-col gap-0.5 overflow-y-auto">
                {NAV_ITEMS.map((item) => {
                  const isActive = activePanel === item.id;
                  const Icon = item.icon;

                  return (
                    <button
                      key={item.id}
                      onClick={() => handleNavClick(item.id)}
                      className={cn(
                        'relative flex items-center h-10 mx-1 rounded transition-colors duration-150',
                        isActive
                          ? 'bg-bg-hover text-accent-blue'
                          : 'text-text-muted hover:bg-bg-hover hover:text-text-primary'
                      )}
                    >
                      {isActive && (
                        <span className="absolute left-0 top-1.5 bottom-1.5 w-0.5 rounded-r bg-accent-blue" />
                      )}
                      <span className="shrink-0 flex items-center justify-center w-10">
                        <Icon size={18} strokeWidth={isActive ? 2 : 1.5} />
                      </span>
                      <span className="text-xs font-medium truncate pr-2">
                        {item.label}
                      </span>
                    </button>
                  );
                })}
              </nav>
            </div>
          </aside>
        </>
      )}
    </>
  );
}
