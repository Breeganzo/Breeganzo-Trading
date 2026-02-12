/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    './src/pages/**/*.{js,ts,jsx,tsx,mdx}',
    './src/components/**/*.{js,ts,jsx,tsx,mdx}',
    './src/app/**/*.{js,ts,jsx,tsx,mdx}',
  ],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        // Institutional dark theme palette
        bg: {
          primary: '#0a0e17',
          secondary: '#111827',
          tertiary: '#1a2332',
          elevated: '#1e293b',
          hover: '#243044',
        },
        border: {
          DEFAULT: '#1e293b',
          light: '#2a3a4e',
          focus: '#3b82f6',
        },
        text: {
          primary: '#e2e8f0',
          secondary: '#94a3b8',
          muted: '#64748b',
          inverse: '#0f172a',
        },
        accent: {
          blue: '#3b82f6',
          cyan: '#06b6d4',
          indigo: '#6366f1',
        },
        profit: '#10b981',
        loss: '#ef4444',
        warning: '#f59e0b',
        neutral: '#6b7280',
        // Regime colors
        regime: {
          bull: '#10b981',
          bear: '#ef4444',
          high_vol: '#f59e0b',
          low_vol: '#3b82f6',
        },
      },
      fontFamily: {
        mono: ['JetBrains Mono', 'Fira Code', 'monospace'],
        sans: ['Inter', 'system-ui', 'sans-serif'],
      },
      fontSize: {
        'data': ['0.75rem', { lineHeight: '1rem' }],
        'label': ['0.6875rem', { lineHeight: '0.875rem' }],
      },
      animation: {
        'ticker': 'ticker 30s linear infinite',
        'pulse-subtle': 'pulse-subtle 2s ease-in-out infinite',
      },
      keyframes: {
        ticker: {
          '0%': { transform: 'translateX(0)' },
          '100%': { transform: 'translateX(-50%)' },
        },
        'pulse-subtle': {
          '0%, 100%': { opacity: '1' },
          '50%': { opacity: '0.7' },
        },
      },
    },
  },
  plugins: [],
};
