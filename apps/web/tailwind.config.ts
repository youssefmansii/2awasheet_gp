import type { Config } from 'tailwindcss';

export default {
  content: [
    './index.html',
    './src/**/*.{ts,tsx}',
    './apps/web/index.html',
    './apps/web/src/**/*.{ts,tsx}'
  ],
  theme: {
    extend: {
      colors: {
        ink: '#080b0d',
        panel: '#14191d',
        line: '#2b3339',
        lime: '#c8f45a',
        muted: '#93a0a8'
      },
      boxShadow: {
        panel: '0 24px 70px rgba(0, 0, 0, 0.35)'
      }
    }
  },
  plugins: []
} satisfies Config;
