/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
  theme: {
    extend: {
      boxShadow: {
        glow: '0 0 30px rgba(34, 197, 94, 0.18)',
      },
      colors: {
        brand: {
          50: '#ecfeff',
          500: '#06b6d4',
          600: '#0ea5e9',
        },
      },
    },
  },
  plugins: [],
}
