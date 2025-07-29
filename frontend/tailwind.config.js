/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./src/**/*.{js,jsx,ts,tsx}",
    "./public/index.html"
  ],
  theme: {
    extend: {
      colors: {
        'purple': {
          50: '#faf5ff',
          100: '#f3e8ff',
          500: '#8b5cf6',
          600: '#7c3aed',
          900: '#581c87'
        },
        'slate': {
          900: '#0f172a'
        }
      },
      animation: {
        'bounce': 'bounce 1s infinite',
        'fadeIn': 'fadeIn 0.3s ease-out',
        'slideIn': 'slideIn 0.3s ease-out'
      },
      backdropBlur: {
        'xl': '20px'
      }
    },
  },
  plugins: [],
}