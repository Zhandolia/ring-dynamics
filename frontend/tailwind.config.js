/** @type {import('tailwindcss').Config} */
module.exports = {
    content: [
        './src/pages/**/*.{js,ts,jsx,tsx,mdx}',
        './src/components/**/*.{js,ts,jsx,tsx,mdx}',
        './src/app/**/*.{js,ts,jsx,tsx,mdx}',
    ],
    theme: {
        extend: {
            colors: {
                'fighter-blue': 'rgb(59, 130, 246)',
                'fighter-red': 'rgb(239, 68, 68)',
            },
            animation: {
                'punch-hit': 'punchHit 0.3s ease-out',
            },
            keyframes: {
                punchHit: {
                    '0%, 100%': { transform: 'scale(1)', opacity: 1 },
                    '50%': { transform: 'scale(1.5)', opacity: 0.8 },
                }
            }
        },
    },
    plugins: [],
}
