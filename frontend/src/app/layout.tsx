import type { Metadata } from 'next'
import './globals.css'

export const metadata: Metadata = {
    title: 'Ring Dynamics - Boxing Analytics',
    description: 'Production-grade boxing analytics with computer vision and Bayesian scoring',
}

export default function RootLayout({
    children,
}: {
    children: React.ReactNode
}) {
    return (
        <html lang="en">
            <body>{children}</body>
        </html>
    )
}
