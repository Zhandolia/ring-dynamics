// Ring Dynamics mark: a pose-tracked fighter throwing a jab inside a tracking box. Kept in sync with app/icon.svg (favicon).
const JOINTS: [number, number][] = [
    [22, 24], [32, 24], [18, 30], [23, 32], [40, 23], [49, 22],
    [24, 34], [30, 34], [19, 42], [16, 49], [36, 41], [37, 49],
]

export default function Logo({ size = 24 }: { size?: number }) {
    return (
        <svg width={size} height={size} viewBox="0 0 64 64" aria-hidden="true" className="shrink-0">
            <rect width="64" height="64" rx="14" fill="#000000" />
            <path d="M10 17V10h7M47 10h7v7M54 47v7h-7M17 54h-7v-7" fill="none" stroke="#22c55e" strokeWidth="2" strokeLinecap="round" opacity=".55" />
            <g fill="none" stroke="#22c55e" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round">
                <circle cx="27" cy="16" r="4.5" />
                <path d="M27 21v13" />
                <path d="M22 24h10" />
                <path d="M22 24l-4 6 5 2" />
                <path d="M32 24l8-1 9-1" />
                <path d="M24 34h6" />
                <path d="M24 34l-5 8-3 7" />
                <path d="M30 34l6 7 1 8" />
            </g>
            <g fill="#bbf7d0">
                {JOINTS.map(([cx, cy]) => <circle key={`${cx}-${cy}`} cx={cx} cy={cy} r="2" />)}
            </g>
        </svg>
    )
}
