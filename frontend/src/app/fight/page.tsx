'use client';

import { useState, useEffect, useCallback, useRef } from 'react';

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

interface FightData {
    id: string;
    status: string;
    created_at: string;
    video_url?: string;
    annotated_video_url?: string;
    processing_time?: number;
    metrics_url?: string;
}

interface MetricsData {
    total_frames: number;
    fps: number;
    duration: number;
    processing_time: number;
    final_scores: number[];
    final_activity: number[];
    final_aggression: number[];
    final_ring_control: number[];
    final_pressure: number[];
    events: { frame: number; text: string }[];
    timeline: {
        time: number;
        activity: number[];
        aggression: number[];
        ring_control: number[];
        pressure: number[];
        distance: string;
        round_pts: number[];
    }[];
}

export default function FightPage() {
    const videoRef = useRef<HTMLVideoElement>(null);

    const [fightId, setFightId] = useState<string | null>(null);
    const [fight, setFight] = useState<FightData | null>(null);
    const [metrics, setMetrics] = useState<MetricsData | null>(null);
    const [error, setError] = useState<string | null>(null);
    const [dots, setDots] = useState('');
    const [currentTime, setCurrentTime] = useState(0);

    // Extract fight ID from URL hash: /fight/#<id>
    useEffect(() => {
        const hash = window.location.hash.replace('#', '');
        if (hash) {
            setFightId(hash);
        } else {
            // Also check query params: /fight/?id=<id>
            const params = new URLSearchParams(window.location.search);
            const id = params.get('id');
            if (id) setFightId(id);
            else setError('No fight ID provided');
        }
    }, []);

    // Animate processing dots
    useEffect(() => {
        const interval = setInterval(() => {
            setDots(prev => prev.length >= 3 ? '' : prev + '.');
        }, 500);
        return () => clearInterval(interval);
    }, []);

    const fetchStatus = useCallback(async () => {
        if (!fightId) return;
        try {
            const res = await fetch(`${API_URL}/api/fights/${fightId}`);
            if (!res.ok) {
                if (res.status === 404) { setError('Fight not found'); return; }
                throw new Error(`HTTP ${res.status}`);
            }
            const data = await res.json();
            setFight(data);

            // Fetch metrics when completed
            if (data.status === 'completed' && !metrics) {
                try {
                    const mRes = await fetch(`${API_URL}/api/fights/${fightId}/metrics`);
                    if (mRes.ok) {
                        const mData = await mRes.json();
                        setMetrics(mData);
                    }
                } catch { /* metrics optional */ }
            }
        } catch (err) {
            console.error('Failed to fetch fight status:', err);
            setError('Failed to connect to server');
        }
    }, [fightId, metrics]);

    // Poll status
    useEffect(() => {
        if (!fightId) return;
        fetchStatus();
        const interval = setInterval(() => {
            if (fight?.status === 'completed' || fight?.status === 'failed') {
                clearInterval(interval);
                return;
            }
            fetchStatus();
        }, 2000);
        return () => clearInterval(interval);
    }, [fetchStatus, fightId, fight?.status]);

    // Video time tracking
    useEffect(() => {
        const video = videoRef.current;
        if (!video) return;
        const handler = () => setCurrentTime(video.currentTime);
        video.addEventListener('timeupdate', handler);
        return () => video.removeEventListener('timeupdate', handler);
    }, [fight?.status]);

    // Get current metrics snapshot based on video time
    const getCurrentSnapshot = () => {
        if (!metrics?.timeline?.length) return null;
        const idx = metrics.timeline.findIndex(s => s.time > currentTime);
        if (idx <= 0) return metrics.timeline[0];
        return metrics.timeline[idx - 1];
    };

    const snapshot = getCurrentSnapshot();

    // Compute dominance %
    const getDominance = () => {
        if (!snapshot) return [50, 50];
        const a = (snapshot.activity[0] + snapshot.aggression[0] + snapshot.ring_control[0] * 10 + snapshot.pressure[0]);
        const b = (snapshot.activity[1] + snapshot.aggression[1] + snapshot.ring_control[1] * 10 + snapshot.pressure[1]);
        const total = a + b || 1;
        return [Math.round(a / total * 100), Math.round(b / total * 100)];
    };

    const [domA, domB] = getDominance();

    const goHome = () => {
        const base = process.env.NODE_ENV === 'production' ? '/ring-dynamics' : '';
        window.location.href = `${base}/`;
    };

    if (error) {
        return (
            <main className="min-h-screen bg-[#0a0a0f] text-white flex items-center justify-center">
                <div className="text-center">
                    <div className="text-6xl mb-4">❌</div>
                    <h1 className="text-2xl font-bold mb-2">Error</h1>
                    <p className="text-gray-400 mb-6">{error}</p>
                    <button onClick={goHome}
                        className="bg-red-600 hover:bg-red-700 px-6 py-3 rounded-lg font-semibold transition-colors">
                        ← Back to Upload
                    </button>
                </div>
            </main>
        );
    }

    return (
        <main className="min-h-screen bg-[#0a0a0f] text-white">
            {/* ── Nav Bar ─────────────────────────────────────────── */}
            <nav className="flex items-center justify-between px-6 py-3 border-b border-[#1f1f2e]">
                <div className="flex items-center gap-3">
                    <button onClick={goHome}
                        className="text-gray-500 hover:text-white transition-colors text-sm">
                        ←
                    </button>
                    <h1 className="text-lg font-bold tracking-wider uppercase"
                        style={{ color: '#e53e3e' }}>
                        🥊 Ring Dynamics
                    </h1>
                </div>
                <div className="flex items-center gap-4 text-xs">
                    {fight && (
                        <span className="text-gray-500">
                            {fight.status === 'completed' ? '● Completed' :
                                fight.status === 'annotating' ? `● Analyzing${dots}` :
                                    `● ${fight.status}${dots}`}
                        </span>
                    )}
                </div>
            </nav>

            {/* ── Loading / No fight ID ────────────────────────────── */}
            {!fight && !error && (
                <div className="flex items-center justify-center" style={{ height: 'calc(100vh - 52px)' }}>
                    <div className="text-center">
                        <div className="text-5xl mb-4">⏳</div>
                        <p className="text-gray-400">Loading fight data{dots}</p>
                    </div>
                </div>
            )}

            {/* ── Processing State ────────────────────────────────── */}
            {fight && fight.status !== 'completed' && fight.status !== 'failed' && (
                <div className="flex items-center justify-center" style={{ height: 'calc(100vh - 52px)' }}>
                    <div className="text-center max-w-md">
                        <div className="text-5xl mb-6">⚙️</div>
                        <h2 className="text-xl font-bold mb-3">
                            {fight.status === 'pending' ? 'Preparing Analysis' : 'Analyzing Fight'}
                        </h2>
                        <div className="w-full h-1.5 rounded-full bg-[#1f1f2e] overflow-hidden mb-4">
                            <div className="h-full rounded-full progress-shimmer"
                                style={{ width: fight.status === 'pending' ? '15%' : '60%' }} />
                        </div>
                        <p className="text-sm text-gray-500">
                            {fight.status === 'pending'
                                ? 'Loading YOLOv8 model and preparing video...'
                                : 'Running 3-box fighter tracking with scoring...'}
                        </p>
                    </div>
                </div>
            )}

            {/* ── Failed State ────────────────────────────────────── */}
            {fight?.status === 'failed' && (
                <div className="flex items-center justify-center" style={{ height: 'calc(100vh - 52px)' }}>
                    <div className="text-center">
                        <div className="text-5xl mb-4">❌</div>
                        <p className="text-red-300 mb-6">Analysis failed. Please try again.</p>
                        <button onClick={goHome}
                            className="bg-red-600 hover:bg-red-700 px-6 py-3 rounded-lg font-semibold">
                            Try Again
                        </button>
                    </div>
                </div>
            )}

            {/* ── Completed: Split Layout ─────────────────────────── */}
            {fight?.status === 'completed' && fight.annotated_video_url && (
                <div className="flex" style={{ height: 'calc(100vh - 52px)' }}>

                    {/* ── LEFT: Video Player (~70%) ─────────────────── */}
                    <div className="flex-1 flex flex-col bg-black relative">
                        <video
                            ref={videoRef}
                            key={fight.annotated_video_url}
                            controls
                            autoPlay
                            muted
                            playsInline
                            className="w-full h-full object-contain"
                        >
                            <source
                                src={`${API_URL}${fight.annotated_video_url}`}
                                type="video/mp4"
                            />
                        </video>

                        {/* Round / Timer overlay */}
                        <div className="absolute bottom-14 left-4 bg-[#e53e3e] text-white text-xs font-bold px-3 py-1.5 rounded">
                            RND 1 of 12 &nbsp;&nbsp; {Math.floor(currentTime / 60)}:{String(Math.floor(currentTime % 60)).padStart(2, '0')}
                        </div>
                    </div>

                    {/* ── RIGHT: Stats Panel (~30%) ─────────────────── */}
                    <div className="w-[340px] flex-shrink-0 border-l border-[#1f1f2e] bg-[#0d0d14] overflow-y-auto">
                        <div className="p-5">

                            {/* LIVE ANALYSIS header */}
                            <div className="flex items-center justify-between mb-5">
                                <h3 className="text-xs font-bold tracking-widest text-gray-400 uppercase">
                                    Live Analysis
                                </h3>
                                <div className="flex items-center gap-1.5">
                                    <div className="pulse-dot" />
                                    <span className="text-xs font-bold text-red-400">LIVE</span>
                                </div>
                            </div>

                            {/* Fighter dominance bar */}
                            <div className="mb-6">
                                <div className="flex justify-between text-sm font-semibold mb-1.5">
                                    <span className="fighter-a-color">Fighter A</span>
                                    <span className="fighter-b-color">Fighter B</span>
                                </div>
                                <div className="flex h-2.5 rounded-full overflow-hidden bg-[#1f1f2e]">
                                    <div className="bg-fighter-a transition-all duration-500"
                                        style={{ width: `${domA}%` }} />
                                    <div className="bg-fighter-b transition-all duration-500"
                                        style={{ width: `${domB}%` }} />
                                </div>
                                <div className="flex justify-between mt-1.5">
                                    <span className="text-2xl font-bold fighter-a-color">{domA}%</span>
                                    <span className="text-2xl font-bold fighter-b-color">{domB}%</span>
                                </div>
                            </div>

                            {/* Stats table */}
                            <div className="grid grid-cols-3 gap-x-3 text-xs mb-6">
                                <div className="text-center">
                                    <h4 className="font-bold fighter-a-color text-xs mb-2 uppercase tracking-wider">Fighter A</h4>
                                </div>
                                <div />
                                <div className="text-center">
                                    <h4 className="font-bold fighter-b-color text-xs mb-2 uppercase tracking-wider">Fighter B</h4>
                                </div>

                                <div className="text-right font-mono text-white">
                                    {snapshot ? snapshot.activity[0].toFixed(1) : '—'}
                                </div>
                                <div className="text-center text-gray-500">Activity</div>
                                <div className="text-left font-mono text-white">
                                    {snapshot ? snapshot.activity[1].toFixed(1) : '—'}
                                </div>

                                <div className="text-right font-mono text-white">
                                    {snapshot ? snapshot.aggression[0].toFixed(1) : '—'}
                                </div>
                                <div className="text-center text-gray-500">Aggression</div>
                                <div className="text-left font-mono text-white">
                                    {snapshot ? snapshot.aggression[1].toFixed(1) : '—'}
                                </div>

                                <div className="text-right font-mono text-white">
                                    {snapshot ? `${Math.round(snapshot.ring_control[0] * 100)}%` : '—'}
                                </div>
                                <div className="text-center text-gray-500">Ring Ctrl</div>
                                <div className="text-left font-mono text-white">
                                    {snapshot ? `${Math.round(snapshot.ring_control[1] * 100)}%` : '—'}
                                </div>

                                <div className="text-right font-mono text-white">
                                    {snapshot ? snapshot.pressure[0].toFixed(1) : '—'}
                                </div>
                                <div className="text-center text-gray-500">Pressure</div>
                                <div className="text-left font-mono text-white">
                                    {snapshot ? snapshot.pressure[1].toFixed(1) : '—'}
                                </div>

                                <div className="col-span-3 mt-2 text-center">
                                    <span className="text-gray-500 text-xs">Distance: </span>
                                    <span className="text-white font-semibold text-xs px-2 py-0.5 rounded bg-[#1f1f2e]">
                                        {snapshot?.distance || '—'}
                                    </span>
                                </div>

                                <div className="text-center mt-1">
                                    <span className="text-xs px-2 py-0.5 rounded font-semibold"
                                        style={{ background: 'rgba(229,62,62,0.15)', color: '#e53e3e' }}>
                                        Orthodox
                                    </span>
                                </div>
                                <div className="text-center text-gray-500 mt-1">Stance</div>
                                <div className="text-center mt-1">
                                    <span className="text-xs px-2 py-0.5 rounded font-semibold"
                                        style={{ background: 'rgba(66,153,225,0.15)', color: '#4299e1' }}>
                                        Orthodox
                                    </span>
                                </div>
                            </div>

                            <div className="border-t border-[#1f1f2e] mb-4" />

                            {/* EVENT FEED */}
                            <h3 className="text-xs font-bold tracking-widest text-gray-400 uppercase mb-3">
                                Event Feed
                            </h3>
                            <div className="mb-6">
                                {metrics?.events && metrics.events.length > 0 ? (
                                    metrics.events.slice(-6).reverse().map((ev, i) => {
                                        const sec = ev.frame / (metrics?.fps || 30);
                                        const mm = Math.floor(sec / 60);
                                        const ss = Math.floor(sec % 60);
                                        return (
                                            <div key={i} className="event-row">
                                                <span className="text-lg">
                                                    {ev.text.includes('presses') ? '💥' :
                                                        ev.text.includes('controls') ? '🎯' : '🥊'}
                                                </span>
                                                <span className="flex-1 text-gray-300">{ev.text}</span>
                                                <span className="text-gray-500 font-mono text-xs">
                                                    {mm}:{String(ss).padStart(2, '0')}
                                                </span>
                                            </div>
                                        );
                                    })
                                ) : (
                                    <p className="text-gray-600 text-xs text-center py-4">
                                        No events yet
                                    </p>
                                )}
                            </div>

                            <div className="border-t border-[#1f1f2e] mb-4" />

                            {/* SCORECARD */}
                            <h3 className="text-xs font-bold tracking-widest text-gray-400 uppercase mb-3">
                                Scorecard
                            </h3>
                            <div className="mb-6">
                                <div className="flex items-center gap-2 text-xs">
                                    <span className="text-gray-500 w-6">R1</span>
                                    <div className="flex-1 flex h-2 rounded-full overflow-hidden bg-[#1f1f2e]">
                                        <div className="bg-fighter-a"
                                            style={{ width: `${snapshot ? snapshot.round_pts[0] * 10 : 50}%` }} />
                                        <div className="bg-fighter-b"
                                            style={{ width: `${snapshot ? snapshot.round_pts[1] * 10 : 50}%` }} />
                                    </div>
                                    <span className="fighter-a-color font-bold w-4 text-right">
                                        {metrics?.final_scores?.[0] || '—'}
                                    </span>
                                    <span className="fighter-b-color font-bold w-4">
                                        {metrics?.final_scores?.[1] || '—'}
                                    </span>
                                </div>
                            </div>

                            <div className="border-t border-[#1f1f2e] mb-4" />

                            {/* Action buttons */}
                            <div className="space-y-2">
                                <button
                                    onClick={async () => {
                                        const btn = document.getElementById('dl-btn');
                                        if (btn) btn.textContent = '⏳ Downloading...';
                                        try {
                                            const res = await fetch(`${API_URL}/api/fights/${fightId}/video`);
                                            if (!res.ok) throw new Error('Failed to fetch');
                                            const blob = await res.blob();
                                            const file = new Blob([blob], { type: 'video/mp4' });
                                            const url = window.URL.createObjectURL(file);
                                            const a = document.createElement('a');
                                            a.style.display = 'none';
                                            a.href = url;
                                            a.download = `ring_dynamics_${(fightId || 'fight').slice(0, 8)}.mp4`;
                                            document.body.appendChild(a);
                                            a.click();
                                            setTimeout(() => {
                                                window.URL.revokeObjectURL(url);
                                                document.body.removeChild(a);
                                            }, 200);
                                        } catch { alert('Download failed. Is the backend running?'); }
                                        if (btn) btn.textContent = '⬇️ Download Video';
                                    }}
                                    id="dl-btn"
                                    className="w-full py-2.5 rounded-lg font-semibold text-sm text-white transition-all cursor-pointer"
                                    style={{ background: 'linear-gradient(135deg, #e53e3e, #c53030)' }}
                                >
                                    ⬇️ Download Video
                                </button>
                                <button
                                    onClick={goHome}
                                    className="w-full py-2.5 rounded-lg font-semibold text-sm text-gray-400 border border-[#1f1f2e] hover:border-gray-600 hover:text-white transition-all"
                                >
                                    Upload Another
                                </button>
                            </div>

                            {fight.processing_time && (
                                <p className="text-xs text-gray-600 text-center mt-4">
                                    Processed in {fight.processing_time}s • {metrics?.total_frames || '?'} frames
                                </p>
                            )}
                        </div>
                    </div>
                </div>
            )}
        </main>
    );
}
