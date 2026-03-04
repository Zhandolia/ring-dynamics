'use client';

import { useState, useEffect } from 'react';

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

export default function Home() {
    const [file, setFile] = useState<File | null>(null);
    const [youtubeUrl, setYoutubeUrl] = useState('');
    const [uploading, setUploading] = useState(false);
    const [uploadProgress, setUploadProgress] = useState('');
    const [apiStatus, setApiStatus] = useState<'checking' | 'online' | 'offline'>('checking');

    // Check if backend API is reachable
    useEffect(() => {
        fetch(`${API_URL}/health`, { mode: 'cors' })
            .then(r => r.ok ? setApiStatus('online') : setApiStatus('offline'))
            .catch(() => setApiStatus('offline'));
    }, []);

    const navigateToFight = (id: string) => {
        const base = process.env.NODE_ENV === 'production' ? '/ring-dynamics' : '';
        window.location.href = `${base}/fight/#${id}`;
    };

    const handleFileUpload = async (e: React.FormEvent) => {
        e.preventDefault();
        if (!file) return;

        setUploading(true);
        setUploadProgress('Uploading video...');

        // Render's proxy limits uploads to ~100MB
        const isDeployed = !API_URL.includes('localhost');
        const maxSizeMB = isDeployed ? 95 : 500;
        if (file.size > maxSizeMB * 1024 * 1024) {
            alert(`File too large (${(file.size / 1024 / 1024).toFixed(0)}MB).\n\n${isDeployed
                ? `The hosted version supports up to ${maxSizeMB}MB due to Render's proxy limits.\n\nFor larger videos, run the app locally:\n  cd backend && uvicorn app.main:app --port 8000\n  cd frontend && npm run dev`
                : `Max file size: ${maxSizeMB}MB.`
                }`);
            setUploading(false);
            setUploadProgress('');
            return;
        }

        const formData = new FormData();
        formData.append('file', file);

        try {
            const response = await fetch(`${API_URL}/api/fights/upload`, {
                method: 'POST',
                body: formData,
            });

            if (!response.ok) {
                const err = await response.json();
                throw new Error(err.detail || `Upload failed (${response.status})`);
            }

            const data = await response.json();
            setUploadProgress('Redirecting to analysis...');
            navigateToFight(data.id);
        } catch (error: unknown) {
            console.error('Upload failed:', error);
            const message = error instanceof Error ? error.message : 'Upload failed';
            alert(`Upload failed: ${message}`);
        } finally {
            setUploading(false);
            setUploadProgress('');
        }
    };

    const handleYoutubeSubmit = async (e: React.FormEvent) => {
        e.preventDefault();
        if (!youtubeUrl) return;

        setUploading(true);
        try {
            const response = await fetch(`${API_URL}/api/fights/youtube`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ youtube_url: youtubeUrl }),
            });
            const data = await response.json();
            navigateToFight(data.id);
        } catch (error) {
            console.error('Failed:', error);
            alert('Failed. Is the backend running?');
        } finally {
            setUploading(false);
        }
    };

    return (
        <main className="relative min-h-screen text-white overflow-hidden">

            {/* ── Background Video ──────────────────────────────── */}
            <video
                autoPlay
                muted
                loop
                playsInline
                className="absolute inset-0 w-full h-full object-cover"
                style={{ zIndex: 0 }}
            >
                <source src={`${process.env.NODE_ENV === 'production' ? '/ring-dynamics' : ''}/demo.mp4`} type="video/mp4" />
            </video>

            {/* Dark overlay to dim the video */}
            <div
                className="absolute inset-0"
                style={{
                    zIndex: 1,
                    background: 'linear-gradient(180deg, rgba(10,10,15,0.92) 0%, rgba(10,10,15,0.80) 40%, rgba(10,10,15,0.92) 100%)',
                }}
            />

            {/* ── Foreground Content ───────────────────────────── */}
            <div className="relative" style={{ zIndex: 2 }}>

                {/* Nav */}
                <nav className="flex items-center justify-between px-6 py-3"
                    style={{ borderBottom: '1px solid rgba(255,255,255,0.08)' }}>
                    <h1 className="text-lg font-bold tracking-wider uppercase" style={{ color: '#e53e3e' }}>
                        🥊 Ring Dynamics
                    </h1>
                    <span className={`text-xs ${apiStatus === 'online' ? 'text-green-400' :
                        apiStatus === 'offline' ? 'text-red-400' : 'text-gray-500'
                        }`}>
                        {apiStatus === 'online' ? 'API ✓' :
                            apiStatus === 'offline' ? 'API ✕ (backend offline)' : 'API ...'}
                    </span>
                </nav>

                <div className="container mx-auto px-4 py-16">
                    <div className="text-center mb-12">
                        <h1 className="text-7xl font-black mb-4 pb-2 leading-tight tracking-tight"
                            style={{
                                background: 'linear-gradient(135deg, #e53e3e 0%, #ff6b6b 50%, #e53e3e 100%)',
                                WebkitBackgroundClip: 'text',
                                WebkitTextFillColor: 'transparent',
                                textShadow: 'none',
                            }}>
                            Ring Dynamics
                        </h1>
                        <p className="text-xl text-gray-300 max-w-2xl mx-auto" style={{ lineHeight: '1.6' }}>
                            Computational physics meets real-time biomechanics — quantifying every punch, step, and shift through mathematical modeling and statistical inference.
                        </p>
                    </div>

                    <div className="max-w-4xl mx-auto grid md:grid-cols-2 gap-8">
                        {/* Video Upload */}
                        <div className="rounded-2xl p-8"
                            style={{
                                background: 'rgba(229, 62, 62, 0.08)',
                                backdropFilter: 'blur(20px)',
                                border: '1px solid rgba(229, 62, 62, 0.2)',
                            }}>
                            <h2 className="text-2xl font-bold mb-6 fighter-a-color">Upload Video</h2>
                            <form onSubmit={handleFileUpload} className="space-y-4">
                                <div className="border-2 border-dashed rounded-lg p-8 text-center hover:border-red-600 transition-colors"
                                    style={{ borderColor: 'rgba(229, 62, 62, 0.25)' }}>
                                    <input
                                        type="file"
                                        accept="video/*"
                                        onChange={(e) => setFile(e.target.files?.[0] || null)}
                                        style={{ position: 'absolute', width: '1px', height: '1px', opacity: 0, overflow: 'hidden' }}
                                        id="file-upload"
                                    />
                                    <label htmlFor="file-upload" className="cursor-pointer">
                                        <div className="text-4xl mb-2">🎥</div>
                                        <div className="text-sm text-gray-300">
                                            {file ? (
                                                <span className="text-red-400 font-semibold">{file.name}</span>
                                            ) : (
                                                'Click to select video'
                                            )}
                                        </div>
                                        <div className="text-xs text-gray-500 mt-2">
                                            MP4, AVI, MOV (max {API_URL.includes('localhost') ? '500' : '95'}MB)
                                        </div>
                                    </label>
                                </div>
                                <button
                                    type="submit"
                                    disabled={!file || uploading}
                                    className="w-full text-white font-semibold py-3 rounded-lg transition-all disabled:opacity-40 disabled:cursor-not-allowed"
                                    style={{ background: !file || uploading ? 'rgba(50,50,50,0.6)' : 'linear-gradient(135deg, #e53e3e, #c53030)' }}
                                >
                                    {uploading ? uploadProgress || 'Uploading...' : 'Upload & Analyze'}
                                </button>
                            </form>
                        </div>

                        {/* YouTube URL */}
                        <div className="rounded-2xl p-8"
                            style={{
                                background: 'rgba(255, 255, 255, 0.04)',
                                backdropFilter: 'blur(20px)',
                                border: '1px solid rgba(255, 255, 255, 0.08)',
                            }}>
                            <h2 className="text-2xl font-bold mb-6 fighter-b-color">YouTube URL</h2>
                            <form onSubmit={handleYoutubeSubmit} className="space-y-4">
                                <div>
                                    <input
                                        type="url"
                                        placeholder="https://youtube.com/watch?v=..."
                                        value={youtubeUrl}
                                        onChange={(e) => setYoutubeUrl(e.target.value)}
                                        className="w-full rounded-lg px-4 py-3 focus:outline-none transition-colors"
                                        style={{
                                            background: 'rgba(17,17,24,0.7)',
                                            border: '1px solid rgba(255,255,255,0.08)',
                                        }}
                                    />
                                </div>
                                <button
                                    type="submit"
                                    disabled={!youtubeUrl || uploading}
                                    className="w-full text-white font-semibold py-3 rounded-lg transition-all disabled:opacity-40 disabled:cursor-not-allowed"
                                    style={{ background: !youtubeUrl || uploading ? 'rgba(50,50,50,0.6)' : 'linear-gradient(135deg, #4299e1, #3182ce)' }}
                                >
                                    {uploading ? 'Processing...' : 'Process YouTube Video'}
                                </button>
                            </form>
                            <div className="mt-6 text-xs text-gray-400 space-y-1">
                                <p>✓ Automatically downloads in best quality</p>
                                <p>✓ YOLOv8 + ByteTrack fighter detection</p>
                                <p>✓ 3-box tracking: body, head, core</p>
                            </div>
                        </div>
                    </div>

                    {/* Features */}
                    <div className="mt-16 grid md:grid-cols-4 gap-6 text-center">
                        {[
                            { icon: '🥊', title: 'Fighter Detection', desc: 'YOLOv8 + anti-audience filtering' },
                            { icon: '🎯', title: '3-Box Tracking', desc: 'Full body, head, and core zones' },
                            { icon: '📊', title: 'Live Scoring', desc: 'Real-time activity & aggression metrics' },
                            { icon: '📹', title: 'Camera-Cut Proof', desc: 'Auto-recovers after angle changes' },
                        ].map((f, i) => (
                            <div key={i} className="rounded-xl p-6"
                                style={{
                                    background: 'rgba(255, 255, 255, 0.04)',
                                    backdropFilter: 'blur(12px)',
                                    border: '1px solid rgba(255, 255, 255, 0.06)',
                                }}>
                                <div className="text-3xl mb-2">{f.icon}</div>
                                <h3 className="font-semibold mb-2">{f.title}</h3>
                                <p className="text-sm text-gray-400">{f.desc}</p>
                            </div>
                        ))}
                    </div>
                </div>
            </div>
        </main>
    );
}
