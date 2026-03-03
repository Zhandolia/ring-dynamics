'use client';

import { useState } from 'react';

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

export default function Home() {
    const [file, setFile] = useState<File | null>(null);
    const [youtubeUrl, setYoutubeUrl] = useState('');
    const [uploading, setUploading] = useState(false);
    const [uploadProgress, setUploadProgress] = useState('');

    const navigateToFight = (id: string) => {
        const base = process.env.NODE_ENV === 'production' ? '/ring-dynamics' : '';
        window.location.href = `${base}/fight/#${id}`;
    };

    const handleFileUpload = async (e: React.FormEvent) => {
        e.preventDefault();
        if (!file) return;

        setUploading(true);
        setUploadProgress('Uploading video...');
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
            alert(`Upload failed: ${message}\n\nMake sure the backend is running:\ncd backend && uvicorn app.main:app --reload --port 8000`);
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
        <main className="min-h-screen bg-[#0a0a0f] text-white">
            {/* Nav */}
            <nav className="flex items-center justify-between px-6 py-3 border-b border-[#1f1f2e]">
                <h1 className="text-lg font-bold tracking-wider uppercase" style={{ color: '#e53e3e' }}>
                    🥊 Ring Dynamics
                </h1>
                <span className="text-green-400 text-xs">API ✓</span>
            </nav>

            <div className="container mx-auto px-4 py-16">
                <div className="text-center mb-12">
                    <h1 className="text-6xl font-bold mb-4 pb-2 leading-normal"
                        style={{ color: '#e53e3e' }}>
                        Ring Dynamics
                    </h1>
                    <p className="text-lg text-gray-500">
                        AI-powered boxing analysis with 3-box fighter tracking & live scoring
                    </p>
                </div>

                <div className="max-w-4xl mx-auto grid md:grid-cols-2 gap-8">
                    {/* Video Upload */}
                    <div className="glass-red rounded-2xl p-8">
                        <h2 className="text-2xl font-bold mb-6 fighter-a-color">Upload Video</h2>
                        <form onSubmit={handleFileUpload} className="space-y-4">
                            <div className="border-2 border-dashed border-[#2a1a1a] rounded-lg p-8 text-center hover:border-red-700 transition-colors">
                                <input
                                    type="file"
                                    accept="video/*"
                                    onChange={(e) => setFile(e.target.files?.[0] || null)}
                                    style={{ position: 'absolute', width: '1px', height: '1px', opacity: 0, overflow: 'hidden' }}
                                    id="file-upload"
                                />
                                <label htmlFor="file-upload" className="cursor-pointer">
                                    <div className="text-4xl mb-2">🎥</div>
                                    <div className="text-sm text-gray-400">
                                        {file ? (
                                            <span className="text-red-400 font-semibold">{file.name}</span>
                                        ) : (
                                            'Click to select video'
                                        )}
                                    </div>
                                    <div className="text-xs text-gray-600 mt-2">
                                        MP4, AVI, MOV (max 500MB)
                                    </div>
                                </label>
                            </div>
                            <button
                                type="submit"
                                disabled={!file || uploading}
                                className="w-full text-white font-semibold py-3 rounded-lg transition-all disabled:opacity-40 disabled:cursor-not-allowed"
                                style={{ background: !file || uploading ? '#333' : 'linear-gradient(135deg, #e53e3e, #c53030)' }}
                            >
                                {uploading ? uploadProgress || 'Uploading...' : 'Upload & Analyze'}
                            </button>
                        </form>
                    </div>

                    {/* YouTube URL */}
                    <div className="glass rounded-2xl p-8">
                        <h2 className="text-2xl font-bold mb-6 fighter-b-color">YouTube URL</h2>
                        <form onSubmit={handleYoutubeSubmit} className="space-y-4">
                            <div>
                                <input
                                    type="url"
                                    placeholder="https://youtube.com/watch?v=..."
                                    value={youtubeUrl}
                                    onChange={(e) => setYoutubeUrl(e.target.value)}
                                    className="w-full bg-[#111118] border border-[#1f1f2e] rounded-lg px-4 py-3 focus:outline-none focus:border-blue-500 transition-colors"
                                />
                            </div>
                            <button
                                type="submit"
                                disabled={!youtubeUrl || uploading}
                                className="w-full text-white font-semibold py-3 rounded-lg transition-all disabled:opacity-40 disabled:cursor-not-allowed"
                                style={{ background: !youtubeUrl || uploading ? '#333' : 'linear-gradient(135deg, #4299e1, #3182ce)' }}
                            >
                                {uploading ? 'Processing...' : 'Process YouTube Video'}
                            </button>
                        </form>
                        <div className="mt-6 text-xs text-gray-600 space-y-1">
                            <p>✓ Automatically downloads in best quality</p>
                            <p>✓ YOLOv8 + ByteTrack fighter detection</p>
                            <p>✓ 3-box tracking: body, head, core</p>
                        </div>
                    </div>
                </div>

                {/* Features */}
                <div className="mt-16 grid md:grid-cols-4 gap-6 text-center">
                    <div className="glass rounded-xl p-6">
                        <div className="text-3xl mb-2">🥊</div>
                        <h3 className="font-semibold mb-2">Fighter Detection</h3>
                        <p className="text-sm text-gray-500">YOLOv8 + anti-audience filtering</p>
                    </div>
                    <div className="glass rounded-xl p-6">
                        <div className="text-3xl mb-2">🎯</div>
                        <h3 className="font-semibold mb-2">3-Box Tracking</h3>
                        <p className="text-sm text-gray-500">Full body, head, and core zones</p>
                    </div>
                    <div className="glass rounded-xl p-6">
                        <div className="text-3xl mb-2">📊</div>
                        <h3 className="font-semibold mb-2">Live Scoring</h3>
                        <p className="text-sm text-gray-500">Real-time activity & aggression metrics</p>
                    </div>
                    <div className="glass rounded-xl p-6">
                        <div className="text-3xl mb-2">📹</div>
                        <h3 className="font-semibold mb-2">Camera-Cut Proof</h3>
                        <p className="text-sm text-gray-500">Auto-recovers after angle changes</p>
                    </div>
                </div>
            </div>
        </main>
    );
}
