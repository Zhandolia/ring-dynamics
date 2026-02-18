'use client';

import { useState } from 'react';

export default function Home() {
    const [file, setFile] = useState<File | null>(null);
    const [youtubeUrl, setYoutubeUrl] = useState('');
    const [uploading, setUploading] = useState(false);

    const handleFileUpload = async (e: React.FormEvent) => {
        e.preventDefault();
        if (!file) return;

        setUploading(true);
        const formData = new FormData();
        formData.append('file', file);

        try {
            const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/api/fights/upload`, {
                method: 'POST',
                body: formData,
            });
            const data = await response.json();
            console.log('Upload successful:', data);
            alert(`Fight uploaded! ID: ${data.id}`);
        } catch (error) {
            console.error('Upload failed:', error);
            alert('Upload failed. Is the backend running?');
        } finally {
            setUploading(false);
        }
    };

    const handleYoutubeSubmit = async (e: React.FormEvent) => {
        e.preventDefault();
        if (!youtubeUrl) return;

        setUploading(true);
        try {
            const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/api/fights/youtube`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ youtube_url: youtubeUrl }),
            });
            const data = await response.json();
            console.log('YouTube processing started:', data);
            alert(`Fight queued! ID: ${data.id}`);
        } catch (error) {
            console.error('Failed:', error);
            alert('Failed. Is the backend running?');
        } finally {
            setUploading(false);
        }
    };

    return (
        <main className="min-h-screen bg-gradient-to-b from-gray-900 to-black text-white">
            <div className="container mx-auto px-4 py-16">
                <div className="text-center mb-12">
                    <h1 className="text-6xl font-bold mb-4 bg-gradient-to-r from-blue-400 to-red-400 bg-clip-text text-transparent">
                        Ring Dynamics
                    </h1>
                    <p className="text-xl text-gray-400">
                        Production-grade boxing analytics with computer vision and Bayesian scoring
                    </p>
                </div>

                <div className="max-w-4xl mx-auto grid md:grid-cols-2 gap-8">
                    {/* Video Upload */}
                    <div className="glass rounded-2xl p-8">
                        <h2 className="text-2xl font-bold mb-6 fighter-0-color">Upload Video</h2>
                        <form onSubmit={handleFileUpload} className="space-y-4">
                            <div className="border-2 border-dashed border-gray-600 rounded-lg p-8 text-center hover:border-blue-500 transition-colors">
                                <input
                                    type="file"
                                    accept="video/*"
                                    onChange={(e) => setFile(e.target.files?.[0] || null)}
                                    className="hidden"
                                    id="file-upload"
                                />
                                <label htmlFor="file-upload" className="cursor-pointer">
                                    <div className="text-4xl mb-2">🎥</div>
                                    <div className="text-sm text-gray-400">
                                        {file ? file.name : 'Click to select video'}
                                    </div>
                                    <div className="text-xs text-gray-500 mt-2">
                                        MP4, AVI, MOV (max 500MB)
                                    </div>
                                </label>
                            </div>
                            <button
                                type="submit"
                                disabled={!file || uploading}
                                className="w-full bg-fighter-0 hover:bg-blue-600 disabled:bg-gray-600 disabled:cursor-not-allowed text-white font-semibold py-3 rounded-lg transition-colors"
                            >
                                {uploading ? 'Uploading...' : 'Upload & Analyze'}
                            </button>
                        </form>
                    </div>

                    {/* YouTube URL */}
                    <div className="glass rounded-2xl p-8">
                        <h2 className="text-2xl font-bold mb-6 fighter-1-color">YouTube URL</h2>
                        <form onSubmit={handleYoutubeSubmit} className="space-y-4">
                            <div>
                                <input
                                    type="url"
                                    placeholder="https://youtube.com/watch?v=..."
                                    value={youtubeUrl}
                                    onChange={(e) => setYoutubeUrl(e.target.value)}
                                    className="w-full bg-gray-800 border border-gray-600 rounded-lg px-4 py-3 focus:outline-none focus:border-red-500 transition-colors"
                                />
                            </div>
                            <button
                                type="submit"
                                disabled={!youtubeUrl || uploading}
                                className="w-full bg-fighter-1 hover:bg-red-600 disabled:bg-gray-600 disabled:cursor-not-allowed text-white font-semibold py-3 rounded-lg transition-colors"
                            >
                                {uploading ? 'Processing...' : 'Process YouTube Video'}
                            </button>
                        </form>
                        <div className="mt-6 text-xs text-gray-500">
                            <p>✓ Automatically downloads in best quality</p>
                            <p>✓ Supports full fight broadcasts</p>
                            <p>✓ Real-time analysis updates</p>
                        </div>
                    </div>
                </div>

                {/* Features */}
                <div className="mt-16 grid md:grid-cols-4 gap-6 text-center">
                    <div className="glass rounded-xl p-6">
                        <div className="text-3xl mb-2">🥊</div>
                        <h3 className="font-semibold mb-2">Multi-Object Detection</h3>
                        <p className="text-sm text-gray-400">YOLO + ByteTrack tracking</p>
                    </div>
                    <div className="glass rounded-xl p-6">
                        <div className="text-3xl mb-2">🎯</div>
                        <h3 className="font-semibold mb-2">Punch Classification</h3>
                        <p className="text-sm text-gray-400">Jab, cross, hook, uppercut</p>
                    </div>
                    <div className="glass rounded-xl p-6">
                        <div className="text-3xl mb-2">📊</div>
                        <h3 className="font-semibold mb-2">Bayesian Scoring</h3>
                        <p className="text-sm text-gray-400">Probabilistic round scores</p>
                    </div>
                    <div className="glass rounded-xl p-6">
                        <div className="text-3xl mb-2">📈</div>
                        <h3 className="font-semibold mb-2">Win Probability</h3>
                        <p className="text-sm text-gray-400">Monte Carlo simulation</p>
                    </div>
                </div>

                {/* Status */}
                <div className="mt-12 text-center text-sm text-gray-500">
                    <p>Backend API: {process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'}</p>
                </div>
            </div>
        </main>
    );
}
