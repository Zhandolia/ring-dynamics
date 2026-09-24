export async function readSubmission(response: Response): Promise<{ id: string }> {
    const data = await response.json().catch(() => null);
    if (!response.ok) {
        const detail = typeof data?.detail === 'string' ? data.detail : null;
        throw new Error(detail || `The server could not accept the video (HTTP ${response.status}). Please try again.`);
    }
    if (typeof data?.id !== 'string' || !/^[0-9a-f]{8}-[0-9a-f-]{27}$/i.test(data.id)) {
        throw new Error('The server returned an invalid analysis ID. Please try again.');
    }
    return data;
}
