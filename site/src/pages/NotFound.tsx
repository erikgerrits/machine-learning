import { Link } from 'react-router-dom';

export function NotFound() {
    return (
        <div style={{ textAlign: 'center', padding: '80px 20px' }}>
            <h1 style={{ fontSize: 48, margin: 0 }}>404</h1>
            <p style={{ color: 'var(--muted)' }}>That page wandered off into feature space.</p>
            <Link to="/">← Back home</Link>
        </div>
    );
}
