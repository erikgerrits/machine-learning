import { NavLink, Link, Outlet } from 'react-router-dom';
import { ALGORITHMS } from '../algorithms';
import styles from './Layout.module.css';

export function Layout() {
    return (
        <div className={styles.shell}>
            <header className={styles.header}>
                <Link to="/" className={styles.brand}>
                    <span className={styles.logo}>◍</span>
                    <span>machine-learning</span>
                </Link>
                <a
                    className={styles.gh}
                    href="https://github.com/erikgerrits/machine-learning"
                    target="_blank"
                    rel="noreferrer"
                >
                    GitHub ↗
                </a>
            </header>

            <div className={styles.body}>
                <nav className={styles.sidebar}>
                    <span className={styles.navHeading}>Algorithms</span>
                    {ALGORITHMS.map(algo =>
                        algo.status === 'live' ? (
                            <NavLink
                                key={algo.id}
                                to={algo.path}
                                className={({ isActive }) =>
                                    `${styles.navLink} ${isActive ? styles.active : ''}`
                                }
                            >
                                {algo.title}
                            </NavLink>
                        ) : (
                            <span key={algo.id} className={`${styles.navLink} ${styles.soon}`}>
                                {algo.title}
                                <span className={styles.soonTag}>soon</span>
                            </span>
                        ),
                    )}
                </nav>

                <main className={styles.content}>
                    <Outlet />
                </main>
            </div>
        </div>
    );
}
