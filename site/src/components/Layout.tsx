import { Fragment, Suspense } from 'react';
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
                    <span className={styles.navHeading}>The Drifting Leaf · ML course</span>
                    {ALGORITHMS.map((algo, i) => {
                        const startsPart = algo.part && algo.part !== ALGORITHMS[i - 1]?.part;
                        const label = algo.chapter ? `Ch ${algo.chapter} · ${algo.title}` : algo.title;
                        return (
                            <Fragment key={algo.id}>
                                {startsPart && <span className={styles.partHeading}>{algo.part}</span>}
                                {algo.status === 'live' ? (
                                    <NavLink
                                        to={algo.path}
                                        className={({ isActive }) =>
                                            `${styles.navLink} ${isActive ? styles.active : ''}`
                                        }
                                    >
                                        {label}
                                    </NavLink>
                                ) : (
                                    <span className={`${styles.navLink} ${styles.soon}`}>
                                        {label}
                                        <span className={styles.soonTag}>soon</span>
                                    </span>
                                )}
                            </Fragment>
                        );
                    })}
                </nav>

                <main className={styles.content}>
                    <Suspense fallback={<div className={styles.loading}>Loading…</div>}>
                        <Outlet />
                    </Suspense>
                </main>
            </div>
        </div>
    );
}
