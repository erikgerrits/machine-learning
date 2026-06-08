import { Link } from 'react-router-dom';
import { ALGORITHMS, type AlgorithmEntry } from '../algorithms';
import { HeroCanvas } from '../components/HeroCanvas';
import styles from './Landing.module.css';

// Group the chapters by part, preserving course order, so the landing reads as a table of contents.
function groupByPart(entries: AlgorithmEntry[]): { part: string; items: AlgorithmEntry[] }[] {
    const groups: { part: string; items: AlgorithmEntry[] }[] = [];
    for (const entry of entries) {
        const part = entry.part ?? 'More';
        let group = groups.find(g => g.part === part);
        if (!group) {
            group = { part, items: [] };
            groups.push(group);
        }
        group.items.push(entry);
    }
    return groups;
}

export function Landing() {
    const parts = groupByPart(ALGORITHMS);

    return (
        <div className={styles.landing}>
            <section className={styles.hero}>
                <div className={styles.heroText}>
                    <h1>Learn machine learning by saving a café.</h1>
                    <p>
                        Nadia inherits a failing café and a shoebox of receipts. Each chapter, a new
                        problem forces her to invent a new method — and you watch the{' '}
                        <strong>real, dependency-free TypeScript library</strong> learn it{' '}
                        <strong>live in your browser</strong>.
                    </p>
                    <div className={styles.actions}>
                        <Link to="/the-ledger" className={styles.cta}>
                            Start the story →
                        </Link>
                        <a
                            className={styles.secondary}
                            href="https://github.com/erikgerrits/machine-learning"
                            target="_blank"
                            rel="noreferrer"
                        >
                            GitHub
                        </a>
                    </div>
                    <code className={styles.install}>yarn add machine-learning</code>
                </div>
                <div className={styles.heroArt}>
                    <HeroCanvas />
                </div>
            </section>

            <section>
                <h2 className={styles.sectionTitle}>The course</h2>
                <p className={styles.sectionLead}>
                    One café, growing chapter by chapter — each new problem cornering Nadia into a
                    new machine-learning method. Numbering gaps are chapters still to come.
                </p>
                {parts.map(group => (
                    <div key={group.part} className={styles.partBlock}>
                        <h3 className={styles.partTitle}>{group.part}</h3>
                        <div className={styles.cards}>
                            {group.items.map(algo => {
                                const badge = algo.chapter != null ? `Ch ${algo.chapter}` : 'Prologue';
                                const inner = (
                                    <>
                                        <div className={styles.cardHead}>
                                            <span className={styles.cardBadge}>{badge}</span>
                                            {algo.status === 'soon' && (
                                                <span className={styles.tag}>soon</span>
                                            )}
                                        </div>
                                        <span className={styles.cardTitle}>{algo.title}</span>
                                        <p className={styles.cardTagline}>{algo.tagline}</p>
                                    </>
                                );
                                return algo.status === 'live' ? (
                                    <Link
                                        key={algo.id}
                                        to={algo.path}
                                        className={`${styles.card} ${styles.live}`}
                                    >
                                        {inner}
                                    </Link>
                                ) : (
                                    <div key={algo.id} className={styles.card}>
                                        {inner}
                                    </div>
                                );
                            })}
                        </div>
                    </div>
                ))}
            </section>
        </div>
    );
}
