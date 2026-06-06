import { Link } from 'react-router-dom';
import { ALGORITHMS } from '../algorithms';
import { HeroCanvas } from '../components/HeroCanvas';
import styles from './Landing.module.css';

export function Landing() {
    return (
        <div className={styles.landing}>
            <section className={styles.hero}>
                <div className={styles.heroText}>
                    <h1>Machine learning you can watch.</h1>
                    <p>
                        A small, dependency-free TypeScript library — and a playground where every
                        algorithm trains <strong>live in your browser</strong>. Tune the knobs, press
                        play, and watch the math learn.
                    </p>
                    <div className={styles.actions}>
                        <Link to="/neural-network" className={styles.cta}>
                            Open the playground →
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
                <h2 className={styles.sectionTitle}>Explore the algorithms</h2>
                <div className={styles.cards}>
                    {ALGORITHMS.map(algo => {
                        const card = (
                            <>
                                <div className={styles.cardHead}>
                                    <span className={styles.cardTitle}>{algo.title}</span>
                                    {algo.status === 'soon' && <span className={styles.tag}>soon</span>}
                                </div>
                                <p className={styles.cardTagline}>{algo.tagline}</p>
                            </>
                        );
                        return algo.status === 'live' ? (
                            <Link key={algo.id} to={algo.path} className={`${styles.card} ${styles.live}`}>
                                {card}
                            </Link>
                        ) : (
                            <div key={algo.id} className={styles.card}>{card}</div>
                        );
                    })}
                </div>
            </section>
        </div>
    );
}
