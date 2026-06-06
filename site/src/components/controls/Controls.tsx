import type { ReactNode } from 'react';
import styles from './Controls.module.css';

// Shared, presentational building blocks for every algorithm playground. Each playground
// composes these (transport, sliders, selects, metrics, cards) around its own visualization,
// so adding a new algorithm is mostly "pick controls + draw your model".

export function ControlPanel({ children }: { children: ReactNode }) {
    return <aside className={styles.panel}>{children}</aside>;
}

export function RunControls({
    running,
    onToggle,
    onStep,
    onReset,
}: {
    running: boolean;
    onToggle: () => void;
    onStep: () => void;
    onReset: () => void;
}) {
    return (
        <div className={styles.transport}>
            <button className={`${styles.btn} ${styles.primary}`} onClick={onToggle}>
                {running ? '❚❚ Pause' : '▶ Train'}
            </button>
            <button className={styles.btn} onClick={onStep} disabled={running}>
                Step
            </button>
            <button className={styles.btn} onClick={onReset}>
                Reset
            </button>
        </div>
    );
}

export function Field({ label, value, children }: { label: string; value?: string; children: ReactNode }) {
    return (
        <label className={styles.field}>
            <span>
                {label}
                {value !== undefined && <em>{value}</em>}
            </span>
            {children}
        </label>
    );
}

export function Slider({
    label,
    value,
    display,
    min,
    max,
    step = 1,
    onChange,
}: {
    label: string;
    value: number;
    display?: string;
    min: number;
    max: number;
    step?: number;
    onChange: (value: number) => void;
}) {
    return (
        <Field label={label} value={display}>
            <input
                type="range"
                min={min}
                max={max}
                step={step}
                value={value}
                onChange={e => onChange(Number(e.target.value))}
            />
        </Field>
    );
}

export interface SelectOption {
    value: string;
    label: string;
}

export function Select({
    label,
    value,
    options,
    onChange,
}: {
    label: string;
    value: string;
    options: SelectOption[];
    onChange: (value: string) => void;
}) {
    return (
        <Field label={label}>
            <select value={value} onChange={e => onChange(e.target.value)}>
                {options.map(option => (
                    <option key={option.value} value={option.value}>
                        {option.label}
                    </option>
                ))}
            </select>
        </Field>
    );
}

export function NumberField({
    label,
    value,
    onChange,
}: {
    label: string;
    value: number;
    onChange: (value: number) => void;
}) {
    return (
        <Field label={label}>
            <input type="number" value={value} onChange={e => onChange(Number(e.target.value))} />
        </Field>
    );
}

export function Hint({ children }: { children: ReactNode }) {
    return <p className={styles.hint}>{children}</p>;
}

export function MetricsRow({ children }: { children: ReactNode }) {
    return <div className={styles.metrics}>{children}</div>;
}

export function Metric({ label, value }: { label: string; value: string }) {
    return (
        <div className={styles.metric}>
            <span className={styles.metricLabel}>{label}</span>
            <span className={styles.metricValue}>{value}</span>
        </div>
    );
}

export function Badge({ children }: { children: ReactNode }) {
    return <div className={styles.badge}>{children}</div>;
}

export function Card({ title, subtitle, children }: { title: string; subtitle?: string; children: ReactNode }) {
    return (
        <div className={styles.card}>
            <h3>
                {title}
                {subtitle && <span>{subtitle}</span>}
            </h3>
            {children}
        </div>
    );
}
