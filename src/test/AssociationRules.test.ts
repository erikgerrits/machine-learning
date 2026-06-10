import { describe, it, expect } from 'vitest';
import AssociationRules from '../lib/machine-learning/unsupervised/AssociationRules';
import Matrix from '../lib/math/linear-algebra/Matrix';
import { ASSOCIATION_INPUTS } from './helpers/fixtures';

const ruleKey = (a: number[], c: number[]) => `${a.join(',')}=>${c.join(',')}`;

describe('AssociationRules', () => {

    it('finds the confident "1 → 0" rule', () => {
        const model = new AssociationRules().setMinSupport(0.5).setMinConfidence(0.6);
        model.train(new Matrix(ASSOCIATION_INPUTS));

        const rules = model.getRules();
        const oneToZero = rules.find(r => ruleKey(r.antecedent, r.consequent) === '1=>0');
        expect(oneToZero).toBeDefined();
        expect(oneToZero!.confidence).toBeCloseTo(1, 6);    // every basket with item 1 also has item 0
        expect(oneToZero!.support).toBeCloseTo(4 / 6, 6);   // {0,1} in 4 of 6 baskets
        expect(oneToZero!.lift).toBeCloseTo(1.2, 6);        // 1.0 / (5/6)
    });

    it('records {0,1} as a frequent itemset with the right support', () => {
        const model = new AssociationRules().setMinSupport(0.5).setMinConfidence(0.6);
        model.train(new Matrix(ASSOCIATION_INPUTS));

        const itemset = model.getFrequentItemsets().find(s => s.items.join(',') === '0,1');
        expect(itemset).toBeDefined();
        expect(itemset!.support).toBeCloseTo(4 / 6, 6);
    });

    it('drops the weaker direction as the confidence bar rises', () => {
        const model = new AssociationRules().setMinSupport(0.5).setMinConfidence(0.9);
        model.train(new Matrix(ASSOCIATION_INPUTS));

        const keys = model.getRules().map(r => ruleKey(r.antecedent, r.consequent));
        expect(keys).toContain('1=>0');     // confidence 1.0 survives
        expect(keys).not.toContain('0=>1'); // confidence 0.8 does not
    });

    it('finds fewer itemsets as the support bar rises', () => {
        const count = (minSupport: number) =>
            new AssociationRules().setMinSupport(minSupport).train(new Matrix(ASSOCIATION_INPUTS)).getFrequentItemsets().length;
        expect(count(0.9)).toBeLessThan(count(0.3));
    });

    it('completes a basket: given item 1, suggests item 0', () => {
        const model = new AssociationRules().setMinSupport(0.5).setMinConfidence(0.6);
        model.train(new Matrix(ASSOCIATION_INPUTS));

        const scores = model.predict(new Matrix([[0, 1, 0, 0]])).toArray()[0];
        expect(scores[0]).toBeCloseTo(1, 6); // item 0 recommended with confidence 1.0
        expect(scores[1]).toBe(0);           // item 1 is already in the basket
    });

    it('round-trips its hyperparameters and chains setters', () => {
        const model = new AssociationRules();
        expect(model.getMinSupport()).toBe(0.1);
        expect(model.getMinConfidence()).toBe(0.5);

        const returned = model.setMinSupport(0.2).setMinConfidence(0.7).setMaxItemsetSize(2);
        expect(returned).toBe(model);
        expect(model.getMinSupport()).toBe(0.2);
        expect(model.getMinConfidence()).toBe(0.7);
        expect(model.getMaxItemsetSize()).toBe(2);
    });
});
