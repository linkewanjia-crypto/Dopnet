#!/usr/bin/env python3
"""Small synthetic checks of reporting arithmetic; no experiment is executed."""
import numpy as np
import report_classical as report


def main():
    rng = np.random.default_rng(42)
    cluster = np.arange(512) // 8
    ids, weights = report.resampling_weights(cluster, 2000, rng)
    x = np.vstack([np.ones(512), np.full(512, 2.0), np.full(512, 3.0)])
    point, boot = report.metric_samples(x, None, cluster, weights, ids)
    np.testing.assert_array_equal(point, [1.0, 2.0, 3.0])
    np.testing.assert_allclose(boot, np.tile(point, (2000, 1)))
    seed_draws = rng.integers(0, 3, size=(2000, 3))
    row, draws = report.interval_row({}, 'dopnet', 'nmse', point, boot, seed_draws)
    assert row['n_training_seeds'] == 3 and row['mean'] == 2.0
    assert row['ci95_lo'] < 2 < row['ci95_hi']
    np.testing.assert_allclose(row['mean_db'], 10 * np.log10(2.0))
    cp, cb = report.metric_samples(np.full((1, 512), 2.0), None, cluster, weights, ids)
    classical, _ = report.interval_row({}, 'oce_bem_joint_ridge', 'nmse', cp, cb, seed_draws)
    assert classical['n_training_seeds'] == 0
    assert classical['ci95_lo'] == classical['ci95_hi'] == 2.0
    paired = report.paired_row({}, 'oce_bem_joint_ridge', 'nmse', cp, cb, point, boot, seed_draws)
    assert paired['paired_difference'] == 0
    assert paired['ci95_lo'] < 0 < paired['ci95_hi']
    # A variable number of bits requires a ratio of totals, not mean frame BER.
    bits = np.where(np.arange(512) < 256, 10, 100)[None, :]
    errors = np.where(np.arange(512) < 256, 1, 20)[None, :]
    p, b = report.metric_samples(errors, bits, cluster, weights, ids)
    np.testing.assert_allclose(p, 21.0 / 110.0)
    assert not np.isclose(p[0], np.mean(errors / bits))
    assert np.isfinite(b).all()
    # A paired constant offset is not obscured by independent resampling.
    dp, db = report.metric_samples(x + 0.5, None, cluster, weights, ids)
    d = report.paired_row({}, 'slow_time', 'nmse', dp, db, point, boot, seed_draws)
    np.testing.assert_allclose([d['paired_difference'], d['ci95_lo'], d['ci95_hi']], .5)
    try:
        report.resampling_weights(np.arange(512), 2000, rng)
        # Equal-size single-frame clusters are mathematically permissible here;
        # the original-eight-frame requirement is enforced in report().
    except ValueError:
        raise AssertionError('Unexpected rejection of equal-size clusters')
    try:
        report.metric_samples(np.array([[-1.0] * 512]), None, cluster, weights, ids)
    except ValueError:
        pass
    else:
        raise AssertionError('Negative NMSE was accepted')
    print('PASS: linear NMSE, weighted BER, no fake classical seeds, paired cluster/seed bootstrap')


if __name__ == '__main__':
    main()
