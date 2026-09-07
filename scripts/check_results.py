#!/usr/bin/env python3
"""Validate recorded request coverage and recompute published medians offline."""
import json
from pathlib import Path
from statistics import median

def main():
    d = json.loads((Path(__file__).resolve().parents[1]/'results/vision-quant-comparison.json').read_text())
    expected = {'Q2-distributed', 'Q3-distributed', 'Q4-distributed', 'Q2-single'}
    assert {v['name'] for v in d['variants']} == expected and len(d['variants']) == 4
    assert len(d['summary']) == 12
    assert {(s['variant'], s['workload']) for s in d['summary']} == {
        (v, w) for v in expected for w in ['cold32k', 'warm32k', 'concurrent4x8k']
    }
    total = 0
    for v in d['variants']:
        assert len(v['groups']) == 9
        assert {(g['workload'],g['repeat']) for g in v['groups']} == {(w,r) for w in ['cold32k','warm32k','concurrent4x8k'] for r in range(3)}
        for g in v['groups']:
            concurrent = g['workload'] == 'concurrent4x8k'
            assert len(g['requests']) == (4 if concurrent else 1)
            assert abs(g['completion_tps']-len(g['requests'])*256/g['wall_s']) < 1e-8
            for r in g['requests']:
                total += 1
                assert r['tokens_predicted'] == r['timings']['predicted_n'] == 256
                assert r['input_tokens'] == (8192 if concurrent else 32768)
                assert r['timings']['cache_n'] == (32764 if g['workload']=='warm32k' else 0)
                assert r['timings']['prompt_n'] == (4 if g['workload']=='warm32k' else r['input_tokens'])
        for summary in [s for s in d['summary'] if s['variant']==v['name']]:
            groups = [g for g in v['groups'] if g['workload']==summary['workload']]
            requests = [r for g in groups for r in g['requests']]
            for key, actual in [('median_group_completion_tps',median(g['completion_tps'] for g in groups)),('median_prompt_tps',median(r['timings']['prompt_per_second'] for r in requests)),('median_decode_tps',median(r['timings']['predicted_per_second'] for r in requests))]:
                assert abs(summary[key]-actual)<1e-8, (v['name'],key)
    assert total == 72
    assert len(d['summary']) == 12
    print('Verified 72 requests, 36 groups, 12 summaries; cache/output counts and medians match.')

if __name__ == '__main__': main()
