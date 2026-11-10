"""Recompute descriptive tables from the deposited evaluation records."""
import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean, median, quantiles, stdev
from math import asin, cos, isclose, radians, sin, sqrt
import json
import sys

root = Path(__file__).resolve().parent.parent
here = root / 'supplementary_materials'

def read(name):
    if name.startswith('split_membership/'):
        path = root / 'data/splits' / Path(name).name
    elif name == 'run_metrics.csv':
        path = root / 'experiments' / name
    else:
        path = here / name
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))

def near_distance(validation, training):
    x, y = radians(float(validation['lat'])), radians(float(validation['lon']))
    result = 1.0
    for a, b, cos_a in training:
        h = sin((a-x)/2)**2 + cos(x)*cos_a*sin((b-y)/2)**2
        result = min(result, h)
    return 12742.0176*asin(sqrt(result))

def numbers(a, b):
    return all(isclose(float(x), float(y), abs_tol=1e-12, rel_tol=0) for x,y in zip(a,b))

main = ('S0_random_patch', 'S2_spatial_block_1r', 'S3_buffered_1r', 'S4_leave_one_region_out')
models = ('M1','M2','M3')
run_data = read('run_metrics.csv')
assert len(run_data) == len({r['run_id'] for r in run_data}) == 75
runs = defaultdict(list)
by_key = {}
for r in run_data:
    r['validation'] = float(r['validation_miou'])
    r['reference'] = float(r['reference_miou'])
    r['gap'] = r['validation'] - r['reference']
    assert numbers([r['gap'],abs(r['gap'])],[r['optimism_bias'],r['absolute_estimation_error']])
    runs[r['protocol']].append(r)
    by_key[r['run_id']] = r

def rank(rows, metric):
    scores = {m:mean(r[metric] for r in rows if r['model'] == m) for m in models}
    return sorted(scores,key=scores.get,reverse=True), scores

def summarize(rows):
    order_v, v = rank(rows,'validation')
    order_r, g = rank(rows,'reference')
    return dict(n=len(rows), validation=mean(r['validation'] for r in rows),
                reference=mean(r['reference'] for r in rows),
                gap=mean(r['gap'] for r in rows), absolute=mean(abs(r['gap']) for r in rows),
                validation_order=order_v, reference_order=order_r,
                model_validation_scores=v,model_reference_scores=g,
                reference_shortfall=g[order_r[0]]-g[order_v[0]])

summary = json.loads((here/'summary.json').read_text(encoding='utf-8'))
common = sorted(set.intersection(*({r['seed'] for r in runs[p]} for p in main)))
assert common == summary['shared_seeds'] == ['1337','2024','42']
assert [summarize(runs[p]) for p in main] == summary['all_runs']
assert [summarize([r for r in runs[p] if r['seed'] in common]) for p in main] == summary['shared_runs']
seed_orders = []
for p in main:
    seed_orders.append([[seed, rank(group,'validation')[0], rank(group,'reference')[0]]
        for seed in sorted({r['seed'] for r in runs[p]}) if (group := [r for r in runs[p] if r['seed']==seed])])
assert seed_orders == summary['seed_orders']
for name, values in summary['scales'].items():
    assert values == [mean(r['gap'] for r in runs[f'{name}_{width}r'] if r['seed']=='42') for width in ('0.5','1','2')]
assert mean(r['gap'] for r in runs['MATCHED_S3_buffered_1r_N1324_rep0']) == summary['reduced_control_gap']

for r in read('Table_S3_Seed_dispersion.csv'):
    group = [g for g in runs[r['protocol']] if r['model'] == 'all_models' or g['model'] == r['model']]
    values = [mean(g[r['metric']] for g in group if g['seed']==seed) for seed in sorted({g['seed'] for g in group})]
    assert numbers([len(values), mean(values), stdev(values),min(values),max(values)],
                   [r[k] for k in ('n_seed_groups','mean','sample_sd','minimum','maximum')])
scores = defaultdict(list)
for r in read('per_region_metrics.csv'):
    if r['evaluation_set']=='geographic_reference':
        scores[by_key[r['run_id']]['protocol'],r['region']].append(float(r['miou']))
for r in read('Table_S4_Region_dispersion.csv'):
    values = [mean(v) for (p,_),v in scores.items() if p==r['protocol']]
    q1, mid, q3 = quantiles(values,n=4,method='inclusive')
    assert numbers([len(values), min(values), q1, mid, q3, q3-q1, max(values)],
                   [r[k] for k in ('n_regions','minimum','q1','median','q3','iqr','maximum')])

classes = defaultdict(list)
for r in read('per_class_metrics.csv'):
    classes[by_key[r['run_id']]['protocol'],r['class'],r['evaluation_set']].append(float(r['iou']))
for r in read('Table_S5_Class_scores.csv'):
    p, c = r['protocol'],r['land_cover_class']
    v, g = (mean(classes[p,c,s]) for s in ('validation_fold','geographic_reference'))
    assert numbers([v,g,v-g],[r['mean_validation_iou'],r['mean_reference_iou'],r['signed_gap']])

for r in read('Table_S1_Executed_design.csv'):
    p = r['protocol']
    partition = read(f'split_membership/{p}.csv')
    if 'eval_fold' in partition[0]:
        fold0 = [x for x in partition if x['eval_fold']=='0']
        train = [x for x in fold0 if x['split']=='train']
        val = [x for x in fold0 if x['split']=='val']
        excluded = [x for x in fold0 if x['split']=='excluded_buffer']
    else:
        excluded = [x for x in partition if x['split'].startswith('excluded')]
        train = [x for x in partition if x['fold']!='0' and not x['split'].startswith('excluded')]
        val = [x for x in partition if x['fold']=='0' and not x['split'].startswith('excluded')]
    assert numbers([len(train),len(val),len(excluded),(len(train)//8)*20,len(runs[p])],
                   [r[k] for k in ('training','validation','excluded','step_attempts','runs')])
    if '--distances' in sys.argv:
        xy = [(radians(float(x['lat'])), radians(float(x['lon']))) for x in train]
        coords = [(a,b,cos(a)) for a,b in xy]
        observed = {x['sample_id']:near_distance(x,coords) for x in val}
        expected = {x['sample_id']:float(x['nearest_train_km']) for x in read('fold0_nearest_distances.csv') if x['protocol']==p}
        assert observed.keys() == expected.keys()
        assert max(abs(observed[k]-expected[k]) for k in observed) < 1e-7
        print(p, 'median separation (km)', round(median(observed.values()),6))

print('Verified 75 runs, main and shared-seed summaries, ranks, 9 split designs, scale points, and regional, class and seed dispersion tables.')
for p, r, shared in zip(main,summary['all_runs'],summary['shared_runs']):
    print(p, f"gap {r['gap']:+.4f}, shared gap {shared['gap']:+.4f}", ' / '.join(r['validation_order']), '|', ' / '.join(r['reference_order']))
