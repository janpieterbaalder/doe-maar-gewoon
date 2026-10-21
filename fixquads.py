"""Controle op paginaverhouding: een rand die op de schaduw in plaats van op de papierrand is
gesnapt geeft een te hoge pagina. Zo'n rand opnieuw zoeken rond de voorspelde positie."""
import os, json
import numpy as np
from detect import load_work, grad_maps, snap_edge, S

HERE = os.path.dirname(os.path.abspath(__file__))
Q = json.load(open(os.path.join(HERE, 'quads_refined.json')))


def dims(c):
    c = np.array(c)
    w = (np.linalg.norm(c[1] - c[0]) + np.linalg.norm(c[2] - c[3])) / 2
    h = (np.linalg.norm(c[3] - c[0]) + np.linalg.norm(c[2] - c[1])) / 2
    return w, h


def line_y(l, x):
    return l[0] * x + l[1]


med = {s: np.median([dims(v['corners'])[0] / dims(v['corners'])[1] for v in Q.values() if v['side'] == s]) for s in 'LR'}
changed = []
for n, v in sorted(Q.items()):
    if v['side'] not in 'LR':
        continue
    w, h = dims(v['corners'])
    if abs(w / h - med[v['side']]) < 0.012:
        continue
    c = np.array(v['corners'], float)
    weak = 'bot' if (v['info'].get('bot') or 0) <= (v['info'].get('top') or 0) else 'top'
    h_pred = w / med[v['side']]
    _, work = load_work(n)
    gx, gy = grad_maps(work)
    if weak == 'bot':
        TL, TR = c[0], c[1]
        dl = (c[3] - c[0]) / np.linalg.norm(c[3] - c[0]); dr = (c[2] - c[1]) / np.linalg.norm(c[2] - c[1])
        BL, BR = TL + dl * h_pred, TR + dr * h_pred
        res = snap_edge(gy, BL * S, BR * S, True, search=16)
        if res:
            a, b = res[0], res[1] / S
            # snijpunt van nieuwe onderlijn met linker- en rechterrand
            def hit(P, d):
                t = (a * P[0] * S + b * S - P[1] * S) / (d[1] * S - a * d[0] * S)
                return P + d * t
            BL, BR = hit(TL, dl), hit(TR, dr)
        BL = BL - dl * 1.5; BR = BR - dr * 1.5      # kleine inzet, net binnen het papier
        c[3], c[2] = BL, BR
    else:
        BL, BR = c[3], c[2]
        dl = (c[0] - c[3]) / np.linalg.norm(c[0] - c[3]); dr = (c[1] - c[2]) / np.linalg.norm(c[1] - c[2])
        TL, TR = BL + dl * h_pred, BR + dr * h_pred
        res = snap_edge(gy, TL * S, TR * S, True, search=16)
        if res:
            a, b = res[0], res[1] / S
            def hit(P, d):
                t = (a * P[0] * S + b * S - P[1] * S) / (d[1] * S - a * d[0] * S)
                return P + d * t
            TL, TR = hit(BL, dl), hit(BR, dr)
        TL = TL - dl * 1.5; TR = TR - dr * 1.5
        c[0], c[1] = TL, TR
    w2, h2 = dims(c)
    v['corners'] = [[round(float(x), 1), round(float(y), 1)] for x, y in c]
    v['info']['fixed_' + weak] = 1
    changed.append((n, weak, round(w / h, 3), round(w2 / h2, 3), bool(res)))

json.dump(Q, open(os.path.join(HERE, 'quads_refined.json'), 'w'), indent=1)
print('mediaan', med)
for row in changed:
    print(row)
