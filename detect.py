"""Verfijnt ruwe paginahoeken: snapt randen op de papierrand en vindt de spiraalgrens."""
import os, sys, json
import numpy as np
import cv2

SRC = os.path.join(os.path.dirname(__file__), '..', "losse foto's")
S = 2.0  # werkschaal t.o.v. 1400-breed coördinaten (dus 2800 px breed)


def load_work(name):
    fn = [f for f in os.listdir(SRC) if f.split('.')[0] == name][0]
    img = cv2.imdecode(np.fromfile(os.path.join(SRC, fn), np.uint8), cv2.IMREAD_COLOR)
    h, w = img.shape[:2]
    W = int(1400 * S)
    work = cv2.resize(img, (W, round(h * W / w)), interpolation=cv2.INTER_AREA)
    return img, work


def grad_maps(work):
    lab = cv2.cvtColor(work, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab = cv2.GaussianBlur(lab, (0, 0), 1.6)
    gx = sum(np.abs(cv2.Sobel(lab[..., c], cv2.CV_32F, 1, 0, ksize=3)) * wt for c, wt in ((0, 1.0), (1, 0.6), (2, 0.6)))
    gy = sum(np.abs(cv2.Sobel(lab[..., c], cv2.CV_32F, 0, 1, ksize=3)) * wt for c, wt in ((0, 1.0), (1, 0.6), (2, 0.6)))
    return gx, gy


def ransac_line(pts, horizontal, iters=400, tol=2.5):
    """pts: Nx2 (x,y). horizontal: y = a*x+b ; anders x = a*y+b."""
    pts = np.asarray(pts, np.float64)
    if len(pts) < 5:
        return None
    u = pts[:, 0] if horizontal else pts[:, 1]
    v = pts[:, 1] if horizontal else pts[:, 0]
    rng = np.random.default_rng(0)
    best = None
    for _ in range(iters):
        i, j = rng.choice(len(pts), 2, replace=False)
        if abs(u[i] - u[j]) < 20:
            continue
        a = (v[j] - v[i]) / (u[j] - u[i])
        b = v[i] - a * u[i]
        inl = np.abs(v - (a * u + b)) < tol
        if best is None or inl.sum() > best.sum():
            best = inl
    if best is None or best.sum() < 0.35 * len(pts):
        return None
    a, b = np.polyfit(u[best], v[best], 1)
    return a, b, best.mean()


def snap_edge(g, p0, p1, horizontal, search=24, trim=(0.06, 0.06), n=120):
    """Zoek langs de ruwe lijn p0->p1 (werkcoördinaten) de sterkste rand loodrecht erop."""
    H, W = g.shape
    pts = []
    for t in np.linspace(trim[0], 1 - trim[1], n):
        x = p0[0] + t * (p1[0] - p0[0]); y = p0[1] + t * (p1[1] - p0[1])
        if horizontal:
            xi = int(round(x)); y0 = int(max(0, y - search)); y1 = int(min(H - 1, y + search))
            if not (0 <= xi < W) or y1 <= y0: continue
            col = g[y0:y1 + 1, max(0, xi - 2):xi + 3].mean(axis=1)
            k = int(np.argmax(col))
            if col[k] > 2.5 * np.median(col) + 4:
                pts.append((x, y0 + k))
        else:
            yi = int(round(y)); x0 = int(max(0, x - search)); x1 = int(min(W - 1, x + search))
            if not (0 <= yi < H) or x1 <= x0: continue
            row = g[max(0, yi - 2):yi + 3, x0:x1 + 1].mean(axis=0)
            k = int(np.argmax(row))
            if row[k] > 2.5 * np.median(row) + 4:
                pts.append((x0 + k, y))
    return ransac_line(pts, horizontal)


def ring_extent(work, x_rough, y_top, y_bot, side):
    """Vind de binnengrens van de spiraal (ringen) via verticale periodiciteit per kolom.
    side 'L' = ringen rechts van de pagina; 'R' = ringen links. Geeft x (werkcoörd.) per helft."""
    L = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY).astype(np.float32)
    res = []
    span = y_bot - y_top
    xs = np.arange(int(x_rough - 50 * S), int(x_rough + 50 * S))
    for (ya, yb) in ((y_top + 0.04 * span, y_top + 0.5 * span), (y_top + 0.5 * span, y_bot - 0.04 * span)):
        ya, yb = int(ya), int(yb)
        block = L[ya:yb, xs[0] - 1:xs[-1] + 2]
        block = cv2.blur(block, (3, 1))[:, 1:-1]                      # 3 kolommen middelen
        block = block - cv2.blur(block, (1, int(60 * S)))             # trend eruit
        block = block - block.mean(axis=0)
        den = (block ** 2).sum(axis=0) + 1e-6
        ac = np.max([(block[:-lag] * block[lag:]).sum(axis=0) / den for lag in range(int(30 * S), int(50 * S))], axis=0)
        std = block.std(axis=0)
        # metalen ringen: sterk periodiek én veel contrast (schaduw op papier is periodiek maar zwak)
        ok = (ac > 0.6) & (std > 12)
        okr = np.convolve(ok.astype(float), np.ones(int(10 * S)), mode='same') >= int(10 * S) * 0.9
        idx = np.where(okr)[0]
        if len(idx) == 0:
            x = None
        else:
            # rand van de ringzone aan de paginakant; terug naar de eerste ok-kolom
            if side == 'L':
                i = idx[0] - int(5 * S)
                while i < len(ok) and not ok[max(i, 0)]: i += 1
            else:
                i = idx[-1] + int(5 * S)
                while i >= 0 and not ok[min(i, len(ok) - 1)]: i -= 1
            x = float(xs[int(np.clip(i, 0, len(xs) - 1))])
        res.append((x, (ya + yb) / 2))
    return res


def intersect(h, v):
    """h: y=a1*x+b1 ; v: x=a2*y+b2"""
    a1, b1 = h; a2, b2 = v
    y = (a1 * b2 + b1) / (1 - a1 * a2)
    x = a2 * y + b2
    return (x, y)


def refine(name, side, TL, TR, BR, BL, margin=5):
    img, work = load_work(name)
    gx, gy = grad_maps(work)
    P = [np.array(p, float) * S for p in (TL, TR, BR, BL)]
    TLw, TRw, BRw, BLw = P
    info = {}
    # boven/onder: aan spiraalkant wat meer afsnijden (ringen geven ruis)
    tr_top = (0.05, 0.10) if side == 'L' else (0.10, 0.05) if side == 'R' else (0.05, 0.05)
    top = snap_edge(gy, TLw, TRw, True, trim=tr_top)
    bot = snap_edge(gy, BLw, BRw, True, trim=tr_top)
    def rough_h(p, q):
        a = (q[1] - p[1]) / (q[0] - p[0]); return (a, p[1] - a * p[0])
    def rough_v(p, q):
        a = (q[0] - p[0]) / (q[1] - p[1]); return (a, p[0] - a * p[1])
    topl = top[:2] if top else rough_h(TLw, TRw)
    botl = bot[:2] if bot else rough_h(BLw, BRw)
    info['top'] = None if not top else round(top[2], 2)
    info['bot'] = None if not bot else round(bot[2], 2)
    if side in ('C', 'L'):
        left = snap_edge(gx, TLw, BLw, False)
        leftl = left[:2] if left else rough_v(TLw, BLw)
        info['left'] = None if not left else round(left[2], 2)
    if side in ('C', 'R'):
        right = snap_edge(gx, TRw, BRw, False)
        rightl = right[:2] if right else rough_v(TRw, BRw)
        info['right'] = None if not right else round(right[2], 2)
    m = margin * S
    if side in ('L', 'R'):
        # ruwe ringgrens (handmatig) als fallback wanneer de detectie te ver afwijkt
        ra, rb = (TRw, BRw) if side == 'L' else (TLw, BLw)
        (x1, y1), (x2, y2) = ring_extent(work, (ra[0] + rb[0]) / 2, (TRw[1] + TLw[1]) / 2, (BRw[1] + BLw[1]) / 2, side)
        exp1 = ra[0] + (rb[0] - ra[0]) * (y1 - ra[1]) / (rb[1] - ra[1])
        exp2 = ra[0] + (rb[0] - ra[0]) * (y2 - ra[1]) / (rb[1] - ra[1])
        if x1 is None and x2 is None: x1, x2 = exp1, exp2; info['ringfallback'] = 1
        elif x1 is None: x1 = exp1 + (x2 - exp2); info['ringfallback'] = 1
        elif x2 is None: x2 = exp2 + (x1 - exp1); info['ringfallback'] = 1
        # beide helften horen bijna dezelfde x te geven (ringen staan recht)
        if abs(x1 - x2) > 8 * S:
            x1 = x2 = (min(x1, x2) if side == 'L' else max(x1, x2)); info['ringfallback'] = 2
        a = (x2 - x1) / (y2 - y1)
        if side == 'L': rightl = (a, x1 - m - a * y1)
        else: leftl = (a, x1 + m - a * y1)
        info['ring'] = (round(x1 / S, 1), round(x2 / S, 1))
    # klein beetje naar binnen om geen achtergrondrandje mee te nemen
    inset = 1.5 * S
    topl = (topl[0], topl[1] + inset); botl = (botl[0], botl[1] - inset)
    if side in ('C', 'L'): leftl = (leftl[0], leftl[1] + inset)
    if side in ('C', 'R'): rightl = (rightl[0], rightl[1] - inset)
    corners = [intersect(topl, leftl), intersect(topl, rightl), intersect(botl, rightl), intersect(botl, leftl)]
    return img, work, [(x / S, y / S) for x, y in corners], info


def job(item):
    name, (side, *pts) = item
    _, _, c, info = refine(name, side, *pts)
    d = max(np.hypot(a[0] - b[0], a[1] - b[1]) for a, b in zip(c, pts))
    info = {k: (None if v is None else (list(map(float, v)) if isinstance(v, tuple) else float(v))) for k, v in info.items()}
    return name, {'side': side, 'corners': [[round(float(x), 1), round(float(y), 1)] for x, y in c], 'info': info, 'maxdiff': round(float(d), 1)}


if __name__ == '__main__':
    from multiprocessing import Pool
    sys.path.insert(0, os.path.dirname(__file__))
    from quads_manual import Q
    with Pool(8) as p:
        res = dict(p.map(job, sorted(Q.items())))
    for n, r in sorted(res.items()):
        print(n, r['side'], 'maxdiff=%.1f' % r['maxdiff'], r['info'])
    json.dump(res, open(os.path.join(os.path.dirname(__file__), 'quads_refined.json'), 'w'), indent=1)
