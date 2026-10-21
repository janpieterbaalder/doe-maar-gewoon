"""Rechtzetten + beeldverbetering van de boekpagina's.

Stappen per pagina:
  1. perspectief rechtzetten op de gevonden paginahoeken (volle resolutie)
  2. randjes achtergrond wegsnijden; omslagen bijsnijden tot dezelfde verhouding
  3. belichting egaliseren: glad lichtveld geschat uit het papier zelf (wit + rood + magenta vel,
     elk met eigen kleurwaarde), alleen een brede gradiënt; inhoud blijft onaangetast
  4. witbalans + belichting: papier naar één consistente licht warme gebroken-wit tint;
     pagina's zonder wit papier ijken op het rode/magenta papier van de rest van het boek
  5. zwartpunt (sluier van strooilicht/glans weg), lichte S-curve en vibrance
  6. ruisonderdrukking: kleurruis sterk, luminantie licht (papierstructuur blijft)
  7. volle resolutie bewaren + webversie (verkleind met anti-aliasing en licht verscherpt)

Gebruik:  python process.py analyse        -> stats.json
          python process.py render [namen] -> verbeterde pagina's + webpagina's
"""
import os, sys, json
import numpy as np
import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, '..', "losse foto's")
OUT_FULL = os.path.join(HERE, '..', "verbeterde pagina's (volle resolutie)")
OUT_WEB = os.path.join(HERE, 'web', 'pages')
OUT_HD = os.path.join(HERE, 'web', 'pages-hd')   # leesweergave (inzoomen op telefoon)
HD_W = 2400
OUT_CMP = os.path.join(HERE, 'vergelijk')

OUT_W, OUT_H = 1545, 1600          # webformaat (verhouding ~0,966 = mediane paginaverhouding)
ASPECT = OUT_W / OUT_H
PAPER_TARGET = np.array([236, 234, 227], np.float32)   # sRGB (R,G,B): licht warm gebroken wit
BLACK_TARGET = 20.0                                     # sRGB-waarde voor de donkerste inkt
CROP_TO_ASPECT = {'IMG_9724', 'IMG_9725', 'IMG_9799'}  # omslagen en binnenkant omslag
KEEP_BALANCE = {'IMG_9724', 'IMG_9799'}                 # buitenomslag (glanzend karton)
LUMA_W = np.array([0.0722, 0.7152, 0.2126], np.float32)  # BGR


# ---------- kleurruimte ----------
def srgb_to_lin(v):
    v = v.astype(np.float32) / 255.0
    return np.where(v <= 0.04045, v / 12.92, ((v + 0.055) / 1.055) ** 2.4).astype(np.float32)


def lin_to_srgb(v):
    v = np.clip(v, 0, None)
    s = np.where(v <= 0.0031308, v * 12.92, 1.055 * np.power(v, 1 / 2.4) - 0.055)
    return np.clip(s * 255.0, 0, 255).astype(np.float32)


# ---------- geometrie ----------
def load_full(name):
    fn = [f for f in os.listdir(SRC) if f.split('.')[0] == name][0]
    return cv2.imdecode(np.fromfile(os.path.join(SRC, fn), np.uint8), cv2.IMREAD_COLOR)


def warp_page(img, corners1400):
    h, w = img.shape[:2]
    s = w / 1400.0
    c = np.array(corners1400, np.float32) * s
    wn = int(round((np.linalg.norm(c[1] - c[0]) + np.linalg.norm(c[2] - c[3])) / 2))
    hn = int(round((np.linalg.norm(c[3] - c[0]) + np.linalg.norm(c[2] - c[1])) / 2))
    dst = np.array([[0, 0], [wn - 1, 0], [wn - 1, hn - 1], [0, hn - 1]], np.float32)
    M = cv2.getPerspectiveTransform(c, dst)
    return cv2.warpPerspective(img, M, (wn, hn), flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REPLICATE)


def sliver_trim(page, max_frac=0.012):
    """Snij smalle donkere randjes (graniet/schaduw van onderliggende vellen) weg."""
    g = cv2.cvtColor(page, cv2.COLOR_BGR2GRAY).astype(np.float32)
    H, W = g.shape
    trims = {}
    for side in ('top', 'bottom', 'left', 'right'):
        if side in ('top', 'bottom'):
            band = g if side == 'top' else g[::-1]
            n = int(H * max_frac) + 1
            segs = np.array_split(np.arange(int(W * 0.08), int(W * 0.92)), 12)
            profs = [band[:n + 40, s].mean(axis=1) for s in segs]
        else:
            band = g if side == 'left' else g[:, ::-1]
            n = int(W * max_frac) + 1
            segs = np.array_split(np.arange(int(H * 0.08), int(H * 0.92)), 12)
            profs = [band[s, :n + 40].mean(axis=0) for s in segs]
        ts = []
        for prof in profs:
            ref = np.median(prof[n:n + 40])
            dark = prof[:n] < 0.82 * ref
            t = 0
            while t < n and dark[t]:
                t += 1
            ts.append(t)
        trims[side] = int(np.median(ts))
    t, b, l, r = trims['top'], trims['bottom'], trims['left'], trims['right']
    return page[t:H - b if b else H, l:W - r if r else W], trims


def crop_aspect(page):
    H, W = page.shape[:2]
    if W / H > ASPECT:
        nw = int(round(H * ASPECT)); x0 = (W - nw) // 2
        return page[:, x0:x0 + nw]
    nh = int(round(W / ASPECT)); y0 = (H - nh) // 2
    return page[y0:y0 + nh]


def repair_edges(page, spine):
    """Dunne donkere randjes en lijnen langs de buitenranden (omgekrulde hoek, papierrand van een
    onderliggend vel, graniet, rood omslag eronder) wegsnijden: per buitenrand de diepste lijn
    zoeken die evenwijdig aan de rand loopt en net daarbinnen afsnijden. Kost hooguit ~1,2% lege
    papiermarge en geeft geen retoucheersporen."""
    H, W = page.shape[:2]
    band = max(8, int(round(0.025 * min(H, W))))       # zoekzone ~2,5% van de pagina
    probe = band + max(4, band // 4)
    L = cv2.cvtColor(page, cv2.COLOR_BGR2LAB)[..., 0].astype(np.float32)
    Ls = cv2.GaussianBlur(L, (0, 0), 1.0)
    near = max(3, int(0.0015 * min(H, W)))              # lijnen mogen iets schuin lopen
    cut = {}
    for e in ('top', 'bottom', 'left', 'right'):
        if e == spine:
            continue
        if e in ('top', 'bottom'):
            strip = Ls[:probe + 3] if e == 'top' else Ls[::-1][:probe + 3]
        else:
            strip = (Ls[:, :probe + 3] if e == 'left' else Ls[:, ::-1][:, :probe + 3]).T
        ref = cv2.blur(strip[probe:probe + 3].mean(0, keepdims=True), (31, 1))   # papier iets naar binnen
        dark = ((strip[:probe] < 0.9 * ref) & (ref > 150)).astype(np.uint8)
        dark[band:] = 0
        # schuine lijnen aaneensluiten, dan alleen houden wat over een flink stuk langs de rand loopt
        dark = cv2.dilate(dark, np.ones((2 * near + 1, 1), np.uint8))
        klen = max(15, int(0.04 * strip.shape[1]))
        dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, np.ones((1, klen), np.uint8)) > 0
        dark[band:] = False
        has = dark.any(axis=0)
        if has.mean() < 0.04:
            cut[e] = 0
            continue
        deep = band - np.argmax(dark[:band][::-1], axis=0)
        cut[e] = int(min(band, np.percentile(deep[has], 98) + 2))
    t, b, l, r = (cut.get(k, 0) for k in ('top', 'bottom', 'left', 'right'))
    # verhouding behouden: wat er aan de ene kant af gaat, gaat ook van de andere buitenrand af
    h2, w2 = H - t - b, W - l - r
    want = W / H
    if w2 / h2 > want + 1e-4:                       # te breed: extra van de buitenzijde
        extra = int(round(w2 - h2 * want))
        if spine == 'left': r += extra
        elif spine == 'right': l += extra
        else: l += extra // 2; r += extra - extra // 2
    elif w2 / h2 < want - 1e-4:                     # te smal: extra van boven/onder
        extra = int(round(h2 - w2 / want))
        t += extra // 2; b += extra - extra // 2
    cut = {'top': t, 'bottom': b, 'left': l, 'right': r}
    return page[t:H - b, l:W - r], cut


SPINE = {}   # naam -> 'left' / 'right' / None (omslagen), gevuld vanuit quads_refined.json


def geometry(name, corners):
    page = warp_page(load_full(name), corners)
    page, trims = sliver_trim(page)
    if name in CROP_TO_ASPECT:
        page = crop_aspect(page)
    page, _ = repair_edges(page, SPINE.get(name))
    return page, trims


# ---------- analyse ----------
def paper_classes(bgr, scale=4):
    """Maskers van egale papiervlakken: 0 = wit, 1 = rood vel, 2 = magenta vel."""
    small = cv2.resize(bgr, (bgr.shape[1] // scale, bgr.shape[0] // scale), interpolation=cv2.INTER_AREA)
    lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB).astype(np.float32)
    L = lab[..., 0] * 100 / 255; a = lab[..., 1] - 128; b = lab[..., 2] - 128
    mu = cv2.blur(L, (9, 9)); sd = np.sqrt(np.clip(cv2.blur(L * L, (9, 9)) - mu * mu, 0, None))
    er = lambda m: cv2.erode(m.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    flat = er(sd < 2.2)
    felt = er(sd < 3.2)  # gekleurd vel heeft een vilt-achtige structuur
    chroma = np.hypot(a, b)
    white = flat & (L > 60) & (chroma < 12)
    if white.sum() > 200:  # alleen de hoofdmodus van het papier, geen lichtgrijze fotovlakken
        Lw = np.median(L[white]); white &= (L > Lw - 14) & (L < Lw + 10)
    red = felt & (a > 30) & (b > 4) & (b < 40) & (L > 35) & (L < 70)
    mag = felt & (a > 22) & (b < -6) & (L > 36) & (L < 65)   # donkerpaarse titelbalken (L~32) vallen erbuiten
    m = np.zeros_like(white); bh, bw = max(1, int(m.shape[0] * 0.015)), max(1, int(m.shape[1] * 0.015))
    m[bh:-bh, bw:-bw] = True
    return small, [white & m, red & m, mag & m]


def design(x, y, deg):
    cols = []
    for d in range(1, deg + 1):
        for i in range(d + 1):
            cols.append((x - .5) ** (d - i) * (y - .5) ** i)
    return np.stack(cols, 1) if cols else np.zeros((len(x), 0))


def fit_light(small, masks, deg=2):
    """log(Y) = poly(x,y) + c_klasse  (robuust). Geeft coëfficiënten + klasse-offsets terug."""
    Y = srgb_to_lin(small) @ LUMA_W
    h, w = Y.shape
    rng = np.random.default_rng(1)
    X, C, V = [], [], []
    used = []
    # één papiersoort: wit als die er genoeg is, anders het grootste gekleurde vel
    # (twee soorten die ruimtelijk gescheiden liggen maken het model onbepaald)
    frac = [m.mean() for m in masks]
    if frac[0] >= 0.12:
        pick = 0
    elif max(frac[1:]) >= 0.05:
        pick = 1 + int(np.argmax(frac[1:]))
    else:
        return None
    for k, m in enumerate(masks):
        n = int(m.sum())
        if k != pick or n < 300:
            continue
        yy, xx = np.nonzero(m)
        sel = rng.choice(n, min(n, 8000), replace=False)
        X.append(np.stack([xx[sel] / w, yy[sel] / h], 1)); C.append(np.full(len(sel), len(used))); V.append(np.log(Y[yy[sel], xx[sel]] + 1e-4))
        used.append(k)
    if not used:
        return None
    X = np.concatenate(X); C = np.concatenate(C); V = np.concatenate(V)
    # bij weinig spreiding van de punten: alleen een vlak (graad 1)
    spread = np.ptp(X[:, 0]) * np.ptp(X[:, 1])
    if spread < 0.35:
        deg = 1
    P = design(X[:, 0], X[:, 1], deg)
    D = np.concatenate([P, np.eye(len(used))[C]], 1)
    keep = np.ones(len(V), bool)
    for _ in range(5):
        coef, *_ = np.linalg.lstsq(D[keep], V[keep], rcond=None)
        res = V - D @ coef
        s = 1.4826 * np.median(np.abs(res[keep])) + 1e-4
        keep = np.abs(res) < 2.5 * s
    poly = coef[:P.shape[1]]
    fit_at_samples = P[keep] @ poly
    return {'deg': deg, 'poly': poly.tolist(), 'classes': used,
            'lo': float(np.percentile(fit_at_samples, 1)), 'hi': float(np.percentile(fit_at_samples, 99)),
            'resid': float(s), 'spread': float(spread)}


def light_gain(field, H, W):
    """Vermenigvuldigingskaart (volle grootte) die het lichtveld vlak trekt; geen extrapolatie buiten
    het gemeten bereik."""
    if field is None:
        return None
    gx, gy = np.meshgrid(np.linspace(0, 1, 96), np.linspace(0, 1, 96))
    F = design(gx.ravel(), gy.ravel(), field['deg']) @ np.array(field['poly'])
    F = np.clip(F, field['lo'] - 0.02, field['hi'] + 0.02).reshape(96, 96)
    ref = (field['lo'] + field['hi']) / 2
    G = np.clip(np.exp(-(F - ref)), 0.8, 1.25).astype(np.float32)
    return cv2.resize(G, (W, H), interpolation=cv2.INTER_CUBIC)


def class_medians(lin_small, masks):
    out = []
    for m in masks:
        out.append(np.median(lin_small[m], axis=0).tolist() if m.sum() > 300 else None)
    return out


def analyse_one(item):
    name, corners = item
    page, trims = geometry(name, corners)
    small, masks = paper_classes(page)
    field = fit_light(small, masks)
    lin = srgb_to_lin(small)
    if field is not None:
        G = light_gain(field, *small.shape[:2])
        lin = lin * G[..., None]
    med = class_medians(lin, masks)
    frac = [float(m.mean()) for m in masks]
    # donkerste inkt (0,5e percentiel van luminantie, na egaliseren)
    Y = lin @ LUMA_W
    dark = float(np.percentile(Y, 0.5))
    return name, {'trims': trims, 'shape': page.shape[:2], 'field': field, 'med': med, 'frac': frac, 'dark': dark}


# ---------- correctie ----------
def compute_gains(stats):
    """Per pagina kanaalversterkingen (lineair, BGR) zodat het papier de doelkleur krijgt."""
    tgt = srgb_to_lin(PAPER_TARGET[::-1].reshape(1, 3))[0]  # BGR lineair
    gains, how = {}, {}
    for n, s in stats.items():
        if s['med'][0] is not None and s['frac'][0] > 0.12:
            gains[n] = (tgt / np.array(s['med'][0])).tolist(); how[n] = 'wit'
    # referentiekleur rood en magenta vel na correctie (mediaan over het boek)
    refs = []
    for k in (1, 2):
        vals = [np.array(stats[n]['med'][k]) * np.array(g) for n, g in gains.items() if stats[n]['med'][k] is not None and stats[n]['frac'][k] > 0.03]
        refs.append(np.median(vals, axis=0) if len(vals) >= 2 else None)
    names = sorted(stats)
    for n in KEEP_BALANCE:  # glanzende buitenomslag: ander materiaal, eigen kleurbalans behouden
        gains[n] = [1.0, 1.0, 1.0]; how[n] = 'origineel'
    for n in names:
        if n in gains:
            continue
        s = stats[n]
        cand = [(s['frac'][k], k) for k in (1, 2) if s['med'][k] is not None and refs[k - 1] is not None and s['frac'][k] > 0.05]
        if cand:
            _, k = max(cand)
            gains[n] = (refs[k - 1] / np.array(s['med'][k])).tolist(); how[n] = ('rood', 'magenta')[k - 1]
    for n in names:   # niets bruikbaars: buren (zelfde fotosessie) als leidraad
        if n in gains:
            continue
        i = names.index(n)
        nb = [gains[m] for m in names[max(0, i - 3):i + 4] if m in gains and how.get(m) == 'wit']
        gains[n] = np.median(nb, axis=0).tolist(); how[n] = 'buren'
    return gains, [None if r is None else r.tolist() for r in refs], how


def tone(lin, dark_lin, paper_lin):
    """Zwartpunt naar BLACK_TARGET met papier vast (op luminantie, kleurverhoudingen blijven gelijk);
    daarna milde S-curve en vibrance (in sRGB/Lab)."""
    b_t = float(srgb_to_lin(np.array([BLACK_TARGET]))[0])
    # alleen als er echt donkere inkt/schaduw is (sRGB < ~75); maximaal tot sRGB ~60 optrekken
    if dark_lin < 0.07:
        B = min(dark_lin, 0.045)
        if B > b_t:
            Y = lin @ LUMA_W
            k = (paper_lin - b_t) / (paper_lin - B)
            Yn = np.where(Y > B, (Y - B) * k + b_t, Y * (b_t / B))
            lin = lin * (Yn / np.maximum(Y, 1e-5))[..., None]
    s = lin_to_srgb(lin) / 255.0
    # milde S-curve rond het middengrijs; papier (~0,92) en zwart blijven vrijwel gelijk
    s = np.clip(s, 0, 1)
    s = s + 0.06 * np.sin(np.pi * s) * (s - 0.5) * 2 * (1 - s) * 1.6
    s8 = np.clip(s * 255, 0, 255).astype(np.uint8)
    lab = cv2.cvtColor(s8, cv2.COLOR_BGR2LAB).astype(np.float32)
    a = lab[..., 1] - 128; b = lab[..., 2] - 128
    c = np.hypot(a, b)
    boost = 1 + 0.10 * np.clip(1 - c / 45, 0, 1)        # vibrance: alleen weinig verzadigde (foto)kleuren
    lab[..., 1] = 128 + a * boost; lab[..., 2] = 128 + b * boost
    return cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)


def denoise(bgr8, strong=False):
    ycc = cv2.cvtColor(bgr8, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycc)
    # kleurruis: randbehoudend gladstrijken van de chroma, gestuurd door de luminantie
    cr = cv2.ximgproc.guidedFilter(y, cr, 4, 40) if hasattr(cv2, 'ximgproc') else cv2.bilateralFilter(cr, 7, 12, 5)
    cb = cv2.ximgproc.guidedFilter(y, cb, 4, 40) if hasattr(cv2, 'ximgproc') else cv2.bilateralFilter(cb, 7, 12, 5)
    # luminantie: lichte NLM (sigma ~1-3 op 8 bit)
    y = cv2.fastNlMeansDenoising(y, None, h=3.0 if strong else 2.2, templateWindowSize=5, searchWindowSize=15)
    return cv2.cvtColor(cv2.merge([y, cr, cb]), cv2.COLOR_YCrCb2BGR)


def unsharp(bgr8, sigma, amount, thresh=2):
    f = bgr8.astype(np.float32)
    blur = cv2.GaussianBlur(f, (0, 0), sigma)
    d = f - blur
    d = np.where(np.abs(d) < thresh, 0, d)
    return np.clip(f + amount * d, 0, 255).astype(np.uint8)


def render_one(args):
    name, corners, st, gain = args
    page, _ = geometry(name, corners)
    H, W = page.shape[:2]
    lin = srgb_to_lin(page)
    G = light_gain(st['field'], H, W)
    if G is not None:
        lin *= G[..., None]
    lin *= np.array(gain, np.float32)
    paper_lin = float(srgb_to_lin(PAPER_TARGET.reshape(1, 3))[0] @ np.array([0.2126, 0.7152, 0.0722]))
    dark = st['dark'] * float(np.array(gain) @ LUMA_W)
    out = tone(lin, dark, paper_lin)
    del lin
    out = denoise(out)
    full = unsharp(out, 1.2, 0.25, 3)
    os.makedirs(OUT_FULL, exist_ok=True)
    cv2.imencode('.jpg', full, [cv2.IMWRITE_JPEG_QUALITY, 93])[1].tofile(os.path.join(OUT_FULL, name + '.jpg'))
    hd_h = int(round(HD_W * OUT_H / OUT_W))
    hd = unsharp(cv2.resize(out, (HD_W, hd_h), interpolation=cv2.INTER_AREA), 1.0, 0.3, 3)
    os.makedirs(OUT_HD, exist_ok=True)
    cv2.imencode('.jpg', hd, [cv2.IMWRITE_JPEG_QUALITY, 80, cv2.IMWRITE_JPEG_PROGRESSIVE, 1])[1].tofile(os.path.join(OUT_HD, name + '.jpg'))
    web = cv2.resize(out, (OUT_W, OUT_H), interpolation=cv2.INTER_AREA)
    web = unsharp(web, 0.8, 0.35, 3)
    os.makedirs(OUT_WEB, exist_ok=True)
    cv2.imencode('.jpg', web, [cv2.IMWRITE_JPEG_QUALITY, 86, cv2.IMWRITE_JPEG_PROGRESSIVE, 1])[1].tofile(os.path.join(OUT_WEB, name + '.jpg'))
    # vergelijking: zelfde uitsnede zonder verbetering, naast elkaar
    raw = cv2.resize(page, (OUT_W, OUT_H), interpolation=cv2.INTER_AREA)
    os.makedirs(OUT_CMP, exist_ok=True)
    cmp_ = np.concatenate([raw, np.full((OUT_H, 12, 3), 255, np.uint8), web], 1)
    cmp_ = cv2.resize(cmp_, (cmp_.shape[1] // 2, cmp_.shape[0] // 2), interpolation=cv2.INTER_AREA)
    cv2.imencode('.jpg', cmp_, [cv2.IMWRITE_JPEG_QUALITY, 88])[1].tofile(os.path.join(OUT_CMP, name + '.jpg'))
    return name


for _n, _v in json.load(open(os.path.join(HERE, 'quads_refined.json'))).items():
    SPINE[_n] = {'L': 'right', 'R': 'left'}.get(_v['side'])

if __name__ == '__main__':
    from multiprocessing import Pool
    Qr = json.load(open(os.path.join(HERE, 'quads_refined.json')))
    mode = sys.argv[1]
    if mode == 'analyse':
        with Pool(6) as p:
            stats = dict(p.map(analyse_one, [(n, v['corners']) for n, v in sorted(Qr.items())]))
        gains, refs, how = compute_gains(stats)
        json.dump({'stats': stats, 'gains': gains, 'refs': refs, 'how': how}, open(os.path.join(HERE, 'stats.json'), 'w'), indent=1)
        for n in sorted(stats):
            s = stats[n]
            f = s['field']
            print(n, 'wit%.0f rood%.0f mag%.0f' % tuple(100 * f for f in s['frac']), 'licht max/min %.2f' % (0 if not f else np.exp(f['hi'] - f['lo'])), 'cls', None if not f else f['classes'],
                  'gain', np.round(gains[n], 2), how[n], 'trim', s['trims'])
        print('refs', refs)
    else:
        data = json.load(open(os.path.join(HERE, 'stats.json')))
        names = sys.argv[2:] or sorted(Qr)
        jobs = [(n, Qr[n]['corners'], data['stats'][n], data['gains'][n]) for n in names]
        with Pool(5) as p:
            for n in p.imap_unordered(render_one, jobs):
                print('klaar', n, flush=True)
