"""Bouwt de webversie (web/index.html, volledige pagina voor Vercel) en de Claude-artifactversie
(artifact/index.html, alleen de inhoud) uit src/book.template.html.

Gebruik:  python build.py
"""
import os, json, hashlib, shutil
import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
WEB = os.path.join(HERE, 'web')
SITE_URL = 'https://doe-maar-gewoon.vercel.app'
TITLE = 'Doe maar gewoon'
DESC = 'Bladerboek van het spiraalboek over de geschiedenis van Nieuw Baalderborg (maart 2005).'


def version(folder):
    """Korte hash over de bestanden, zodat browsers na een update de nieuwe foto's ophalen."""
    h = hashlib.sha1()
    for n in sorted(os.listdir(folder)):
        st = os.stat(os.path.join(folder, n))
        h.update(f'{n}{st.st_size}{int(st.st_mtime)}'.encode())
    return h.hexdigest()[:8]


def icons():
    """Pictogrammen uit de omslag: een uitsnede rond de titelbalk."""
    cover = cv2.imread(os.path.join(WEB, 'pages', 'IMG_9724.jpg'))
    H, W = cover.shape[:2]
    sq = cover[int(H * 0.02):int(H * 0.02) + W, :]                 # vierkant bovenaan (rood + titel)
    for size, name in ((180, 'apple-touch-icon.png'), (192, 'icon-192.png'), (512, 'icon-512.png'), (48, 'favicon.png')):
        cv2.imwrite(os.path.join(WEB, name), cv2.resize(sq, (size, size), interpolation=cv2.INTER_AREA))
    # deelafbeelding (1200x630): omslag op de achtergrondkleur
    og = np.full((630, 1200, 3), (231, 228, 230), np.uint8)
    c = cv2.resize(cover, (int(560 * W / H), 560), interpolation=cv2.INTER_AREA)
    x = (1200 - c.shape[1]) // 2
    og[35:595, x:x + c.shape[1]] = c
    cv2.imwrite(os.path.join(WEB, 'og.jpg'), og, [cv2.IMWRITE_JPEG_QUALITY, 86])


def main():
    names = sorted(f for f in os.listdir(os.path.join(WEB, 'pages')) if f.endswith('.jpg'))
    assert len(names) == 72, len(names)
    hd_ok = os.path.isdir(os.path.join(WEB, 'pages-hd')) and len(os.listdir(os.path.join(WEB, 'pages-hd'))) == 72
    v = version(os.path.join(WEB, 'pages'))
    body = open(os.path.join(HERE, 'src', 'book.template.html'), encoding='utf8').read()

    def fill(pages, hd):
        return body.replace('/*PAGES*/[]', json.dumps(pages)).replace('/*HD*/[]', json.dumps(hd))

    # --- website (Vercel) ---
    pages = [f'pages/{n}?v={v}' for n in names]
    hd = [f'pages-hd/{n}?v={v}' for n in names] if hd_ok else []
    content = fill(pages, hd)
    head = f'''<!doctype html>
<html lang="nl">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<meta name="robots" content="noindex, nofollow">
<meta name="description" content="{DESC}">
<meta name="theme-color" content="#e6e4e7" media="(prefers-color-scheme: light)">
<meta name="theme-color" content="#19171b" media="(prefers-color-scheme: dark)">
<meta property="og:type" content="book">
<meta property="og:title" content="{TITLE}">
<meta property="og:description" content="{DESC}">
<meta property="og:image" content="{SITE_URL}/og.jpg">
<meta property="og:url" content="{SITE_URL}/">
<meta name="twitter:card" content="summary_large_image">
<link rel="icon" type="image/png" href="favicon.png">
<link rel="apple-touch-icon" href="apple-touch-icon.png">
<link rel="manifest" href="manifest.webmanifest">
<meta name="mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
<meta name="apple-mobile-web-app-title" content="{TITLE}">
<link rel="preload" as="image" href="{pages[0]}">
<style>html{{box-sizing:border-box;padding-top:env(safe-area-inset-top,0px)}}body{{margin:0;font:14px system-ui,sans-serif}}img{{max-width:100%}}[hidden]{{display:none!important}}</style>
'''
    # de <title> en het eigen <style>-blok uit de template komen direct na deze kop
    title_end = content.index('</title>') + len('</title>')
    title, rest = content[:title_end], content[title_end:]
    # het artifact-meta-label is buiten claude.ai niet nodig
    rest = rest.replace('<meta name="claude:viewport-fit-sides" content="cover">\n', '')
    style_end = rest.index('</style>') + len('</style>')
    html = head + title + rest[:style_end] + '\n</head>\n<body>\n' + rest[style_end:].lstrip() + '\n</body>\n</html>\n'
    open(os.path.join(WEB, 'index.html'), 'w', encoding='utf8').write(html)

    manifest = {
        'name': TITLE, 'short_name': TITLE, 'start_url': './', 'scope': './', 'display': 'standalone',
        'orientation': 'any', 'background_color': '#e6e4e7', 'theme_color': '#5d2b61', 'lang': 'nl',
        'icons': [{'src': 'icon-192.png', 'sizes': '192x192', 'type': 'image/png'},
                  {'src': 'icon-512.png', 'sizes': '512x512', 'type': 'image/png'}],
    }
    json.dump(manifest, open(os.path.join(WEB, 'manifest.webmanifest'), 'w', encoding='utf8'), indent=1, ensure_ascii=False)
    vercel = {
        'cleanUrls': True,
        'headers': [
            {'source': '/(.*)', 'headers': [{'key': 'X-Robots-Tag', 'value': 'noindex, nofollow'}]},
            {'source': '/pages/(.*)', 'headers': [{'key': 'Cache-Control', 'value': 'public, max-age=31536000, immutable'}]},
            {'source': '/pages-hd/(.*)', 'headers': [{'key': 'Cache-Control', 'value': 'public, max-age=31536000, immutable'}]},
        ],
    }
    json.dump(vercel, open(os.path.join(WEB, 'vercel.json'), 'w'), indent=1)
    open(os.path.join(WEB, 'robots.txt'), 'w').write('User-agent: *\nDisallow: /\n')
    icons()

    # --- Claude-artifact (alleen inhoud; de viewer levert het HTML-skelet) ---
    art = os.path.join(HERE, 'artifact')
    os.makedirs(art, exist_ok=True)
    open(os.path.join(art, 'index.html'), 'w', encoding='utf8').write(fill([f'pages/{n}' for n in names], []))
    print('klaar: versie', v, '| hd:', hd_ok, '| index.html', len(html) // 1024, 'KB')


if __name__ == '__main__':
    main()
