# Doe maar gewoon — bladerboek

Online bladerboek van het spiraalboek *Doe maar gewoon* (Nieuw Baalderborg, maart 2005),
gemaakt van 72 telefoonfoto's van de pagina's.

- **`web/`** is de website zoals Vercel hem publiceert (statisch, geen build-stap).
  - `pages/` — pagina's voor het bladerboek (1545 × 1600)
  - `pages-hd/` — scherpere versies voor de leesweergave (2400 breed)
- **`src/book.template.html`** — de pagina zelf (bladeranimatie, spiraal, inhoud, leesweergave).
- **`build.py`** — maakt `web/index.html`, pictogrammen en `vercel.json` uit de template.
- **`detect.py`, `fixquads.py`, `process.py`** — beeldbewerking van de originele foto's:
  paginahoeken vinden, perspectief rechtzetten, randen opschonen, belichting egaliseren,
  witbalans, contrast, ruisonderdrukking en verscherping.

De originele foto's staan niet in deze repository (map `losse foto's` naast deze map).

## Bijwerken

```bash
python process.py analyse   # papierkleur en belichting per pagina meten
python process.py render    # alle pagina's opnieuw maken
python build.py             # website opbouwen
git add -A && git commit -m "Bijwerken" && git push   # Vercel publiceert automatisch
```

De site staat op `noindex`: bereikbaar via de link, maar niet vindbaar via zoekmachines.
