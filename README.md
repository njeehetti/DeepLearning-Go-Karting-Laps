# Kart Lap Lab v2

Des pilotes à réseau de neurones apprennent à boucler le LYL International Karting Circuit (Sungai Long, Kajang) en kart de location Sodikart 270cc ou 390cc. La v1, qui optimise une trajectoire par algorithme génétique, reste disponible dans `legacy/` et sert de « tour théorique » de référence.

> Travail en cours. Le jalon actuel et la suite sont décrits plus bas.

## Lancer

```bash
npm install
npm run dev          # http://localhost:5173 (la v1 est à /legacy/kart-lap-lab.html)
npm test             # tests rapides : portage v1 bit à bit, chargement du circuit
npm run test:slow    # non-régression complète : AG v1 sur 3 000 générations (≈ 1 min)
```

## Organisation

| Dossier | Rôle |
|---|---|
| `src/track/` | Format de circuit générique, géométrie, virages, pont |
| `src/sim/` | Physique pure (sans DOM) : solveur quasi-statique v1, AG de trajectoire, puis modèle dynamique |
| `src/ai/` | Réseaux de neurones et neuroévolution (à venir) |
| `src/ui/` | Interface (à venir) |
| `data/tracks/` | Circuits au format `TrackDef` (JSON) |
| `legacy/` | v1 d'origine, intacte |

## Référence v1 (non-régression)

Échelle OSM brute (1 575 m), µ = 1,15, pilote 75 kg, AG seedé (12345), 3 000 générations :

| Kart | Tour théorique v1 |
|---|---|
| 390cc | 1:36.85 |
| 270cc | 1:43.86 |

## Jalons

- [x] J0 : socle Vite + TS, portage v1 vérifié bit à bit, non-régression
- [ ] J1 : pipeline circuit (OSM, GPX/GeoJSON, échelle officielle, requêtes spatiales)
- [ ] J2 : physique dynamique
- [ ] J3 : IA headless (neuroévolution)
- [ ] J4 : interface console et carte 2D
- [ ] J5 : 3D, mode live, replay
- [ ] J6 : télémétrie, fiche virages, fantômes, réseau
- [ ] J7 : finition, mobile

Tracé : © contributeurs OpenStreetMap (ODbL).
