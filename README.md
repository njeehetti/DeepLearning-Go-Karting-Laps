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
- [x] J1 : pipeline circuit (OSM, GPX/GeoJSON, échelle officielle, requêtes spatiales)
- [x] J2 : physique dynamique (calage absolu en attente de données)
- [ ] J3 : IA headless (neuroévolution)
- [ ] J4 : interface console et carte 2D
- [ ] J5 : 3D, mode live, replay
- [ ] J6 : télémétrie, fiche virages, fantômes, réseau
- [ ] J7 : finition, mobile

Tracé : © contributeurs OpenStreetMap (ODbL).

## Ajouter un circuit

1. Obtenir le tracé d'un tour complet, dans le sens de roulage :
   - une trace GPX (RaceChrono, Strava, etc.) ;
   - une LineString GeoJSON ;
   - ou une réponse Overpass (`out body geom`) dont les ways se raccordent par leurs extrémités.
2. Créer `data/tracks/<id>.json` au format `TrackDef` (voir `src/track/types.ts`) : `id`, `name`, `coords` en `[lat, lon]`, `width`, et si besoin `officialLength`, `bridge` (`{ height, ramp, plateau }`) et `startFrac`. `parseTrackFile()` (`src/track/import.ts`) convertit les trois formats ci-dessus.
3. Pour le LYL, `npx tsx scripts/fetch-osm.ts` récupère les ways 1388683501 et 1388688194 et les compare au fichier ; `--write` met à jour le fichier.
