# Kart Lap Lab v2

Des pilotes à réseau de neurones apprennent à boucler le LYL International Karting Circuit (Sungai Long, Kajang) en kart de location Sodikart 270cc ou 390cc. La v1, qui optimise une trajectoire par algorithme génétique, reste disponible dans `legacy/` et sert de « tour théorique » de référence.

## Utilisation

1. `npm run dev`, puis ouvrir http://localhost:5173.
2. **Lancer** : une population de pilotes (256 par défaut) roule en parallèle dans des Web Workers. La vue « Génération en cours » montre la génération entière en karts semi-transparents, avec le meilleur en ambre.
3. **Meilleur tour** : replay du meilleur cerveau avec les caméras embarquée, poursuite, hélico et libre, le HUD (vitesse, pédales, g-mètre, prochain virage) et un fantôme au choix (tour théorique, génération N−50, autre kart).
4. **Télémétrie** et **fiche virages** : point de freinage, vitesses d'entrée, de corde et de sortie, placement, écart avec le fantôme. Un clic rejoue le passage concerné en 3D.
5. **Kart et calage** : saisir un tour parfait visé, puis recaler la puissance, l'adhérence ou les deux. Les cerveaux sont gardés dans le navigateur et s'exportent en JSON.

Entraînement sans interface : `npx tsx scripts/train-headless.ts --kart k270 --minutes 10`.

## Calage et hypothèses

Les données du brief (9 ch, 62 km/h, µ 1,15) ne permettent pas de descendre sous 1:43, alors que les chronos réels du 270cc sont de 1:32 à 1:36 (1:33.33 relevé à 57,5 km/h de moyenne, soit un tour chronométré d'environ 1 491 m). Le kart est donc calé pour que le **tour parfait** (borne idéale : adhérence pleine, freinage combiné à 0,6 µg, accélération limitée par le moteur, vibreurs utilisables) fasse 1:31.5 : µ 1,3 et environ 9,9 ch effectifs. La pointe (80 km/h) et la transmission (variateur) sont des hypothèses. Une trace GPS (RaceChrono) permettrait de les caler vraiment.

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
- [x] J3 : IA headless (neuroévolution sep-CMA-ES, premier tour propre en quelques secondes)
- [x] J4 : interface console (la carte 2D est remplacée par la vue hélico et la vue libre en 3D)
- [x] J5 : 3D de nuit, mode live, replay avec 4 caméras
- [x] J6 : télémétrie, fiche virages, fantômes, réseau
- [ ] J7 : finition (tests E2E Playwright, mobile vérifié, entraînements plus longs)

Tracé : © contributeurs OpenStreetMap (ODbL).

## Ajouter un circuit

1. Obtenir le tracé d'un tour complet, dans le sens de roulage :
   - une trace GPX (RaceChrono, Strava, etc.) ;
   - une LineString GeoJSON ;
   - ou une réponse Overpass (`out body geom`) dont les ways se raccordent par leurs extrémités.
2. Créer `data/tracks/<id>.json` au format `TrackDef` (voir `src/track/types.ts`) : `id`, `name`, `coords` en `[lat, lon]`, `width`, et si besoin `officialLength`, `bridge` (`{ height, ramp, plateau }`) et `startFrac`. `parseTrackFile()` (`src/track/import.ts`) convertit les trois formats ci-dessus.
3. Pour le LYL, `npx tsx scripts/fetch-osm.ts` récupère les ways 1388683501 et 1388688194 et les compare au fichier ; `--write` met à jour le fichier.
