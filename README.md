# Kart Lap Lab

**Des pilotes à réseau de neurones apprennent à boucler un circuit de karting** en kart de location 270cc ou 390cc, génération après génération, dans le navigateur. Circuit de référence : le LYL International Karting Circuit (Sungai Long, Kajang, Malaisie), plus huit pistes connues d'Europe. On les regarde progresser en 3D de nuit, puis on étudie le meilleur tour pour savoir où freiner, où remettre les gaz et par où passer.

![Vue Piste : la génération en cours, le chronométrage et le temps perdu par virage](docs/images/apercu.png)

## Ce qu'on y voit

| | |
|---|---|
| ![Génération en cours : le peloton gris et le meneur en ambre](docs/images/live-peloton.png) | **Génération en cours.** Toute la génération roule en même temps : les pilotes en gris, celui qui mène en ambre, suivi par la caméra. Le HUD donne le meilleur tour de la génération et le gain sur la précédente. |
| ![Meilleur tour en poursuite, trajectoire colorée par les pédales](docs/images/pedales.png) | **Meilleur tour.** Replay avec quatre caméras (embarquée, poursuite, hélico, libre), le HUD (vitesse, pédales, g-mètre, prochain virage) et un fantôme : le tour parfait, la génération N−50 ou l'autre kart. La trajectoire se colore par vitesse ou par pédales (vert = gaz, rouge = frein, intensité = dosage). |
| ![Caméra embarquée à l'approche d'une épingle](docs/images/embarquee.png) | **Caméra embarquée.** Ce que voit le pilote, avec le conseil du prochain virage : où freiner, à quelle vitesse passer la corde. |
| ![Tour de chronométrage](docs/images/chrono.png) | **Chronométrage.** Meilleur tour IA, écart au tour parfait et à ton record, plan du circuit avec la position du kart, et le meilleur tour des dernières générations : il ne recule jamais. |
| ![Courbe d'apprentissage](docs/images/apprentissage.png) | **Apprentissage.** Le record (ambre) par génération, le taux de tours bouclés (vert) et le tour parfait (pointillé). |

![Vue Analyse : télémétrie et fiche virages](docs/images/analyse.png)

**Analyse.** Vitesse, gaz et frein le long du tour face au tour parfait, et pour chacun des 19 virages : distance de freinage, vitesses d'entrée, de corde et de sortie, accélération latérale, placement et temps perdu. Un clic rejoue le passage en 3D.

| | |
|---|---|
| ![Vue Cerveau : le réseau du meilleur pilote](docs/images/cerveau.png) | ![Version mobile](docs/images/mobile.png) |
| **Cerveau.** Le réseau du meilleur pilote, avec ses activations en direct pendant le replay. | **Sur téléphone.** Même application, navigation en bas d'écran. |

## Circuits

| Circuit | Lieu | Longueur |
|---|---|---|
| LYL International | Sungai Long, Malaisie (avec pont) | 1 514 m |
| Adria Karting Raceway | Adria, Italie | 1 302 m |
| Karting Genk | Genk, Belgique | 1 360 m |
| Åsum Ring | Kristianstad, Suède | 1 221 m |
| Le Mans Karting International | Le Mans, France (anti-horaire) | 1 384 m |
| PFi | Brandon, Royaume-Uni (avec pont) | 1 382 m |
| Circuito Internazionale Napoli | Sarno, Italie | 1 670 m |
| South Garda Karting | Lonato, Italie | 1 200 m |
| Circuito Internacional de Zuera | Zuera, Espagne | 1 700 m |

Le circuit se choisit en haut de la page (ou avec `?c=genk` dans l'adresse). Chaque circuit garde son meilleur cerveau et ton record, à saisir dans la vue Kart. Le kart reste un kart de location : les records publics de ces pistes, faits en karts de compétition, ne sont pas comparables.

## Lancer

```bash
npm install
npm run dev        # http://localhost:5173
```

Clique **Lancer l'entraînement**. Le premier tour propre arrive en quelques secondes ; laisse tourner quelques minutes pour voir le temps descendre. Le meilleur cerveau est gardé dans le navigateur, et s'exporte en JSON depuis la vue Cerveau.

```bash
npm test           # 81 tests : physique, circuits, IA, non-régression v1, calage
npm run test:slow  # non-régression complète de la v1 (≈ 1 min)
npm run build      # site statique dans dist/
npx tsx scripts/train-headless.ts --track genk --kart k270 --minutes 10   # entraînement sans interface
```


## Comment ça marche

**La physique** tourne à 120 Hz, de façon déterministe : pneus Pacejka simplifiés, essieu arrière rigide, frein arrière seul qui peut bloquer, roue intérieure qui se soulève en virage serré, transfert de charge, pente du pont, herbe, vibreurs, murs de pneus. Un tête-à-queue est possible.

**Le pilote** est un réseau de neurones 18-16-3. Il reçoit 9 rayons vers les bords de piste, sa vitesse, sa dérive, son lacet, son écart latéral, son erreur de cap, son braquage actuel et la courbure de la piste à 10, 25 et 50 m. Il décide 30 fois par seconde de la direction, du gaz et du frein, avec l'inertie des mains sur le volant. Comme un vrai pilote, il n'a qu'un pied droit : dès qu'il freine, il lâche les gaz.

**L'apprentissage** est une neuroévolution (sep-CMA-ES) : à chaque génération, 256 variantes du cerveau roulent en parallèle dans des Web Workers ; les meilleures orientent la génération suivante. Le meilleur pilote reroule à chaque génération, donc le record ne recule jamais. D'abord finir le tour, ensuite aller vite ; scier le volant ou tapoter les pédales coûte des points.

**Le tour parfait** est une borne quasi-statique : la meilleure trajectoire possible pour ce kart, avec son adhérence, son moteur et son freinage. Il sert de fantôme et de référence pour le temps perdu par virage.

## Calage et hypothèses

Avec les données d'un kart de location (9 ch, 62 km/h, µ 1,15), même un pilote parfait ne descend pas sous 1:43, alors que les chronos réels au LYL sont de 1:32 à 1:36 en 270cc (un tour en 1:33.33 relevé à 57,5 km/h de moyenne). Le kart est donc calé pour que le tour parfait fasse **1:31.5**, juste sous les meilleurs chronos connus, avec µ 1,3 et environ 9,9 ch effectifs.

La vitesse de pointe (80 km/h) et la transmission (variateur) sont des hypothèses. Une trace GPS d'un tour réel (RaceChrono, par exemple) permettrait de les caler, et de comparer ta trajectoire à celle de l'IA virage par virage. La vue Kart permet de recaler le modèle sur un autre temps.

## Ajouter un circuit

1. Récupère le tracé d'un tour complet dans le sens de roulage : trace GPX, LineString GeoJSON, ou réponse Overpass (`out body geom`).
2. Crée `data/tracks/<id>.json` au format `TrackDef` (voir `src/track/types.ts`) : `id`, `name`, `coords` en `[lat, lon]`, `width`, et si besoin `officialLength` (avec sa source dans `lengthSource`), `bridge` et `startFrac`. `parseTrackFile()` (`src/track/import.ts`) convertit les trois formats. Le fichier est repris automatiquement par le sélecteur et par les tests.
3. Depuis OpenStreetMap : mets les ways du tracé dans `osm.ways` (et le premier dans `osm.main`), puis `npx tsx scripts/fetch-osm.ts --track <id> --write` les raccorde dans le sens de roulage et remplit `coords`.

## Organisation du code

| Dossier | Rôle |
|---|---|
| `src/track/` | Format de circuit, import GPX/GeoJSON/Overpass, géométrie, virages, pont, requêtes spatiales |
| `src/sim/` | Physique pure (sans DOM) : modèle dynamique, moteur, tour parfait, calage, solveur et AG de la v1 |
| `src/ai/` | Réseau, capteurs, épisodes, sep-CMA-ES, pool de workers, enregistrement des tours |
| `src/workers/` | Workers du navigateur : simulation et outils (replay, tour parfait, calage) |
| `src/ui/` | Interface : vues, scène 3D (Three.js), graphiques, réseau |
| `data/tracks/` | Circuits au format `TrackDef` |
| `legacy/` | La v1, intacte (lien depuis la vue Kart) |

## Crédits

Tracé du circuit : © contributeurs [OpenStreetMap](https://www.openstreetmap.org/copyright), sous licence ODbL.
