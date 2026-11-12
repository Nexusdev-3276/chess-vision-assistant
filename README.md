# Chess Vision — assistant d'échecs visuel passif

Chess Vision lit un échiquier affiché à l'écran, demande le meilleur coup à
Stockfish et dessine une flèche transparente par-dessus le plateau. Il ne
contrôle jamais la souris, le clavier ou l'application de jeu : vous jouez
vous-même, normalement.

> Utilisez cet outil dans le respect des règles de la plateforme sur laquelle
> vous jouez et de la législation applicable. Ne l'utilisez pas pour tricher
> dans une partie en ligne.

## Fonctionnalités

- Capture en temps réel de l'écran principal avec `mss` et OpenCV.
- Reconnaissance des pièces par apprentissage supervisé de votre thème visuel,
  à partir d'une capture du plateau en position initiale.
- Suivi du trait, des droits de roque et de la prise en passant à partir des
  coups légaux observés.
- Analyse Stockfish dans un thread séparé, sans bloquer l'interface.
- Flèche transparente, toujours au-dessus du plateau et traversée par les clics
  (`Qt.WindowTransparentForInput`).
- Exclusions de la flèche des captures d'écran sous Windows
  (`WDA_EXCLUDEFROMCAPTURE`), avec repli automatique.
- Interface et messages en français.
- Aucune automatisation du jeu : uniquement une indication visuelle.

## Ce que l'application ne fait pas

- Elle ne déplace aucune pièce et n'envoie aucun clic ni aucune touche.
- Elle ne se connecte à aucune plateforme de jeu et n'utilise pas le réseau.
- Elle ne lit pas l'état interne du jeu : elle ne voit que des pixels.
- Elle ne gère qu'un seul échiquier, sur l'écran principal.

## Prérequis

- Windows 10/11 (le code contient aussi un chemin non-Windows, mais
  l'exclusion de l'overlay des captures est spécifique à Windows).
- Python 3.10 ou plus récent. Testé avec Python 3.12.
- Le plateau doit rester fixe, entièrement visible, sans fenêtre superposée.
- Un binaire Stockfish local. Il n'est pas installé par `pip`.

## Installation

```powershell
cd C:\Users\hazem\chess-vision-assistant
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Si l'activation PowerShell est bloquée par la stratégie d'exécution, vous
pouvez appeler l'interpréteur directement :

```powershell
.\.venv\Scripts\python.exe main.py
```

## Installer Stockfish

1. Téléchargez une version Windows depuis
   <https://stockfishchess.org/download/>.
2. Décompressez l'archive.
3. Placez l'exécutable dans le dossier `stockfish\stockfish.exe` du projet.
   C'est l'emplacement détecté automatiquement.
4. Vous pouvez aussi choisir n'importe quel autre chemin dans
   **Réglages → Moteur Stockfish → Parcourir…**, ou définir la variable
   d'environnement `STOCKFISH_PATH`.

Le dossier `stockfish\` est ignoré par Git : le binaire reste une dépendance
externe, sous sa propre licence (Stockfish est distribué sous GPLv3).

## Premier lancement et calibration

1. Ouvrez votre échiquier sur l'écran principal.
2. Mettez-le en **position initiale complète**, sans animation, sans pièce
   sélectionnée et sans fenêtre par-dessus.
3. Lancez :

   ```powershell
   .\.venv\Scripts\python.exe main.py
   ```

4. Au premier lancement, cliquez sur **Calibrer le plateau**.
5. Cliquez les quatre coins de la **zone de jeu**, dans cet ordre :
   haut-gauche → haut-droit → bas-droit → bas-gauche.
   - Clic droit, Retour arrière ou Suppr : annuler le dernier point.
   - Échap : annuler la calibration.
6. Dans l'aperçu, vérifiez que les lignes suivent bien les cases, puis
   choisissez l'orientation :
   - **Blancs en bas** si vous jouez les Blancs ;
   - **Noirs en bas** si vous jouez les Noirs.
7. Cliquez sur **Enregistrer la calibration**.

L'application apprend alors l'apparence des douze pièces du thème affiché et
le niveau d'activité d'une case vide. Il n'y a aucun asset de pièces embarqué :
le thème actuel est appris tel qu'il apparaît.

**Une nouvelle calibration est nécessaire** si vous déplacez ou redimensionnez
la fenêtre du plateau, changez de thème, de taille de pièces, de zoom, ou
changez d'écran. La calibration doit toujours être faite à partir de la
position initiale standard.

## Utilisation

- **Démarrer l'analyse** : commence la lecture de l'écran. Si la zone de
  notification est disponible et que l'option est active, la fenêtre de
  contrôle se masque automatiquement ; l'icône du tray permet de la rouvrir.
- **Mettre en pause** : arrête la capture et masque la flèche.
- **Actualiser** : relit immédiatement l'écran et relance l'analyse.
- **Inverser le trait** : corrige manuellement le trait après une
  resynchronisation ou un démarrage en milieu de partie.
- **Réglages** : chemin de Stockfish, temps d'analyse, fréquence de lecture,
  mode de conseil, couleur jouée, libellé sur la flèche, masquage automatique.
- **Quitter** : par le menu de l'icône de notification. Fermer la fenêtre de
  contrôle la masque si le tray est disponible ; sinon, cela quitte
  l'application.

Par défaut, les conseils ne s'affichent que pendant **votre** trait. Le mode
« Toujours » affiche aussi une suggestion pendant le trait adverse. Le score
Stockfish est affiché du point de vue des Blancs, comme indiqué dans la barre
d'état.

## Comment ça marche

1. `mss` capture l'écran principal en BGR ; la flèche est exclue de la capture
   sur Windows, sinon elle est masquée quelques dizaines de millisecondes.
2. Les quatre coins calibrés définissent une homographie qui redresse le
   plateau en une image de 800 × 800 pixels.
3. Chaque case est découpée, normalisée par le fond de la case, et comparée aux
   modèles des douze pièces appris lors de la calibration. Une case dont
   l'activité est trop faible est considérée comme vide.
4. Le placement observé est comparé au plateau interne de `python-chess`. Si un
   ou deux coups légaux expliquent la différence, ils sont appliqués ; sinon la
   position est resynchronisée et le trait est signalé comme à vérifier.
5. Stockfish analyse la position dans un thread dédié.
6. Le meilleur coup est converti en coordonnées écran via l'homographie
   inverse, puis dessiné par une fenêtre Qt transparente, sans focus et
   traversée par les clics.

### Architecture

| Fichier | Rôle |
| --- | --- |
| `main.py` | Point d'entrée, journalisation, cycle de vie Qt. |
| `chess_vision/app.py` | Fenêtre de contrôle, orchestration, boucle de lecture. |
| `chess_vision/screen.py` | Capture `mss` de l'écran principal. |
| `chess_vision/geometry.py` | Coins, homographie, correspondance cases ↔ écran. |
| `chess_vision/recognition.py` | Calibration et reconnaissance des pièces. |
| `chess_vision/tracker.py` | Suivi du trait et des coups légaux. |
| `chess_vision/engine.py` | Thread Stockfish et suggestions. |
| `chess_vision/overlay.py` | Flèche Qt transparente et exclusion de capture. |
| `chess_vision/calibration.py` | Sélection des coins et aperçu de calibration. |
| `chess_vision/settings.py` | Dialogue de réglages. |
| `chess_vision/config.py` | Configuration, chemins de données, détection Stockfish. |

Les réglages, la calibration, les modèles et le journal sont stockés dans :

```text
%LOCALAPPDATA%\ChessVisionAssistant\
```

## Limites importantes

- **Écran principal uniquement.** Un plateau sur un écran secondaire n'est pas
  pris en charge.
- **Plateau fixe.** Ne déplacez pas, ne redimensionnez pas et ne masquez pas la
  fenêtre du plateau pendant l'analyse. Recalibrez après tout changement.
- **Pas d'animation ni de fenêtre superposée.** Attendez la fin des animations
  de déplacement et des effets de survol. Une pièce sélectionnée avec des points
  de coups possibles peut rendre une lecture incertaine ; l'outil attend alors
  une image stable au lieu de deviner.
- **Plein écran exclusif.** L'overlay Qt peut ne pas apparaître au-dessus d'un
  jeu en plein écran exclusif. Utilisez un mode fenêtré ou « borderless ».
- **Suivi du trait.** La position visuelle ne contient pas le trait. L'outil
  part de la position initiale et suit les coups légaux. S'il rate un coup, il
  resynchronise la position et peut se tromper de trait : utilisez
  **Inverser le trait**. Les droits de roque sont déduits de la présence du roi
  et des tours sur leurs cases d'origine ; ils peuvent être trop optimistes
  après une resynchronisation. La prise en passant ne peut pas être déduite
  d'une position statique : elle n'est connue que si les coups ont été suivis
  correctement.
- **Reconnaissance liée au thème.** Un changement de thème, de pièces, de
  taille de plateau ou de résolution impose une recalibration.
- **Pas de lecture des coordonnées de la souris.** Aucun contrôle du jeu.
- **Pas de partie en ligne « assistée »** au sens de la triche : voir la note
  d'usage responsable en haut de ce document.

## Confidentialité

- Aucune donnée n'est envoyée sur le réseau.
- Les captures d'écran sont traitées en mémoire. Seul un aperçu du plateau
  redressé est enregistré localement (`calibration_preview.png`) pour le
  diagnostic.
- Les modèles de pièces et la configuration restent dans
  `%LOCALAPPDATA%\ChessVisionAssistant\`.
- Le journal `chess_vision.log` peut contenir des chemins locaux, mais aucune
  capture d'écran.

## Dépannage

| Symptôme | Piste |
| --- | --- |
| « Stockfish introuvable » | Vérifiez `stockfish\stockfish.exe`, ou choisissez l'exécutable dans **Réglages**. La variable `STOCKFISH_PATH` est aussi acceptée. |
| « Calibration invalide » | Le plateau a probablement bougé ou la résolution a changé. Recalibrez. En cas de fichier corrompu, supprimez `calibration.json` et `templates.npz` dans `%LOCALAPPDATA%\ChessVisionAssistant\`, puis recalibrez. |
| « Lecture incertaine » | Attendez la fin d'une animation, désélectionnez la pièce, vérifiez qu'aucune fenêtre ne recouvre le plateau, puis cliquez sur **Actualiser**. |
| La flèche n'apparaît pas | Le plateau est peut-être en plein écran exclusif ou sur un autre écran. Vérifiez aussi que l'analyse est démarrée et que c'est votre trait. |
| La flèche clignote pendant la capture | L'exclusion Windows n'est pas disponible sur ce système ; l'overlay est masqué brièvement à chaque lecture. Cela reste fonctionnel. |
| L'application semble bloquée | Consultez `%LOCALAPPDATA%\ChessVisionAssistant\chess_vision.log`. |
| Le trait est faux | Utilisez **Inverser le trait**. Le suivi repose sur les coups légaux, pas sur une information cachée du jeu. |

## Tests

Les tests sont autonomes (aucune capture d'écran, aucune fenêtre) :

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Ils couvrent l'homographie et les orientations, la calibration/reconnaissance
sur un plateau synthétique (y compris un thème avec cases claires/sombres et
surbrillance du dernier coup), le suivi des coups, le roque, la
resynchronisation et la détection de Stockfish.
