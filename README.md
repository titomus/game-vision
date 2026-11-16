# GameVision: expérience d'amélioration graphique live (v0.1)

**Prototype Windows 10/11, Direct3D 11** : capture du bureau en direct avec **DXGI Desktop Duplication**, traitement de l'image par **pixel shader HLSL entièrement sur GPU**, et affichage dans une fenêtre indépendante. Aucun hook de jeu, injection, contournement d'anticheat ni changement de pilote.

> Ce n'est **pas DLSS 5**, ni un moteur IA, ni une génération d'images. Le logiciel ne garantit pas 144 FPS. Son premier rôle est de mesurer la faisabilité d'une transformation live, visuellement et techniquement, avant d'essayer une IA neuronale.

## Téléchargement / compilation

Chaque push sur \`main\` lance un workflow [GitHub Actions](https://github.com/titomus/game-vision/actions/workflows/build.yml), qui tente de compiler un exécutable Windows portable.

Si le workflow est vert : ouvrir sa dernière exécution, rubrique **Artifacts**, puis télécharger **GameVision-Windows-x64**. La présence du workflow ne signifie pas que la compilation a été validée; vérifier son statut.

Sinon, sous Windows, installer Visual Studio 2022 avec **Développement Desktop en C++**, ou Build Tools MSVC et CMake. Ouvrir le **x64 Native Tools Command Prompt for VS 2022** :

\`\`\`powershell
git clone https://github.com/titomus/game-vision.git
cd game-vision
cmake -S . -B build -A x64
cmake --build build --config Release
.\build\Release\GameVision.exe --monitor 0
\`\`\`

Le numéro \`--monitor N\` identifie une sortie d'écran telle que DXGI les énumère, à partir de 0. Sur une configuration hybride GPU, l'écran du jeu peut être relié au GPU intégré et non à la RTX.

## Touches

| Touche | Action |
|---|---|
| \`1\` | Passage direct sans effet |
| \`2\` | Traitement léger |
| \`3\` | Traitement équilibré (défaut) |
| \`4\` | Traitement plus agressif |
| \`Espace\` | Basculer la comparaison moitié originale / moitié traitée |
| \`+\` / \`-\` | Régler la force des effets |
| \`V\` | Activer/désactiver la synchronisation de présentation |
| \`Échap\` | Fermer |

La barre de titre affiche le nombre de nouvelles images capturées par seconde, le temps d'attente \`AcquireNextFrame\` et le temps **CPU d'envoi des commandes GPU**. Ce dernier **n'est pas** un véritable temps GPU. Le débit de capture n'est **pas** le nombre de FPS calculés par le moteur du jeu.

## Tester

1. Ouvrir un jeu **solo** en mode fenêtre sans bordure. Idéalement utiliser un deuxième écran pour la fenêtre de GameVision.
2. Lancer \`GameVision.exe --monitor 0\`. Changer l'indice si ce n'est pas le bon écran.
3. Appuyer sur \`Espace\` et comparer les zones sur le **même instant vidéo**.
4. Basculer entre les quatre profils et rechercher des artefacts pendant les mouvements.
5. Comparer le compteur FPS **du jeu** avant et pendant la capture, et noter les ralentissements.

L'application demande à Windows d'exclure sa propre fenêtre de la capture (\`SetWindowDisplayAffinity\`). Cela reste une mesure *best effort*. Si la fenêtre apparaît dans son propre flux et provoque un effet miroir, déplacer la fenêtre sur un deuxième écran.

La capture peut échouer sur les jeux en plein écran exclusif, les contenus protégés, certaines configurations HDR/multi-GPU ou des pilotes incompatibles.

## Architecture

\`\`\`
Écran source -> DXGI Desktop Duplication -> Copie GPU vers texture SRV
              -> D3D11 HLSL (netteté locale + contraste + saturation)
              -> swapchain DXGI -> fenêtre de prévisualisation
\`\`\`

Il n'y a **pas de lecture des pixels par le CPU**. Une copie GPU et une deuxième présentation restent nécessaires : elles consomment donc des ressources. La récupération \`DXGI_ERROR_ACCESS_LOST\` est prévue.

## Critères Go / No-go

- **V0.1 :** capture fluide et stabilité pendant des mouvements; différence visuelle perceptible.
- **V0.2 :** GPU timestamp queries, mesures de latence et benchmarks fiables; capture de comparatifs.
- **V0.3 :** test d'un modèle léger ONNX/DirectML avec estimation des artefacts temporels.

**Stop** si le résultat n'est qu'un preset comparable à ReShade, si les artefacts sont gênants ou si le jeu ne maintient plus les FPS ciblés.

## Confidentialité et sécurité

L'écran complet peut contenir des notifications privées : attention à ce qui apparaît dans la prévisualisation. Tout est traité **localement**, sans envoi réseau ni télémétrie. Pas de code injecté dans Fortnite ou dans un autre jeu.

Licence MIT : voir [LICENSE](LICENSE).
