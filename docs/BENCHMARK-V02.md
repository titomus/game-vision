# V0.2 : protocole de benchmark graphique

Objectif : tester GameVision à 144 FPS **sans confondre** FPS du jeu, cadence de capture, images présentées et coût du shader.

## Lancer

Dans le dossier contenant GameVision.exe :

\`\`\`powershell
.\GameVision.exe --monitor 0 --csv benchmark.csv
\`\`\`

L'application écrit un échantillon approximativement chaque seconde.

| Champ CSV | Description | Limite |
|---|---|---|
| elapsed_s | Temps depuis l'ouverture du journal | Chronomètre CPU |
| capture_fps | Boucles de rendu ayant présenté une nouvelle capture par seconde | **Pas les FPS du jeu** |
| gpu_shader_ms | Durée de la commande Draw mesurée en timestamps GPU D3D11 | Hors copie GPU, capture, présentation |
| capture_wait_ms | Temps CPU passé dans AcquireNextFrame | **Pas la latence complète** |
| cpu_submit_ms | Temps CPU entre préparation des ressources et Draw | **Pas le temps GPU** |
| gpu_queries_completed | Requêtes valides résolues | La récupération est asynchrone |
| gpu_queries_dropped | Requêtes non validées ou anneau occupé | Surveiller les pertes |
| preset/power/split | Réglages au moment de la mesure | Preset 0 = sans effet |

Les GPU timestamps sont lus avec \`D3D11_ASYNC_GETDATA_DONOTFLUSH\` pour éviter d'attendre le GPU. La résolution peut prendre plusieurs images. Une valeur négative indique **pas encore de mesure valide**. Les timestamps eux-mêmes sont disponibles sur les GPU Direct3D 11 disposant de cette fonctionnalité.

## Test reproductible sur RTX 3050

1. Fermer les applications qui utilisent fortement le GPU. Noter la résolution, le mode d'affichage, les réglages du jeu et les drivers.
2. Démarrer une séquence répétable sur un jeu **solo en fenêtre sans bordure**. Mesurer FPS/frametimes du **jeu** avec son compteur ou un logiciel spécialisé indépendant, sans GameVision (base).
3. Lancer GameVision sur le même écran ou un deuxième écran, avec \`--csv\`, profil 1 puis profils 2 à 4, 60 secondes chacun.
4. Noter FPS moyen et 1 % low du jeu, gpu_shader_ms, capture_fps, saccades, latence ressentie et artefacts.
5. Tester sous mouvement rapide et lumières difficiles. Comparer visuellement les profils.
6. Ne jamais comparer deux séquences de jeu différentes comme si les résultats étaient strictement identiques.

**Attention :** la fenêtre de prévisualisation ne remplace pas le jeu en plein écran. Ce projet ne permet pas de jouer *dans* l'image modifiée à ce stade. C'est une étape de faisabilité de shader GPU.

## Étape suivante : prototype intégré au moteur du jeu

L'architecture visée est un plugin pour un moteur de rendu de test compatible, sans injection arbitraire :

\`\`\`
Moteur de jeu / testbed autorisant les plugins
   |
   +-- Couleur (Texture2D) ----------+
   +-- Profondeur (optionnel) -------+--> GameVision RenderPass
   +-- Mouvement (optionnel) --------+       |
   +-- Interface HUD (si séparée) ---+       +-- traitement spatial / temporel
                                            +-- sortie vers swapchain du jeu
\`\`\`

Seul le pipeline **couleur** existe actuellement en exécutable, dans le backend de prévisualisation DXGI. La profondeur, les motion vectors, le rendu neuronal et une intégration dans un moteur de jeu sont des **prochaines tâches**, non des fonctionnalités livrées.

L'évaluation neuronale devra utiliser des poids à provenance et licence vérifiables. Pas de modèle fermé/fuité redistribué.
