# GameVision Neural Lab V0.3 : benchmark de véritable inférence IA

Le nouveau labo contient **un véritable petit réseau neuronal PyTorch**, pas un filtre shader présenté comme de l'IA. Il apprend à supprimer du bruit dans des images, puis mesure son coût à 540p, 720p et 1080p avec des timestamps CUDA sur NVIDIA.

**Ce n'est pas du DLSS 5**, ni une IA qui réinvente les matériaux d'un jeu. Cette étape répond à une question plus modeste et vérifiable : « Combien de millisecondes coûte un réseau neuronal léger sur ma RTX 3050 ? »

## Installation sur Windows (PowerShell, dans le repo)

Prérequis : Python 3.11+, pilote NVIDIA compatible CUDA et Git.

\`\`\`powershell
git clone https://github.com/titomus/game-vision.git
cd game-vision

py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe -m pip install numpy pillow

.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO CUDA')"
\`\`\`

Pour les versions CUDA proposées officiellement et compatibles avec ton installation, voir [PyTorch : Get Started](https://pytorch.org/get-started/locally/). Le dépôt ne remplace pas le choix de la roue CUDA adapté au pilote. Si \`torch.cuda.is_available()\` vaut \`False\`, **ne prétends pas mesurer la RTX** : corrige le pilote / la version PyTorch d'abord.

### 1. Entraîner le petit modèle

\`\`\`powershell
.\.venv\Scripts\python.exe -m neural.experiment train --steps 250 --device cuda
\`\`\`

Sans \`--images\`, l'entraînement utilise des images procédurales créées localement, avec du bruit artificiel. Les poids sont enregistrés dans \`models/tiny_denoiser.pt\`.

Pour adapter le débruitage à tes captures de jeu, place des images PNG/JPG dans un répertoire (captures dont tu as les droits) :

\`\`\`powershell
.\.venv\Scripts\python.exe -m neural.experiment train --images "C:\CapturesGameVision" --steps 500 --device cuda
\`\`\`

Il **n'apprend pas automatiquement le rendu photoréaliste** : il apprend uniquement à enlever le bruit simulé à partir de ces images. Ne pas utiliser de données ou de modèles propriétaires non autorisés.

### 2. Chronométrer l'inférence GPU

\`\`\`powershell
.\.venv\Scripts\python.exe -m neural.experiment bench --device cuda --precision fp16
\`\`\`

Neuf combinaisons sont testées : 960×540, 1280×720, 1920×1080 avec échelles internes 100 %, 75 % et 50 %. Résultats :

- \`results/neural_bench.json\` : contexte de mesure et détails.
- \`results/neural_bench.csv\` : mesures par configuration.
- \`gpu_kernel_ms_median\` : temps GPU mesuré par événements CUDA (réseau, redimensionnements, clamp), pas un simple chronomètre CPU.
- \`per_call_ms_p95\` / \`p99\` : variabilité de latence.
- \`wall_throughput_fps\` : débit mesuré depuis Python, synchronisation incluse.
- \`peak_torch_allocated_mib\` : pic d'allocations PyTorch **uniquement**, pas la VRAM totale du jeu.

Mesurer seulement le réseau non entraîné pour isoler le coût d'architecture (mode explicite) :

\`\`\`powershell
.\.venv\Scripts\python.exe -m neural.experiment bench --untrained --device cuda --precision fp16 --resolutions 1920x1080 --scales 1.0
\`\`\`

Cela ne mesure **aucune amélioration visuelle**.

### 3. Voir l'image avant / après

\`\`\`powershell
.\.venv\Scripts\python.exe -m neural.experiment preview --device cuda --image "C:\CapturesGameVision\scene.png" --noise 0.07
\`\`\`

\`results/preview/compare.png\` place l'entrée bruyante et la sortie débruitée côte à côte. Le bruit est ajouté volontairement pour rendre le test mesurable ; \`--noise 0\` utilise l'image réelle sans dégradation artificielle. Les résultats peuvent être médiocres : un modèle minuscule entraîné sur images synthétiques ne garantit ni qualité ni généralisation.

Pour tout lancer sur un PC sans CUDA en mode démonstration, préciser \`--device cpu --precision fp32\`. Un résultat CPU ne valide **pas** la faisabilité RTX 3050.

## Critères décisionnels

À 144 FPS, chaque image dispose de **6,94 ms au total**, rendu du jeu + IA + présentation. Une inférence IA à 6 ms ne suffit donc pas si le jeu consomme déjà 5 ms.

| Résultat à 1080p | Lecture |
|---|---|
| Moins de 1 ms médiane | Piste de calcul légère à examiner avec un vrai moteur ; pas une garantie 144 FPS |
| Entre 1 et 4 ms | Compromis GPU et résolution à étudier |
| Plus de 4 ms | Risque élevé de dépasser le budget du jeu à 144 FPS |
| Image sans gain perceptible | Pas de raison d'intégrer ce modèle dans le jeu |

### Ce que ce benchmark ne mesure PAS

- GPU en concurrence avec un jeu en cours
- Le coût des captures, des transferts mémoire ou d'une swapchain réelle
- Les données de profondeur / vecteurs de mouvement
- La cohérence temporelle sur une séquence vidéo
- Les résultats d'un modèle de neural rendering de type DLSS
- Les performances **réelles** de ton PC tant que tu n'as pas exécuté le programme dessus

## Évolution vers le rendu du jeu

Si et seulement si les mesures de GPU et les images sont convaincantes :

1. Exposer le traitement dans un plugin de moteur **avec accès autorisé aux textures**.
2. Ajouter couleur, profondeur, mouvements, masquage HUD, sans interception de jeux protégés.
3. Exporter le modèle en ONNX ou TorchScript / adapter un backend de runtime GPU pour l'hôte.
4. Tester stabilité temporelle, latence de bout en bout et FPS du jeu.

Le code du banc d'essai est dans [neural/experiment.py](../neural/experiment.py).
