# Étude technique : DLSS5oneclick et évolution GameVision

Analyse du dépôt public https://github.com/faisalkindi/DLSS5oneclick (consulté le 16 novembre 2026).

## Ce que ce projet fait réellement

DLSS5oneclick (Rust, eframe/egui) est un **orchestrateur d'installation et de diagnostic**, pas un nouveau moteur neuronal. Il détecte les jeux et leurs API graphiques, sélectionne des intégrations tierces (ReShade, RenoDX DLSS 5 add-on, DLSS5-Feeder, LumeniteFX, OptiScaler et variantes), installe des DLL dans le répertoire du jeu, maintient des manifests, propose des fallbacks et des procédures de suppression. Source : README, `src/game.rs`, `src/installer.rs`, `src/quality_preset.rs`, `src/ngx.rs`.

### Chemins d'intégration

- Jeux intégrant NGX/DLSS : essayer de raccorder le traitement à leur pipeline DLSS, en récupérant les données utiles avant ou après l'upscaling.
- Jeux sans DLSS : ReShade + Feeder + informations de mouvement approximées, avec des risques plus élevés d'artefacts.
- DX11 et DX12 requièrent souvent des solutions spécifiques ; Vulkan suit une voie différente.
- Le passage neuronal à la résolution de rendu avant upscaling réduit le nombre de pixels, mais peut modifier le rendu et la stabilité temporelle.
- Les profils « low / medium / high » modifient notamment résolution de travail, sharpen, optical flow et masques.

### Points de vigilance

1. Le README présente le modèle utilisé comme une **fuite fermée et redistribuée par la communauté**. Ne pas importer automatiquement ces binaires dans GameVision ni les redistribuer sans clarification de provenance, autorisation et sécurité.
2. La compatibilité annoncée RTX 20/30 passe par des variantes modifiées en FP16 ; elle ne démontre pas 144 FPS sur RTX 3050.
3. Le travail à 75 % ou 50 % de résolution réduit le nombre de pixels à traiter (~56,25 % ou 25 %), **pas nécessairement** le coût GPU total dans les mêmes proportions.
4. Des problèmes de scintillement, d'évaluation répétée du modèle et de crash sont décrits dans la documentation, notamment sur Cyberpunk 2077.
5. Les modifications du pipeline des jeux en ligne peuvent déclencher leurs anticheats. Ne pas ajouter de mode de contournement ; tester uniquement sur titres solo autorisant les mods.
6. DLSS5oneclick indique que certains modules ont seulement fait l'objet d'un contrôle d'installation, pas d'un test en jeu : ne pas confondre test de fichiers et validation de fonctionnement.

## Comparatif avec GameVision V0.1

| Critère | GameVision V0.1 | DLSS5oneclick |
|---|---|---|
| Capture/interception | DXGI Desktop Duplication, après composition | Injection de DLL/proxy / add-ons dans pipeline jeu |
| Accès profondeur et mouvement | Non | Possiblement, selon jeu/intégration |
| Amélioration | Shaders HLSL légers | Réseau neuronal tiers |
| Fonctionnement universaliste | Bureau Windows compatible | Dépend fortement du jeu, de ses API et DLL |
| Coût GPU | À mesurer | Potentiellement élevé |
| Anticheat | Pas de modification du jeu | Risque explicite et refus des jeux protégés |
| Statut | Prototype compilé, non encore validé en jeu | Orchestrateur avancé avec compatibilité variable |

## Feuille de route proposée : GameVision V0.2 et V0.3

### P0 : instrumentation fiable

- Timestamp queries GPU, temps par frame, FPS réellement produits par le jeu vs capture/présentation.
- Détection de répétitions d'image et comparaison de stabilité temporelle.
- Captures synchronisées A/B et protocole de benchmark reproductible.
- Règle : ne jamais présenter CPU submission time comme GPU processing time.

### P1 : prototype *in-engine* facultatif et responsable

- **Conserver** le chemin Desktop Duplication pour les applications non compatibles.
- Ajouter une architecture backend `PreviewCapture` / `GameRenderPlugin`.
- Développer un backend plugin **expérimental** uniquement pour un moteur/testbed open source ou un jeu solo avec support officiel de mods.
- Récupérer, lorsque disponible, couleur, profondeur et motion vectors, avec détection explicite des ressources réellement exposées.
- Aucune injection sur les jeux protégés, aucun désarmement ou contournement d'anticheat.

### P2 : neural rendering reproductible

- Évaluer uniquement des modèles aux poids et licences clairement autorisés (ONNX Runtime DirectML, WinML, etc.).
- Tester à 50/75/100 % résolution de travail, isoler le coût GPU, surveiller la VRAM.
- Recomposer avec upscale et stabilité temporelle ; protéger HUD, texte et éléments d'interface.
- Mesurer artefacts, ghosting, stabilité d'image et latence en mouvement.
- Test de non-régression : désactiver le traitement lorsqu'il coûte trop cher ou rend l'image moins fidèle.

### Go / No-go

Cible expérimentale : amélioration nette visible à qualité/latence quantifiées, puis tentative de 144 images rendues par seconde. En cas d'impossibilité sur RTX 3050, rapporter honnêtement le compromis. **Ne pas confondre 144 FPS affichés par interpolation avec 144 FPS réellement rendus.**

## Références

- https://github.com/faisalkindi/DLSS5oneclick
- https://github.com/jlrouzies-fr/DLSS5-Feeder
- https://github.com/clshortfuse/renodx
- https://github.com/Dagherbou/OptiScaler_DLSSNR
- https://github.com/crosire/reshade-shaders
