# Bad Apple sur Shadertoy - Guide Complet

Projet de compression vidéo procédurale pour [Shadertoy.com](https://www.shadertoy.com) utilisant un **réseau de neurones Tiny** embarqué directement dans le code GLSL (pas de texture custom).

## 🎯 Concept

Bad Apple (480×360, 6572 frames, 1.08 GB) → Réseau de neurones Tiny (4353 paramètres, 17 KB) → Shader GLSL multi-pass sur Shadertoy

**Approche** : Un NN qui apprend la fonction `f(frame, x, y) → pixel_value` et reconstruit la vidéo en temps réel.

## 🚀 Quick Start

### 1. Installer les dépendances

```bash
uv sync
# ou
pip install -e .
```

### 2. Entraîner le réseau de neurones

```bash
python3 bad_apple/train_nn.py
```

**Durée** : 
- CPU: ~1-2 heures
- GPU: ~10-15 minutes

**Sortie** :
- `bad_apple/nn_weights_tiny.npz` - Poids du réseau
- `bad_apple/nn_weights_tiny_metadata.json` - Métadonnées

### 3. Générer le shader Shadertoy

```bash
python3 generate_shadertoy_multipass.py bad_apple/nn_weights_tiny.npz
```

**Sortie** :
- `bad_apple/shadertoy_buffer_a.glsl` - Buffer A (stockage des poids)
- `bad_apple/shadertoy_image.glsl` - Image (inférence NN)
- `bad_apple/SHADERTOY_SETUP.md` - Instructions détaillées

### 4. Upload sur Shadertoy

1. **Créer un nouveau shader** : https://www.shadertoy.com/new

2. **Ajouter Buffer A** :
   - Cliquer "+" → "Buf A"
   - Copier le contenu de `shadertoy_buffer_a.glsl`
   - Coller dans l'onglet "Buf A"

3. **Configurer Image** :
   - Aller dans l'onglet "Image"
   - Cliquer sur **iChannel0** → Sélectionner **"Buf A"**
   - Copier le contenu de `shadertoy_image.glsl`
   - Coller dans l'onglet "Image"

4. **Compiler** : Alt+Enter

5. 🎉 **La vidéo devrait jouer !**

## 📁 Structure du projet

```
shadertoys/
├── bad_apple/
│   ├── video.webm                      # Vidéo source (480×360, 6572 frames)
│   ├── video_pixels.parquet            # Pixels extraits (1.08 GB)
│   ├── train_nn.py                     # Entraînement NN Tiny
│   ├── nn_weights_tiny.npz             # Poids entraînés (généré)
│   ├── shadertoy_buffer_a.glsl         # Shader Buffer A (généré)
│   ├── shadertoy_image.glsl            # Shader Image (généré)
│   └── SHADERTOY_SETUP.md              # Instructions (généré)
├── generate_shadertoy_multipass.py     # Générateur shader multi-pass
├── extract_pixels.py                   # Extraction pixels → Parquet
├── pyproject.toml                      # Dépendances
└── README.md                           # Ce fichier
```

## 🧠 Architecture du réseau

**Tiny NN** : `[3] → [32] → [64] → [32] → [1]`

- **Input** : `(frame_norm, x_norm, y_norm)` ∈ [0, 1]³
- **Hidden** : 3 couches fully-connected avec ReLU
- **Output** : `pixel_value` ∈ [0, 1] via Sigmoid
- **Paramètres** : 4,353 (~17 KB)

**Training** :
- Sample rate : 5% des pixels (~57M pixels)
- Batch size : 8192
- Epochs : 30
- Loss : MSE
- Optimizer : Adam (lr=0.001)

## 📊 Résultats attendus

| Métrique | Valeur |
|----------|--------|
| **Compression** | ~60,000x (1.08 GB → 17 KB) |
| **Qualité** | ⭐⭐ Acceptable mais floue |
| **PSNR** | ~20-25 dB |
| **Performance** | 🐌 Lent (calcul per-pixel) |
| **Code size** | ~10-15K caractères GLSL |

**Qualité** : Le réseau est intentionnellement très petit pour tenir dans le code Shadertoy. La vidéo sera reconnaissable mais floue/pixelisée. C'est un proof-of-concept, pas une compression haute fidélité.

## ⚙️ Configuration avancée

### Augmenter la qualité (sacrifie la taille)

Éditer `bad_apple/train_nn.py` :

```python
architectures = [
    {
        "name": "Tiny",
        "hidden": [64, 128, 64],  # Plus gros réseau
        "sample_rate": 0.1,        # Plus de données
        "epochs": 50,              # Plus d'entraînement
        "batch_size": 4096,
    },
]
```

⚠️ **Attention** : Un réseau plus gros peut dépasser la limite de 65K caractères de Shadertoy !

### Tester sur quelques frames

Pour debug rapide, modifier l'extraction de données dans `train_nn.py` :

```python
# Filtrer seulement les 100 premières frames
df = df.filter(pl.col("frame") < 100)
```

## 🔬 Pipeline technique

### 1. Extraction des pixels

```python
# extract_pixels.py
video → OpenCV → Grayscale → Polars DataFrame → Parquet
```

### 2. Entraînement NN

```python
# train_nn.py
Parquet → Sample 5% → (frame, x, y, pixel) → PyTorch → Weights
```

### 3. Génération GLSL

```python
# generate_shadertoy_multipass.py
Weights NPZ → Linearize → GLSL array → Buffer A + Image shaders
```

### 4. Multi-pass Shadertoy

```glsl
// Buffer A: Encode weights as texture
const float NN_WEIGHTS[4353] = float[](...);
// Pack into RGBA pixels

// Image: NN forward pass
texelFetch(iChannel0, ...) → Weights → neuralNetwork(frame, x, y) → pixel
```

## 🐛 Troubleshooting

### Entraînement trop lent
- Réduire `sample_rate` à 0.02
- Réduire `epochs` à 10
- Utiliser GPU si disponible

### Shader ne compile pas
- Vérifier que Buffer A est bien connecté à iChannel0 dans Image
- Code trop gros ? Réduire la taille du réseau
- Erreur syntaxe GLSL ? Vérifier les tableaux constants

### Vidéo noire
- Buffer A connecté à iChannel0 ? ✓
- Compilation réussie ? ✓
- Vérifier les normalisations (frame/max, etc.)

### Qualité médiocre
- C'est normal ! Le NN est minuscule (4K params pour 1GB de données)
- Pour améliorer : augmenter taille réseau, sample_rate, epochs
- Trade-off : qualité ↔ taille du code

### Performance lente sur Shadertoy
- C'est attendu : calcul NN complet par pixel (480×360 = 172K forward passes)
- Impossible à optimiser sans changer l'approche
- Alternative : réduire résolution de sortie

## 💡 Améliorations possibles

### NN hybride - Prédire coefficients DCT
Au lieu de pixels directs, prédire les coefficients DCT par bloc :
- Input : `(frame, block_x, block_y)`
- Output : 10 coefficients DCT
- Avantages : structure plus compacte, meilleure qualité

### Downscaling
- Entraîner sur 240×180 au lieu de 480×360
- Upscale dans le shader (bilinear)
- 4x moins de calculs, qualité acceptable

### Frames clés + interpolation
- NN prédit 1 frame sur 10
- Interpolation linéaire entre frames
- 10x moins de variance temporelle à apprendre

## 📚 Ressources

- [Shadertoy.com](https://www.shadertoy.com) - Plateforme WebGL
- [Bad Apple Wikipedia](https://en.wikipedia.org/wiki/Bad_Apple!!) - Histoire
- [Neural Compression](https://arxiv.org/abs/2001.04451) - Recherche académique
- [SIREN](https://vsitzmann.github.io/siren/) - Implicit Neural Representations

## 🎨 Crédits

- **Bad Apple!!** © Alstroemeria Records / Touhou Project
- **Shadertoy** - Iñigo Quilez & Pol Jeremias
- **Concept** - Compression procédurale / Neural implicit functions

## 📝 Notes

**Pourquoi pas de texture custom ?** Shadertoy n'accepte que des textures preset. Cette contrainte force l'embarquement des données directement dans le code GLSL.

**Pourquoi multi-pass ?** Les tableaux GLSL constants ont des limites de taille. Utiliser Buffer A comme "texture de stockage" permet de contourner certaines contraintes.

**Pourquoi si petit ?** Shadertoy limite à ~65K caractères de code. Un array de 4353 floats ≈ 50K caractères, ce qui laisse de la place pour le code de décodage.

---

**Amusez-vous bien ! 🍎✨**
