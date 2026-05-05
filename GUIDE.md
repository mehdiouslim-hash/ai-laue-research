# GUIDE COMPLET — Projet AI-Laue

## De zéro à un preprint arXiv en photonique computationnelle

\---

## C'EST QUOI CE PROJET ?

Tu vas créer un **agent IA de recherche autonome** en physique/photonique,
inspiré du système AI-Mandel (arXiv:2511.11752).

L'agent s'appelle **AI-Laue**. Il va :

1. Chercher des gaps dans la littérature scientifique (arXiv)
2. Générer des idées de recherche nouvelles
3. Vérifier que l'idée n'est pas déjà publiée
4. Évaluer si c'est faisable sur un laptop
5. Écrire le code Python + les figures + le papier en LaTeX

**Objectif final** : un preprint publié sur arXiv sous ton nom,
qui prouve ta capacité à faire de la recherche indépendante.
Ça débloque tes candidatures PhD.

\---

## OUTILS UTILISÉS

|Outil|Rôle|Où l'obtenir|
|-|-|-|
|PowerShell|Terminal pour les commandes système|Déjà sur Windows|
|Git|Versionner ton code|git-scm.com|
|GitHub|Héberger ton projet en ligne|github.com|
|Python|Langage de programmation|python.org|
|Claude Code|Agent IA qui génère le code de recherche|Déjà installé|
|tmm|Simulation optique (Transfer Matrix Method)|pip install tmm|
|numpy / scipy|Calcul scientifique|pip install numpy scipy|
|matplotlib|Graphiques publication|pip install matplotlib|
|scikit-learn|Machine learning|pip install scikit-learn|
|arxiv|Chercher des papiers|pip install arxiv|

\---

## STRUCTURE DU PROJET

```
ai-laue-research/
│
├── CLAUDE.md           ← Le "cerveau" de l'agent AI-Laue
│                          (system prompt pour Claude Code)
│
├── GUIDE.md            ← Ce fichier — tu le gardes ouvert
│
├── idea\_pool.md        ← Journal de toutes les idées générées
│                          (créé automatiquement par l'agent)
│
├── src/                ← Tout le code Python
│   ├── simulation.py   ← Génération de données (TMM)
│   ├── ml\_model.py     ← Modèle ML surrogate
│   └── inverse\_design.py ← Design inverse
│
├── data/               ← Données générées (ignorées par git)
│   └── dataset.npz
│
└── output/
    ├── figures/        ← Figures pour le papier (PDF + PNG)
    └── paper/
        ├── main.tex    ← Papier LaTeX complet
        └── references.bib
```

\---

## ÉTAPES DU PROJET

\---

### ÉTAPE 1 — Préparer l'environnement

**Durée estimée : 20 minutes
Outil : PowerShell**

#### 1.1 — Vérifier que Python est installé

```powershell
python --version
```

Tu dois voir quelque chose comme `Python 3.11.x`
Si erreur → va sur python.org et installe Python

#### 1.2 — Vérifier que Git est installé

```powershell
git --version
```

Tu dois voir `git version 2.x.x`
Si erreur → va sur git-scm.com et installe Git

#### 1.3 — Créer le dossier projet

```powershell
mkdir ai-laue-research
cd ai-laue-research
```

#### 1.4 — Initialiser Git

```powershell
git init
```

Tu dois voir : `Initialized empty Git repository`

#### 1.5 — Créer la structure de dossiers

```powershell
mkdir src
mkdir data
mkdir output
mkdir output\\figures
mkdir output\\paper
```

#### 1.6 — Installer les packages Python

```powershell
pip install arxiv tmm numpy scipy matplotlib scikit-learn
```

Attends que tout s'installe (2-5 minutes)

#### 1.7 — Créer le fichier .gitignore

Crée un fichier `.gitignore` à la racine avec ce contenu :

```
data/\*.npz
data/\*.csv
\_\_pycache\_\_/
\*.pyc
.env
\*.egg-info/
.DS\_Store
```

#### ✅ Vérification étape 1

```powershell
ls
```

Tu dois voir : `src/  data/  output/  .gitignore`

\---

### ÉTAPE 2 — Créer le repo GitHub

**Durée estimée : 10 minutes
Outil : github.com + PowerShell**

#### 2.1 — Créer un compte GitHub

Va sur github.com → Sign up (si pas déjà fait)

#### 2.2 — Créer un nouveau repo

* Clique sur "New repository"
* Nom : `ai-laue-research`
* Description : `Autonomous AI research agent for computational photonics`
* Visibilité : **Public** (important pour la crédibilité scientifique)
* Ne coche PAS "Add README" (on le fera manuellement)
* Clique "Create repository"

#### 2.3 — Connecter ton dossier local à GitHub

GitHub t'affiche des commandes. Dans PowerShell :

```powershell
git remote add origin https://github.com/TON\_USERNAME/ai-laue-research.git
git branch -M main
```

#### 2.4 — Configurer ton identité Git (si première fois)

```powershell
git config --global user.name "Ton Nom"
git config --global user.email "ton@email.com"
```

#### 2.5 — Déposer les fichiers CLAUDE.md et GUIDE.md

Place les deux fichiers téléchargés à la racine du dossier.
Puis :

```powershell
git add .
git commit -m "init: project structure + AI-Laue system prompt"
git push -u origin main
```

#### ✅ Vérification étape 2

Va sur github.com/TON\_USERNAME/ai-laue-research
Tu dois voir tes fichiers en ligne.

\---

### ÉTAPE 3 — Premier lancement de Claude Code

**Durée estimée : 30 minutes
Outil : Claude Code**

#### 3.1 — Ouvrir Claude Code dans le bon dossier

```powershell
cd ai-laue-research
claude
```

#### 3.2 — Lancer AI-Laue en mode guidé

Tape exactement dans Claude Code :

```
AI-Laue, begin research cycle.
Topic: "ML surrogate model for 1D GaAs photonic crystal reflectance"
Constraint: laptop-only, Python only, TMM simulation, no HPC.
Target: Optics Express preprint within 3 weeks.
Start with SCOUT AGENT only. List gaps found. Stop and wait.
```

#### 3.3 — Valider les gaps trouvés

L'agent va te proposer 3-5 gaps dans la littérature.
Lis-les et choisis celui qui te semble le plus intéressant.
Réponds : `Proceed with GAP\_00X`

#### 3.4 — Laisser tourner jusqu'à l'Agent 3

L'agent va générer une idée (Agent 2) puis vérifier la nouveauté (Agent 3).
**STOP obligatoire après Agent 3** — vérifie le verdict NOVEL avec moi
avant de continuer.

#### ✅ Vérification étape 3

Tu as un fichier `idea\_pool.md` créé avec au moins une idée NOVEL.

\---

### ÉTAPE 4 — Génération des données de simulation

**Durée estimée : 1-2 heures
Outil : Claude Code + Python**

#### 4.1 — Demander à Claude Code de générer simulation.py

```
AI-Laue, Agent 5 Phase A.
Generate src/simulation.py using TMM for 1D GaAs/AlGaAs photonic crystal.
Parameter space: layer thickness 50-200nm, number of periods 5-15.
Output: dataset of 500 samples (params → reflectance spectrum 800-1100nm).
Save to data/dataset.npz
```

#### 4.2 — Exécuter la simulation

```powershell
python src/simulation.py
```

Durée : 5-15 minutes selon le nombre de samples.

#### 4.3 — Vérifier les données

```powershell
python -c "import numpy as np; d=np.load('data/dataset.npz'); print(d.files); print(d\['params'].shape); print(d\['spectra'].shape)"
```

Tu dois voir quelque chose comme `(500, 3)` et `(500, 100)`

#### ✅ Vérification étape 4

Le fichier `data/dataset.npz` existe et contient des données valides.

\---

### ÉTAPE 5 — Entraînement du modèle ML

**Durée estimée : 1-2 heures
Outil : Claude Code + Python**

#### 5.1 — Demander à Claude Code de générer ml\_model.py

```
AI-Laue, Agent 5 Phase B.
Generate src/ml\_model.py
Train a Gaussian Process surrogate on data/dataset.npz
Include: train/test split 80/20, RMSE metric, R² score,
learning curve, uncertainty quantification.
Save model predictions and metrics to output/
```

#### 5.2 — Lancer l'entraînement

```powershell
python src/ml\_model.py
```

#### 5.3 — Interpréter les résultats

Cherche dans la sortie :

* RMSE < 5% → excellent, publiable
* RMSE 5-10% → acceptable, discutable
* RMSE > 10% → problème, ajuster le modèle

#### ✅ Vérification étape 5

Tu as des métriques et un modèle entraîné. Résultat interprétable.

\---

### ÉTAPE 6 — Génération des figures

**Durée estimée : 1 heure
Outil : Claude Code + Python**

#### 6.1 — Demander les figures

```
AI-Laue, Agent 5 Phase C.
Generate all publication figures in output/figures/
Figure 1: Structure schematic + ML pipeline
Figure 2: Dataset parameter space
Figure 3: Parity plot predicted vs simulated
Figure 4: Error map across parameter space
Figure 5: Learning curve RMSE vs N samples
All figures: 300 DPI, no title, proper axis labels with units, PDF format.
```

#### 6.2 — Vérifier les figures

```powershell
ls output\\figures\\
```

Tu dois voir : `fig1.pdf fig2.pdf fig3.pdf fig4.pdf fig5.pdf`

#### ✅ Vérification étape 6

5 figures PDF de qualité publication dans output/figures/

\---

### ÉTAPE 7 — Rédaction du papier LaTeX

**Durée estimée : 3-5 heures
Outil : Claude Code + éditeur LaTeX (Overleaf recommandé)**

#### 7.1 — Demander le squelette LaTeX

```
AI-Laue, Agent 5 Phase D.
Generate output/paper/main.tex
Format: Optics Express (optica-article class)
Include all sections with detailed placeholders.
Abstract already written from results.
All figure references included.
```

#### 7.2 — Compiler sur Overleaf

* Va sur overleaf.com (gratuit)
* New Project → Upload → dépose main.tex + figures
* Compile et vérifie le rendu

#### 7.3 — Remplir les sections

Sections à rédiger toi-même (Claude Code t'aide) :

* Introduction : contexte + gap + contribution
* Methods : structure + simulation + ML
* Results : résultats avec références aux figures
* Discussion : comparaison littérature + limites
* Conclusion : 3 claims + perspectives

#### ✅ Vérification étape 7

Un PDF compilé, lisible, avec toutes les sections et figures.

\---

### ÉTAPE 8 — Soumission arXiv

**Durée estimée : 1 heure
Outil : arxiv.org**

#### 8.1 — Créer un compte arXiv

Va sur arxiv.org → Register

#### 8.2 — Checklist avant soumission

* \[ ] Abstract < 150 mots
* \[ ] Toutes les figures en PDF ou EPS
* \[ ] Références complètes dans .bib
* \[ ] Affiliations : "Independent Researcher" si pas d'institution
* \[ ] Catégories : physics.optics + cond-mat.mtrl-sci
* \[ ] Code sur GitHub (ajoute le lien dans le papier)

#### 8.3 — Soumettre

* arxiv.org → Submit
* Upload main.tex + figures + references.bib
* Prévisualise le PDF final
* Confirme la soumission

#### 8.4 — Partager

Une fois l'arXiv ID reçu (ex: arXiv:2506.XXXXX) :

* Ajoute-le dans ton CV
* Mets le lien dans tes prochaines candidatures PhD
* Partage sur LinkedIn / Twitter

#### ✅ Vérification étape 8

Tu as un arXiv ID. Ton papier est public et citable.

\---

### ÉTAPE 9 — Git final et repo propre

**Durée estimée : 15 minutes**

```powershell
git add src/ output/paper/ output/figures/ idea\_pool.md
git commit -m "feat: complete research pipeline + paper draft"
git push
```

Ajoute un README.md propre sur GitHub avec :

* Titre du papier
* Lien arXiv
* Instructions pour reproduire les résultats

\---

## RÉSUMÉ DES DURÉES

|Étape|Durée|Outil|
|-|-|-|
|1. Environnement|20 min|PowerShell|
|2. GitHub|10 min|github.com|
|3. Premier lancement|30 min|Claude Code|
|4. Simulation|2h|Claude Code + Python|
|5. ML model|2h|Claude Code + Python|
|6. Figures|1h|Claude Code + Python|
|7. Rédaction LaTeX|5h|Claude Code + Overleaf|
|8. Soumission arXiv|1h|arxiv.org|
|**TOTAL**|**\~12h sur 2-3 semaines**||

\---

## EN CAS DE PROBLÈME

**Erreur pip install** → essaie `pip install --user \[package]`

**Erreur git push** → vérifie que tu es bien authentifié :

```powershell
git config --global credential.helper manager
```

**Claude Code ne lit pas CLAUDE.md** → vérifie que le fichier
est exactement à la racine du dossier où tu lances `claude`

**Modèle ML mauvais RMSE** → augmente le dataset (500 → 1000 samples)
ou change de modèle (GP → MLP)

**LaTeX ne compile pas** → colle l'erreur dans Claude Code,
il corrige automatiquement

\---

## RESSOURCES UTILES

* arXiv cond-mat : arxiv.org/list/cond-mat.mtrl-sci/recent
* arXiv physics.optics : arxiv.org/list/physics.optics/recent
* Materials Project API : materialsproject.org/api
* Overleaf (LaTeX en ligne) : overleaf.com
* Optics Express author guide : opg.optica.org/oe/submit
* tmm documentation : github.com/sbyrnes321/tmm

\---

*Guide rédigé pour le projet AI-Laue v2.0
Objectif : premier preprint arXiv en 3 semaines*

