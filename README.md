# AI-Laue Research Pipeline

**Autonomous AI research agent for computational photonics**  
*Inspired by AI-Mandel (arXiv:2511.11752)*

---

## Paper

**Machine-learning surrogate model for one-dimensional GaAs/Al₀.₃Ga₀.₇As distributed Bragg reflector spectra**

**Author:** Mehdi Ouslim — Independent Researcher, Algeria  
**Status:** Ready for arXiv submission (eess.SP / physics.optics)

### Abstract

We present a Gaussian Process (GP) surrogate model for the normal-incidence reflectance spectrum of one-dimensional GaAs/Al₀.₃Ga₀.₇As distributed Bragg reflectors (DBRs). A Latin-hypercube dataset of 1,500 transfer-matrix method (TMM) simulations spanning N_periods ∈ [5, 30], d_GaAs ∈ [50, 200] nm, and d_AlGaAs ∈ [50, 200] nm over the 900–1100 nm window is used to train and evaluate the model. Principal component analysis (PCA) reduces the 150-point spectral output to 26 components (≥ 99.9% variance retained); one GP is fitted per component. On a held-out test set (n = 150) the GP achieves RMSE = 0.085 and R² = 0.276, while a Random Forest (RF) baseline reaches RMSE = 0.065 and R² = 0.572. Uncertainty calibration shows that the GP 95% prediction band covers 98.9% of test residuals. These results establish a rapid surrogate for DBR design-space exploration and motivate further work on sparse GP formulations.

---

## Results

| Model | RMSE | R² |
|-------|------|-----|
| Gaussian Process | 0.085 | 0.276 |
| Random Forest | 0.065 | 0.572 |

- Dataset: 1,500 TMM simulations, validated against Palik 1985 refractive index data
- Spectral range: 900–1100 nm
- Parameter space: N_periods ∈ [5, 30], d_GaAs ∈ [50, 200] nm, d_AlGaAs ∈ [50, 200] nm

---

## Project Structure

```
ai-laue-research/
├── CLAUDE.md                    ← AI-Laue agent system prompt (Claude Code)
├── GUIDE.md                     ← Step-by-step project guide
├── EXPLICATIONS_TECHNIQUES.md  ← Technical explanations
├── idea_pool.md                 ← Research idea log
├── src/
│   ├── simulation.py            ← TMM dataset generation
│   ├── ml_model.py              ← GP + RF surrogate training
│   └── figures.py               ← Publication-quality figure generation
├── data/
│   └── dataset.npz              ← 1,500 TMM simulations (899 KB)
└── output/
    ├── figures/                 ← 5 publication figures (PDF + PNG)
    │   ├── fig1_parity.pdf
    │   ├── fig2_mae_heatmap.pdf
    │   ├── fig3_spectra.pdf
    │   ├── fig4_stopband.pdf
    │   └── fig5_learning_curve.pdf
    └── paper/
        └── main.tex             ← Full LaTeX paper (Optics Express format)
```

---

## Reproduce the Results

### 1. Install dependencies

```bash
pip install arxiv tmm numpy scipy matplotlib scikit-learn
```

### 2. Generate the dataset

```bash
python src/simulation.py
```

Generates `data/dataset.npz` — 1,500 TMM simulations (~2 minutes on a standard laptop).

### 3. Train the surrogate model

```bash
python src/ml_model.py
```

Trains GP and Random Forest surrogates, saves metrics to `output/metrics.json`.

### 4. Generate figures

```bash
python src/figures.py
```

Produces 5 publication-quality PDF figures in `output/figures/`.

---

## AI-Laue Agent

This paper was produced using **AI-Laue**, an autonomous research agent built on Claude Code, inspired by AI-Mandel (Arlt, Gu, Krenn 2025 — arXiv:2511.11752).

AI-Laue runs a 5-agent pipeline:
1. **Scout** — arXiv literature mining, gap identification
2. **Idea Generator** — hypothesis formulation
3. **Novelty Checker** — prior art verification
4. **Feasibility Judge** — 5-criterion scoring rubric
5. **Implementation Expert** — code, figures, LaTeX paper

The system prompt is available in `CLAUDE.md`.

---

## Dependencies

- Python 3.11
- `tmm` — Transfer Matrix Method for multilayer optics
- `numpy`, `scipy` — Scientific computing
- `scikit-learn` — GP and Random Forest models
- `matplotlib` — Publication figures
- `arxiv` — Literature search

---

## License

MIT License — free to use, modify, and distribute with attribution.

---

## Contact

**Mehdi Ouslim**  
Independent Researcher — Algeria  
Email: mehdi.ouslim@gmail.com  
GitHub: [mehdiouslim-hash](https://github.com/mehdiouslim-hash)

---

*This project demonstrates that rigorous, publishable computational physics research is accessible to independent researchers without institutional affiliation, using open-source tools and public databases.*
