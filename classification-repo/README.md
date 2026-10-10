# Student Dropout & Academic Success Prediction (Classification)

Multi-class classification of university students into **Dropout**, **Enrolled** or **Graduate**, comparing baseline models, ensemble methods and hyperparameter tuning.

> **Academic project** for the **Artificial Intelligence** module, Intelligent Information Systems Engineering (ISII), University of Algiers 1 – Benyoucef Benkhedda, 2025/2026 (supervisor: Dr. Saliha Mezzoudj).
> The project was done as a team and also included a regression task (student grade prediction), carried out by a teammate. **This repository contains only the classification part, which I implemented.**

## Dataset

[Predict Students' Dropout and Academic Success](https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success) (UCI Machine Learning Repository): 4,424 students, 36 features (demographics, admission data, 1st/2nd-semester academic performance, socio-economic indicators) and a 3-class target.

| Class | Samples |
|-------|--------:|
| Graduate | 2,209 (49.9%) |
| Dropout | 1,421 (32.1%) |
| Enrolled | 794 (17.9%) |

The raw file is not included. Download it from the link above (see *Getting started*).

## Methodology

1. **Data exploration and cleaning**: missing values, duplicates, outlier analysis
2. **Feature engineering**: average grade across both semesters; correlation and mutual-information analysis
3. **Preprocessing**: standardization (`StandardScaler`), class imbalance handled with **SMOTE**, 60 / 20 / 20 train / validation / test split
4. **Baseline models**: Logistic Regression, MLP
5. **Ensemble models**: tree-based and boosting methods, with LightGBM performing best
6. **Optimization**: `RandomizedSearchCV` with 5-fold cross-validation
7. **Evaluation**: accuracy, F1-score, ROC-AUC (one-vs-rest), precision-recall curves, error analysis

## Results

| Model | Accuracy | F1-score | ROC-AUC | Stage |
|-------|---------:|---------:|--------:|-------|
| Logistic Regression | 73% | 0.73 | 0.881 | Baseline |
| MLP | 81% | 0.82 | 0.914 | Baseline |
| LightGBM | 84.24% | 0.842 | 0.951 | Ensemble |
| **Optimized LightGBM** | **84.9%** | **0.849** | **0.95** | Tuned |

The final model detects graduates (91% recall) and dropouts (81% recall) well. *Enrolled* students are the hardest class to predict.

## Repository structure

```
├── classification.ipynb     # full pipeline: preprocessing, models, tuning, evaluation
├── plots/                   # figures (feature analysis, model comparison, PR curves, errors)
├── processed_data/          # preprocessed train / validation / test splits
├── requirements.txt
└── README.md
```

## Getting started

```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook classification.ipynb
```

- The notebook was developed in **Google Colab** and mounts Google Drive in its first cells (`/content/drive/...`). To run it locally, remove those cells and point the paths to your own folders.
- Download the dataset from the UCI link and place `data.csv` next to the notebook (it is read with `sep=';'`). Alternatively, skip preprocessing and load the ready-made files in `processed_data/`.

## Built with

Python, pandas, NumPy, scikit-learn, imbalanced-learn, XGBoost, LightGBM, CatBoost, matplotlib, seaborn.

## License

Educational project. The dataset belongs to its original authors; see the UCI page for its terms.
