# Iris Flower Classification (Streamlit + SVM)

**Iris Flower Classification** is a simple machine learning demo that trains a Support Vector Machine (SVM) on the classic Iris dataset and exposes an interactive Streamlit app to make predictions from user-provided flower measurements.

---

## Project Overview

- **Goal:** Train an SVM classifier on the Iris dataset and provide a web-based UI to predict Iris species from sepal/petal measurements.
- **Main file:** `Iris.py` (loads data, trains the model, evaluates it, and serves a Streamlit UI)
- **Model:** scikit-learn `SVC` (RBF kernel)

---

## Features

- Trains and evaluates an SVM classifier (prints accuracy and classification report).
- Shows a confusion matrix heatmap (Matplotlib + Seaborn).
- Interactive Streamlit UI with sliders to input sepal/petal measurements and display predictions.

---

## Requirements

- Python 3.8+
- Packages:
  - streamlit
  - scikit-learn
  - pandas
  - numpy
  - seaborn
  - matplotlib

You can install them with:

```bash
pip install streamlit scikit-learn pandas numpy seaborn matplotlib
```

(Or add a `requirements.txt` and run `pip install -r requirements.txt`.)

---

## How to Run

1. Install dependencies (see above).
2. In the project directory, run:

```bash
streamlit run Iris.py
```

3. The Streamlit app will open in your browser. Use the sliders to provide measurements; the app shows the predicted species, model accuracy, and the confusion matrix.

---

## Results

- The code prints model accuracy and the classification report to the console. In the current run, the model accuracy is approximately **0.9667** (as shown in the app).

---

## Code Notes

- `Iris.py` loads `sklearn.datasets.load_iris()` and splits the data using `train_test_split(..., stratify=y)`.
- Standardization is applied via `StandardScaler()`.
- Model: `SVC(kernel='rbf', C=1.0, gamma='scale')`.
- The saved confusion matrix plot is shown inside the Streamlit page using `st.pyplot()`.

---

## Suggestions / Next steps

- Save the trained model to disk (e.g., with `joblib`) so the app does not need to re-train on each run.
- Add a `requirements.txt` and CI checks.
- Add unit or integration tests for the preprocessing and prediction pipeline.

---

## License

Add a license of your choice (e.g., MIT) or keep this repository as-is.

---
