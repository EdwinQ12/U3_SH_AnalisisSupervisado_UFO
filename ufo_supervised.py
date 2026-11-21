# -*- coding: utf-8 -*-
"""
U3 Saber Hacer - Análisis supervisado con el dataset UFO.
Clasificación: predecir cobertura mediática (Media = Y) de un avistamiento.
"""
import json
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, "resultados_ufo_sup")
os.makedirs(OUT, exist_ok=True)
plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = 9
GREEN = "#1F3B2E"
GREEN2 = "#2E5D45"
GOLD = "#C9A227"
RED = "#A23B4A"
BLUE = "#2F6B8A"

# ----------------------------------------------------------------------
# Datos
# ----------------------------------------------------------------------
raw = pd.read_csv(os.path.join(BASE, "ufo_repo", "ufo_data.csv"))
df = raw.copy()
df["fecha"] = pd.to_datetime(df["Date"], format="%m/%d/%Y %H:%M", errors="coerce")
df = df.dropna(subset=["fecha", "Summary"])
df = df[(df["fecha"].dt.year >= 1940) & (df["fecha"].dt.year <= 2024)].copy()
df["year"] = df["fecha"].dt.year
df["month"] = df["fecha"].dt.month
df["hour"] = df["fecha"].dt.hour
df["weekday"] = df["fecha"].dt.weekday
df["media"] = (df["Media"] == "Y").astype(int)

shape_map = {
    "Triangle": "Geometric", "Chevron": "Geometric", "Formation": "Geometric",
    "Diamond": "Geometric", "Rectangle": "Geometric", "Cigar": "Geometric",
    "Cylinder": "Geometric", "Teardrop": "Geometric", "Cross": "Geometric",
    "Disk": "Circular", "Circle": "Circular", "Sphere": "Circular",
    "Oval": "Circular", "Round": "Circular",
    "Light": "Luminous", "Orb": "Luminous", "Fireball": "Luminous",
}
df["shape_group"] = df["Shape"].map(shape_map).fillna("Other/Unknown")
country_map = {"USA": "USA", "Canada": "Canada", "United Kingdom": "UK"}
df["country_group"] = df["Country"].map(country_map).fillna("Other/Unknown")
top_states = df["State"].value_counts().head(12).index
df["state_group"] = df["State"].where(df["State"].isin(top_states), "Other/Unknown")
df["state_group"] = df["state_group"].fillna("Unknown")

s = df["Summary"].str.lower()
df["summary_len"] = s.str.len()
df["word_count"] = s.str.split().str.len()
keywords = {
    "light": r"light|lights|glow|bright|flash",
    "sound": r"sound|noise|hum|silent|silence",
    "speed": r"speed|fast|accelerat|hover|stationary",
    "triangle": r"triangle|triangular|chevron|delta",
    "disk": r"disk|disc|saucer|circular|sphere",
    "multiple": r"multiple|several|group|formation|objects",
    "military": r"military|air force|army|navy|pilot|radar",
    "photo": r"photo|photograph|camera|picture|video",
    "entity": r"entity|being|creature|occupant|alien",
    "night": r"night|dark|evening|midnight",
}
for name, pat in keywords.items():
    df[f"kw_{name}"] = s.str.contains(pat, regex=True, na=False).astype(int)

feature_cols = ["year", "month", "hour", "weekday", "summary_len", "word_count"] + [f"kw_{k}" for k in keywords]
for col in ["shape_group", "country_group", "state_group"]:
    dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
    df = pd.concat([df, dummies], axis=1)
    feature_cols += list(dummies.columns)

X = df[feature_cols].values.astype(float)
y = df["media"].values.astype(int)

clean = df[["Date", "City", "State", "Country", "Shape", "Summary", "Media", "media", "year", "month", "hour", "weekday", "shape_group", "country_group", "state_group", "summary_len", "word_count"] + [f"kw_{k}" for k in keywords]].copy()
clean.to_csv(os.path.join(BASE, "ufo_clean.csv"), index=False, encoding="utf-8")

# ----------------------------------------------------------------------
# Utilidades
# ----------------------------------------------------------------------
def stratified_split(X, y, test_size=0.30, seed=42):
    rng = np.random.default_rng(seed)
    tr, te = [], []
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        n = max(1, int(round(len(idx) * test_size)))
        te.extend(idx[:n]); tr.extend(idx[n:])
    return X[np.array(sorted(tr))], X[np.array(sorted(te))], y[np.array(sorted(tr))], y[np.array(sorted(te))]

def standardize(a, b):
    mu = a.mean(0); sd = a.std(0); sd[sd == 0] = 1
    return (a - mu) / sd, (b - mu) / sd

def metrics(y, p):
    tp = int(np.sum((y == 1) & (p == 1))); tn = int(np.sum((y == 0) & (p == 0)))
    fp = int(np.sum((y == 0) & (p == 1))); fn = int(np.sum((y == 1) & (p == 0)))
    acc = (tp + tn) / max(1, len(y))
    prec = tp / (tp + fp) if tp + fp else 0
    rec = tp / (tp + fn) if tp + fn else 0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0
    return {"accuracy": float(acc), "precision": float(prec), "recall": float(rec),
            "f1": float(f1), "tp": tp, "tn": tn, "fp": fp, "fn": fn}

def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

def logistic(Xtr, ytr, Xte, lr=0.2, epochs=3000, l2=0.01):
    Xb = np.c_[np.ones(len(Xtr)), Xtr]; Xteb = np.c_[np.ones(len(Xte)), Xte]
    w = np.zeros(Xb.shape[1])
    for _ in range(epochs):
        p = sigmoid(Xb @ w)
        w -= lr * (Xb.T @ (p - ytr) / len(Xb) + l2 * np.r_[0, w[1:]])
    return sigmoid(Xteb @ w)

def knn(Xtr, ytr, Xte, k=15):
    d = ((Xte[:, None, :] - Xtr[None, :, :]) ** 2).sum(2)
    idx = np.argsort(d, axis=1)[:, :k]
    return np.array([np.bincount(ytr[row], minlength=2).argmax() for row in idx])

def gini(y):
    if len(y) == 0: return 0
    p = np.bincount(y, minlength=2) / len(y)
    return 1 - np.sum(p ** 2)

def best_split(X, y, min_samples, features, n_th=6):
    best = None; best_score = np.inf
    for f in features:
        for t in np.unique(np.quantile(X[:, f], np.linspace(0.1, 0.9, n_th))):
            left = X[:, f] <= t; right = ~left
            if left.sum() < min_samples or right.sum() < min_samples: continue
            score = (left.sum() * gini(y[left]) + right.sum() * gini(y[right])) / len(y)
            if score < best_score:
                best_score = score; best = (f, float(t), left, right)
    return best

def build_tree(X, y, depth, max_depth, min_samples, rng, max_features):
    if depth >= max_depth or len(y) < 2 * min_samples:
        probs = np.bincount(y, minlength=2) / len(y)
        return {"leaf": True, "probs": probs, "value": int(np.argmax(probs))}
    feats = rng.choice(X.shape[1], size=min(max_features, X.shape[1]), replace=False)
    split = best_split(X, y, min_samples, feats)
    if split is None:
        probs = np.bincount(y, minlength=2) / len(y)
        return {"leaf": True, "probs": probs, "value": int(np.argmax(probs))}
    f, t, left, right = split
    return {"leaf": False, "feature": f, "threshold": t,
            "left": build_tree(X[left], y[left], depth + 1, max_depth, min_samples, rng, max_features),
            "right": build_tree(X[right], y[right], depth + 1, max_depth, min_samples, rng, max_features)}

def tree_probs(tree, X):
    out = []
    for row in X:
        node = tree
        while not node["leaf"]:
            node = node["left"] if row[node["feature"]] <= node["threshold"] else node["right"]
        out.append(node["probs"])
    return np.array(out)

def rf_fit(X, y, n_trees=40, max_depth=6, max_features=5, seed=42):
    rng = np.random.default_rng(seed)
    trees = []
    for _ in range(n_trees):
        idx = rng.integers(0, len(X), len(X))
        trees.append(build_tree(X[idx], y[idx], 0, max_depth, 10, rng, max_features))
    return trees

def rf_probs(trees, X):
    return np.mean([tree_probs(t, X)[:, 1] for t in trees], axis=0)

def roc_curve(y, score):
    order = np.argsort(-score); y = y[order]
    tps = np.cumsum(y == 1); fps = np.cumsum(y == 0)
    return np.r_[0, fps / max(1, fps[-1])], np.r_[0, tps / max(1, tps[-1])]

def pr_curve(y, score):
    order = np.argsort(-score); y = y[order]
    tps = np.cumsum(y == 1); fps = np.cumsum(y == 0)
    return tps / max(1, tps[-1]), tps / np.maximum(1, tps + fps)

def auc(x, y):
    return float(np.trapezoid(y, x))

# ----------------------------------------------------------------------
# Modelos
# ----------------------------------------------------------------------
Xtr, Xte, ytr, yte = stratified_split(X, y, 0.30, seed=42)
Xtr_s, Xte_s = standardize(Xtr, Xte)

# Validación interna para optimizar
Xfit, Xval, yfit, yval = stratified_split(Xtr_s, ytr, 0.22, seed=7)

models = {}
scores = {}
scores["Regresión logística"] = logistic(Xfit, yfit, Xte_s)
models["Regresión logística"] = (scores["Regresión logística"] >= 0.5).astype(int)
models["k-NN (k=15)"] = knn(Xfit, yfit, Xte_s, 15)
scores["k-NN (k=15)"] = models["k-NN (k=15)"].astype(float)
tree = build_tree(Xfit, yfit, 0, 5, 10, np.random.default_rng(1), 5)
scores["Árbol de decisión"] = tree_probs(tree, Xte_s)[:, 1]
models["Árbol de decisión"] = (scores["Árbol de decisión"] >= 0.5).astype(int)
rf = rf_fit(Xfit, yfit, 40, 6, 5, seed=42)
scores["Random Forest"] = rf_probs(rf, Xte_s)
models["Random Forest"] = (scores["Random Forest"] >= 0.5).astype(int)

results = {k: metrics(yte, v) for k, v in models.items()}

# Optimización de profundidad y umbral con validación
depths = [3, 4, 5, 6, 7]
depth_scores = []
for d in depths:
    t = rf_fit(Xfit, yfit, 25, d, 5, seed=9)
    p = rf_probs(t, Xval)
    depth_scores.append(metrics(yval, (p >= 0.5).astype(int))["f1"])
best_depth = depths[int(np.argmax(depth_scores))]

val_probs = rf_probs(rf_fit(Xfit, yfit, 40, best_depth, 5, seed=42), Xval)
thresholds = np.arange(0.1, 0.91, 0.05)
th_scores = [metrics(yval, (val_probs >= t).astype(int))["f1"] for t in thresholds]
best_threshold = float(thresholds[int(np.argmax(th_scores))])

rf_opt = rf_fit(Xtr_s, ytr, 50, best_depth, 5, seed=42)
scores["Random Forest optimizado"] = rf_probs(rf_opt, Xte_s)
models["Random Forest optimizado"] = (scores["Random Forest optimizado"] >= best_threshold).astype(int)
results["Random Forest optimizado"] = metrics(yte, models["Random Forest optimizado"])

# Importancia por permutación
def perm_importance(score_fn, Xb, yb, repeats=8, seed=99):
    rng = np.random.default_rng(seed)
    base = metrics(yb, (score_fn(Xb) >= 0.5).astype(int))["f1"]
    out = []
    for j in range(Xb.shape[1]):
        vals = []
        for _ in range(repeats):
            Xp = Xb.copy(); rng.shuffle(Xp[:, j])
            vals.append(metrics(yb, (score_fn(Xp) >= 0.5).astype(int))["f1"] - base)
        out.append(float(np.mean(vals)))
    return np.array(out)

imp = perm_importance(lambda Z: rf_probs(rf_opt, Z), Xte_s, yte)

# ----------------------------------------------------------------------
# Figuras
# ----------------------------------------------------------------------
# Distribución de la variable objetivo por año
fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
by_year = df.groupby("year")["media"].agg(["mean", "count"])
axes[0].plot(by_year.index, by_year["mean"], color=GREEN, lw=1.8)
axes[0].set_title("Tasa de cobertura mediática por año", fontsize=10, fontweight="bold", color=GREEN)
axes[0].set_xlabel("Año"); axes[0].set_ylabel("Proporción con Media = Y"); axes[0].grid(alpha=0.25)
counts = df["media"].value_counts().sort_index()
axes[1].bar(["Sin media (0)", "Con media (1)"], counts.values, color=[BLUE, GOLD], edgecolor="white")
axes[1].set_title("Distribución de la variable objetivo", fontsize=10, fontweight="bold", color=GREEN)
axes[1].set_ylabel("Registros")
for i, v in enumerate(counts.values):
    axes[1].text(i, v + 8, str(v), ha="center", fontsize=9)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig_target.png"), dpi=300, bbox_inches="tight", facecolor="white")
plt.close(fig)

# Métricas por modelo
names = list(results.keys())
fig, ax = plt.subplots(figsize=(9.5, 4.1))
xpos = np.arange(len(names)); w = 0.2
for i, (m, lab, col) in enumerate(zip(["accuracy", "precision", "recall", "f1"],
                                     ["Exactitud", "Precisión", "Recall", "F1"],
                                     [GREEN, BLUE, GOLD, RED])):
    vals = [results[n][m] for n in names]
    ax.bar(xpos + (i - 1.5) * w, vals, w, label=lab, color=col, edgecolor="white")
ax.set_xticks(xpos); ax.set_xticklabels(names, rotation=15, ha="right", fontsize=7.5)
ax.set_ylim(0, 1.02); ax.set_ylabel("Valor")
ax.set_title("Comparación de métricas de clasificación (UFO)", fontsize=11, fontweight="bold", color=GREEN)
ax.legend(ncol=4, fontsize=7.5, loc="lower center", bbox_to_anchor=(0.5, -0.32))
ax.grid(axis="y", alpha=0.25)
fig.savefig(os.path.join(OUT, "fig_metricas.png"), dpi=300, bbox_inches="tight", facecolor="white")
plt.close(fig)

best_name = max(results, key=lambda n: results[n]["f1"])
best_pred = models[best_name]
m = metrics(yte, best_pred)
cm = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])
fig, ax = plt.subplots(figsize=(4.4, 3.8))
ax.imshow(cm, cmap="Greens")
for i in range(2):
    for j in range(2):
        ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                color="white" if cm[i, j] > cm.max() / 2 else GREEN, fontsize=13, fontweight="bold")
ax.set_xticks([0, 1]); ax.set_xticklabels(["Sin media", "Con media"], fontsize=8)
ax.set_yticks([0, 1]); ax.set_yticklabels(["Sin media", "Con media"], fontsize=8)
ax.set_xlabel("Predicción"); ax.set_ylabel("Valor real")
ax.set_title(f"Matriz de confusión: {best_name}", fontsize=10, fontweight="bold", color=GREEN)
fig.colorbar(ax.imshow(cm, cmap="Greens"), ax=ax, fraction=0.046, pad=0.04)
fig.savefig(os.path.join(OUT, "fig_matriz.png"), dpi=300, bbox_inches="tight", facecolor="white")
plt.close(fig)

# ROC y PR
score = scores[best_name.replace(" (umbral óptimo)", "")] if " (umbral óptimo)" in best_name else scores[best_name]
fpr, tpr = roc_curve(yte, score); rec, prec = pr_curve(yte, score)
fig, axes = plt.subplots(1, 2, figsize=(9, 3.7))
axes[0].plot(fpr, tpr, color=GREEN, lw=2); axes[0].plot([0, 1], [0, 1], "--", color="#999", lw=1)
axes[0].set_title(f"ROC (AUC = {auc(fpr, tpr):.3f})", fontsize=10, fontweight="bold", color=GREEN)
axes[0].set_xlabel("Falsos positivos"); axes[0].set_ylabel("Verdaderos positivos")
axes[1].plot(rec, prec, color=BLUE, lw=2)
axes[1].set_title("Curva precisión-recall", fontsize=10, fontweight="bold", color=GREEN)
axes[1].set_xlabel("Recall"); axes[1].set_ylabel("Precisión")
for a in axes: a.grid(alpha=0.25)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig_roc_pr.png"), dpi=300, bbox_inches="tight", facecolor="white")
plt.close(fig)

# Importancia
order = np.argsort(imp)
top = order[-12:]
fig, ax = plt.subplots(figsize=(7.8, 4.4))
ax.barh(np.array(feature_cols)[top], imp[top], color=GREEN2, edgecolor="white")
ax.set_xlabel("Cambio en F1 al permutar la variable")
ax.set_title("Importancia de variables (Random Forest)", fontsize=10, fontweight="bold", color=GREEN)
ax.grid(axis="x", alpha=0.25)
fig.savefig(os.path.join(OUT, "fig_importancia.png"), dpi=300, bbox_inches="tight", facecolor="white")
plt.close(fig)

# Optimización
fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6))
axes[0].plot(depths, depth_scores, marker="o", color=GREEN, lw=2)
axes[0].set_title("Profundidad vs. F1 (validación)", fontsize=10, fontweight="bold", color=GREEN)
axes[0].set_xlabel("max_depth"); axes[0].set_ylabel("F1"); axes[0].grid(alpha=0.25)
axes[1].plot(thresholds, th_scores, marker="o", color=GOLD, lw=2)
axes[1].axvline(best_threshold, color=RED, ls="--", lw=1.2)
axes[1].set_title("Umbral vs. F1 (validación)", fontsize=10, fontweight="bold", color=GREEN)
axes[1].set_xlabel("umbral"); axes[1].set_ylabel("F1"); axes[1].grid(alpha=0.25)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig_optimizacion.png"), dpi=300, bbox_inches="tight", facecolor="white")
plt.close(fig)

summary = {
    "datos": {"raw": int(len(raw)), "clean": int(len(df)), "train": int(len(ytr)),
              "test": int(len(yte)), "media_rate": float(y.mean()),
              "media_train": float(ytr.mean()), "media_test": float(yte.mean()),
              "features": feature_cols},
    "resultados": results,
    "optimizacion": {"depths": depths, "depth_scores": depth_scores, "best_depth": best_depth,
                     "thresholds": thresholds.tolist(), "threshold_scores": th_scores,
                     "best_threshold": best_threshold},
    "importancia": dict(zip(feature_cols, imp.tolist())),
    "mejor": best_name,
}
with open(os.path.join(OUT, "resultados.json"), "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)
print(json.dumps(summary, ensure_ascii=False, indent=2))
