# -*- coding: utf-8 -*-
"""
U4 Saber Hacer - Análisis no supervisado con dataset UFO.
K-Means + PCA, métricas, optimización y comparación antes/después.
"""
import json
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, "resultados_ufo_u4")
os.makedirs(OUT, exist_ok=True)
plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = 9
GREEN = "#1F3B2E"; GREEN2 = "#2E5D45"; GOLD = "#C9A227"; RED = "#A23B4A"; BLUE = "#2F6B8A"
PALETTE = ["#1F3B2E", "#2F6B8A", "#C9A227", "#A23B4A", "#7A4E8A", "#3D7A4A", "#B5652A", "#6B5B2A"]

df = pd.read_csv(os.path.join(BASE, "ufo_clean.csv"))
num_cols = ["year", "month", "hour", "weekday", "summary_len", "word_count"] + [c for c in df.columns if c.startswith("kw_")]
cat_cols = [c for c in df.columns if c.startswith("shape_group_") or c.startswith("country_group_") or c.startswith("state_group_")]
feature_cols = num_cols + cat_cols

Xraw = df[feature_cols].values.astype(float)
mu = Xraw.mean(0); sd = Xraw.std(0); sd[sd == 0] = 1
X = (Xraw - mu) / sd

# ----------------------------------------------------------------------
# PCA
# ----------------------------------------------------------------------
def pca_fit(X, n_components=None):
    Xc = X - X.mean(0)
    cov = np.cov(Xc, rowvar=False)
    vals, vecs = np.linalg.eigh(cov)
    order = np.argsort(vals)[::-1]
    vals = np.maximum(vals[order], 0)
    vecs = vecs[:, order]
    if n_components is None:
        n_components = X.shape[1]
    return vecs[:, :n_components], vals

def pca_transform(X, comps):
    return (X - X.mean(0)) @ comps

def reconstruction_error(X, comps):
    Xc = X - X.mean(0)
    Z = Xc @ comps
    Xr = Z @ comps.T + X.mean(0)
    return float(np.mean((X - Xr) ** 2))

comps_all, eigvals = pca_fit(X)
explained = eigvals / eigvals.sum()
cum_explained = np.cumsum(explained)
n80 = int(np.searchsorted(cum_explained, 0.80) + 1)
n90 = int(np.searchsorted(cum_explained, 0.90) + 1)
comps80 = comps_all[:, :n80]
Z80 = pca_transform(X, comps80)
Z2 = pca_transform(X, comps_all[:, :2])

# ----------------------------------------------------------------------
# Métricas
# ----------------------------------------------------------------------
def dist_matrix(A, B=None):
    if B is None:
        B = A
    return np.sqrt(((A[:, None, :] - B[None, :, :]) ** 2).sum(2))

def silhouette(X, labels):
    uniq = np.unique(labels)
    if len(uniq) < 2:
        return -1.0
    D = dist_matrix(X)
    scores = []
    for i in range(len(X)):
        own = labels == labels[i]
        own[i] = False
        if own.sum() == 0:
            continue
        a = D[i, own].mean()
        b = min(D[i, labels == u].mean() for u in uniq if u != labels[i])
        scores.append((b - a) / max(a, b))
    return float(np.mean(scores)) if scores else -1.0

def davies_bouldin(X, labels):
    uniq = np.unique(labels)
    if len(uniq) < 2:
        return np.inf
    centroids = np.array([X[labels == u].mean(0) for u in uniq])
    scatters = np.array([np.mean(np.linalg.norm(X[labels == u] - centroids[i], axis=1)) for i, u in enumerate(uniq)])
    db = []
    for i in range(len(uniq)):
        vals = []
        for j in range(len(uniq)):
            if i == j:
                continue
            d = np.linalg.norm(centroids[i] - centroids[j])
            vals.append((scatters[i] + scatters[j]) / d if d else np.inf)
        db.append(max(vals))
    return float(np.mean(db))

def calinski_harabasz(X, labels):
    uniq = np.unique(labels)
    if len(uniq) < 2:
        return 0.0
    overall = X.mean(0)
    between = 0.0; within = 0.0
    for u in uniq:
        Xu = X[labels == u]
        between += len(Xu) * np.sum((Xu.mean(0) - overall) ** 2)
        within += np.sum((Xu - Xu.mean(0)) ** 2)
    return float((between / (len(uniq) - 1)) / (within / (len(X) - len(uniq))))

# ----------------------------------------------------------------------
# K-Means
# ----------------------------------------------------------------------
def kmeans(X, k, n_init=12, max_iter=200, seed=42):
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(n_init):
        idx = rng.choice(len(X), size=k, replace=False)
        C = X[idx].copy()
        for _ in range(max_iter):
            D = dist_matrix(X, C)
            labels = D.argmin(1)
            newC = np.array([X[labels == j].mean(0) if np.any(labels == j) else C[j] for j in range(k)])
            if np.allclose(newC, C):
                break
            C = newC
        inertia = float(np.sum((X - C[labels]) ** 2))
        if best is None or inertia < best[0]:
            best = (inertia, labels, C)
    return best

k_values = list(range(2, 11))
k_results = []
for k in k_values:
    inertia, labels, C = kmeans(Z80, k, n_init=8, seed=42)
    k_results.append({
        "k": k, "inertia": inertia, "silhouette": silhouette(Z80, labels),
        "davies_bouldin": davies_bouldin(Z80, labels),
        "calinski_harabasz": calinski_harabasz(Z80, labels),
        "labels": labels, "centroids": C,
    })
best_k_row = max(k_results, key=lambda r: r["silhouette"])
silhouette_max_k = best_k_row["k"]
silhouette_max = best_k_row["silhouette"]
# Selección final: k=6 por equilibrio entre codo, tamaño de clusters e interpretabilidad.
best_k = 6
best_k_row = next(r for r in k_results if r["k"] == best_k)
best_labels = best_k_row["labels"]
best_centroids = best_k_row["centroids"]

# ----------------------------------------------------------------------
# Comparación antes vs después de la optimización
# ----------------------------------------------------------------------
# Antes: k=3 sobre todas las variables estandarizadas.
_, init_labels, _ = kmeans(X, 3, n_init=8, seed=7)
initial_metrics = {
    "config": "k=3, todas las variables",
    "k": 3, "components": len(feature_cols),
    "silhouette": silhouette(X, init_labels),
    "davies_bouldin": davies_bouldin(X, init_labels),
    "calinski_harabasz": calinski_harabasz(X, init_labels),
}
# Después: k óptimo sobre PCA con 80 % de varianza.
optimized_metrics = {
    "config": f"k={best_k}, PCA {n80} componentes",
    "k": best_k, "components": n80,
    "silhouette": best_k_row["silhouette"],
    "davies_bouldin": best_k_row["davies_bouldin"],
    "calinski_harabasz": best_k_row["calinski_harabasz"],
}

# ----------------------------------------------------------------------
# Perfiles
# ----------------------------------------------------------------------
df_out = df.copy()
df_out["cluster"] = best_labels
profile_rows = []
for c in range(best_k):
    sub = df_out[df_out["cluster"] == c]
    profile_rows.append({
        "cluster": int(c), "n": int(len(sub)),
        "media_rate": float(sub["media"].mean()),
        "year_mean": float(sub["year"].mean()),
        "hour_mean": float(sub["hour"].mean()),
        "summary_len": float(sub["summary_len"].mean()),
        "top_shape": sub["shape_group"].value_counts().index[0],
        "top_country": sub["country_group"].value_counts().index[0],
        "top_state": sub["state_group"].value_counts().index[0],
    })

# ----------------------------------------------------------------------
# Figuras
# ----------------------------------------------------------------------
# PCA varianza
fig, ax = plt.subplots(figsize=(7.6, 3.8))
ax.bar(range(1, len(explained) + 1), explained, color=GREEN2, edgecolor="white")
ax.plot(range(1, len(explained) + 1), cum_explained, color=GOLD, marker="o", markersize=3, lw=1.8)
ax.axhline(0.80, color=RED, ls="--", lw=1)
ax.set_xlabel("Componente principal"); ax.set_ylabel("Proporción")
ax.set_title("Varianza explicada por PCA", fontsize=11, fontweight="bold", color=GREEN)
ax.set_ylim(0, 1.02); ax.grid(alpha=0.25)
fig.savefig(os.path.join(OUT, "fig_pca.png"), dpi=300, bbox_inches="tight", facecolor="white")
plt.close(fig)

# Error de reconstrucción
fig, ax = plt.subplots(figsize=(7.2, 3.7))
errs = [reconstruction_error(X, comps_all[:, :m]) for m in range(1, 11)]
ax.plot(range(1, 11), errs, marker="o", color=BLUE, lw=2)
ax.axvline(n80, color=RED, ls="--", lw=1, label=f"80 % varianza: {n80} componentes")
ax.set_xlabel("Número de componentes"); ax.set_ylabel("Error de reconstrucción")
ax.set_title("Error de reconstrucción de PCA", fontsize=11, fontweight="bold", color=GREEN)
ax.legend(fontsize=8); ax.grid(alpha=0.25)
fig.savefig(os.path.join(OUT, "fig_error_pca.png"), dpi=300, bbox_inches="tight", facecolor="white")
plt.close(fig)

# Selección de k
fig, axes = plt.subplots(2, 2, figsize=(9.6, 6.2))
axes[0, 0].plot(k_values, [r["inertia"] for r in k_results], marker="o", color=GREEN, lw=2)
axes[0, 0].set_title("Codo: inercia", fontsize=9, fontweight="bold", color=GREEN)
axes[0, 0].set_xlabel("k"); axes[0, 0].set_ylabel("Inercia"); axes[0, 0].grid(alpha=0.25)
axes[0, 1].plot(k_values, [r["silhouette"] for r in k_results], marker="o", color=GOLD, lw=2)
axes[0, 1].axvline(best_k, color=RED, ls="--", lw=1)
axes[0, 1].set_title("Silhouette (mayor es mejor)", fontsize=9, fontweight="bold", color=GREEN)
axes[0, 1].set_xlabel("k"); axes[0, 1].set_ylabel("Silhouette"); axes[0, 1].grid(alpha=0.25)
axes[1, 0].plot(k_values, [r["davies_bouldin"] for r in k_results], marker="o", color=RED, lw=2)
axes[1, 0].set_title("Davies-Bouldin (menor es mejor)", fontsize=9, fontweight="bold", color=GREEN)
axes[1, 0].set_xlabel("k"); axes[1, 0].set_ylabel("DB"); axes[1, 0].grid(alpha=0.25)
axes[1, 1].plot(k_values, [r["calinski_harabasz"] for r in k_results], marker="o", color=BLUE, lw=2)
axes[1, 1].set_title("Calinski-Harabasz (mayor es mejor)", fontsize=9, fontweight="bold", color=GREEN)
axes[1, 1].set_xlabel("k"); axes[1, 1].set_ylabel("CH"); axes[1, 1].grid(alpha=0.25)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig_k_seleccion.png"), dpi=300, bbox_inches="tight", facecolor="white")
plt.close(fig)

# Clusters en PCA 2D
fig, ax = plt.subplots(figsize=(7.4, 5.8))
for c in range(best_k):
    mask = best_labels == c
    ax.scatter(Z2[mask, 0], Z2[mask, 1], s=14, alpha=0.65, color=PALETTE[c % len(PALETTE)],
               edgecolor="white", linewidth=0.2, label=f"Cluster {c} (n={mask.sum()})")
ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
ax.set_title(f"Clusters K-Means sobre el plano PCA (k={best_k})", fontsize=11, fontweight="bold", color=GREEN)
ax.legend(fontsize=7.5, loc="best"); ax.grid(alpha=0.2)
fig.savefig(os.path.join(OUT, "fig_clusters.png"), dpi=300, bbox_inches="tight", facecolor="white")
plt.close(fig)

# Perfiles
fig, axes = plt.subplots(1, 4, figsize=(12, 3.8))
for ax, (key, title) in zip(axes, [("media_rate", "Tasa de media"), ("year_mean", "Año promedio"),
                                   ("hour_mean", "Hora promedio"), ("summary_len", "Longitud del resumen")]):
    vals = [r[key] for r in profile_rows]
    xs = np.arange(len(vals))
    ax.bar([f"C{r['cluster']}" for r in profile_rows], vals, color=GREEN2, edgecolor="white")
    if key == "year_mean":
        ax.set_ylim(min(vals) - 5, max(vals) + 5)
    elif key == "hour_mean":
        ax.set_ylim(min(vals) - 1, max(vals) + 1)
    elif key == "summary_len":
        ax.set_ylim(0, max(vals) + 20)
    elif key == "media_rate":
        ax.set_ylim(0, max(vals) + 0.08)
    for i, v in enumerate(vals):
        fmt = f"{v:.2f}" if key == "media_rate" else f"{v:.0f}"
        ax.text(i, v, fmt, ha="center", va="bottom", fontsize=6.5, color="#333333")
    ax.set_title(title, fontsize=9, fontweight="bold", color=GREEN)
    ax.grid(axis="y", alpha=0.25); ax.tick_params(labelsize=8)
fig.suptitle("Perfil de los clusters K-Means", fontsize=11, fontweight="bold", color=GREEN)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig_perfiles.png"), dpi=300, bbox_inches="tight", facecolor="white")
plt.close(fig)

# Distribución de formas por cluster
shape_counts = pd.crosstab(df_out["cluster"], df_out["shape_group"], normalize="index")
fig, ax = plt.subplots(figsize=(8.4, 4.0))
bottom = np.zeros(len(shape_counts))
colors = [GREEN, BLUE, GOLD, RED, "#7A4E8A"]
for i, col in enumerate(shape_counts.columns):
    ax.bar(shape_counts.index.astype(str), shape_counts[col].values, bottom=bottom,
           label=col, color=colors[i % len(colors)], edgecolor="white")
    bottom += shape_counts[col].values
ax.set_xlabel("Cluster"); ax.set_ylabel("Proporción")
ax.set_title("Distribución de formas por cluster", fontsize=10, fontweight="bold", color=GREEN)
ax.legend(fontsize=8, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.18))
fig.savefig(os.path.join(OUT, "fig_formas.png"), dpi=300, bbox_inches="tight", facecolor="white")
plt.close(fig)

# Comparación antes vs después
fig, ax = plt.subplots(figsize=(7.6, 3.6))
labels = ["Silhouette", "Davies-Bouldin\n(invertido)", "Calinski-Harabasz\n(normalizado)"]
# Normalizar para comparar visualmente
init_vals = [initial_metrics["silhouette"], 1/(1+initial_metrics["davies_bouldin"]), initial_metrics["calinski_harabasz"]]
opt_vals = [optimized_metrics["silhouette"], 1/(1+optimized_metrics["davies_bouldin"]), optimized_metrics["calinski_harabasz"]]
init_norm = [init_vals[0], init_vals[1], init_vals[2]/max(init_vals[2], opt_vals[2])]
opt_norm = [opt_vals[0], opt_vals[1], opt_vals[2]/max(init_vals[2], opt_vals[2])]
x = np.arange(3); w = 0.35
ax.bar(x - w/2, init_norm, w, label="Antes (k=3, todas)", color="#BBBBBB", edgecolor="white")
ax.bar(x + w/2, opt_norm, w, label=f"Después (k={best_k}, PCA {n80})", color=GREEN2, edgecolor="white")
ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
ax.set_title("Comparación antes vs. después de la optimización", fontsize=10, fontweight="bold", color=GREEN)
ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.25)
fig.savefig(os.path.join(OUT, "fig_comparacion.png"), dpi=300, bbox_inches="tight", facecolor="white")
plt.close(fig)

# ----------------------------------------------------------------------
# JSON
# ----------------------------------------------------------------------
summary = {
    "datos": {"n": int(len(df)), "features": feature_cols, "media_rate": float(df["media"].mean())},
    "pca": {"explained": explained.tolist(), "cum_explained": cum_explained.tolist(),
            "n80": n80, "n90": n90},
    "kmeans": [{"k": r["k"], "inertia": r["inertia"], "silhouette": r["silhouette"],
                "davies_bouldin": r["davies_bouldin"], "calinski_harabasz": r["calinski_harabasz"]}
               for r in k_results],
    "best_k": best_k,
    "best_silhouette": float(best_k_row["silhouette"]),
    "silhouette_max_k": silhouette_max_k,
    "silhouette_max": float(silhouette_max),
    "profiles": profile_rows,
    "comparacion": {"antes": initial_metrics, "despues": optimized_metrics},
}

# Dataset transformado para el repositorio
transformado = df.copy()
transformado["cluster"] = best_labels
transformado["PC1"] = Z2[:, 0]
transformado["PC2"] = Z2[:, 1]
transformado.to_csv(os.path.join(BASE, "ufo_u4_transformado.csv"), index=False, encoding="utf-8")

with open(os.path.join(OUT, "resultados.json"), "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)
print(json.dumps(summary, ensure_ascii=False, indent=2))
