# U4 Saber Hacer · Análisis no supervisado con dataset UFO

Análisis de agrupación y reducción de dimensionalidad sobre el mismo dataset de avistamientos de la U3. Se aplicaron **K-Means** y **PCA**, con evaluación mediante silhouette, Davies-Bouldin y Calinski-Harabasz.

## Contenido

- `ufo_kmeans_pca.py`: PCA, K-Means, métricas, optimización y figuras.
- `ufo_data.csv`: dataset original del repositorio Dataset_UFO.
- `ufo_clean.csv`: dataset limpio compartido con la U3.
- `ufo_u4_transformado.csv`: dataset transformado con cluster, PC1 y PC2.
- `resultados/`: figuras PNG y `resultados.json`.
- `docs/`: documento final en PDF y DOCX (en la carpeta raíz `docs/`).

## Requisitos

- Python 3.10 o superior.
- NumPy, pandas y Matplotlib.

## Ejecución

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
pip install -r ../requirements.txt
python ufo_kmeans_pca.py
```

## Caso

- 1,164 reportes limpios y 16 variables estructuradas.
- La variable `media` se excluye del clustering; se usa solo para describir los clusters.
- Estandarización z-score, PCA y K-Means con 8 inicializaciones por k.
- PCA: 12 componentes explican el 80 % de la varianza; 13 explican el 90 %.
- PC1 y PC2 explican 21.81 % de la varianza y se usan solo para visualizar.

## Resultados principales

### Selección de k

| k | Silhouette | Davies-Bouldin | Calinski-Harabasz |
|---:|---:|---:|---:|
| 2 | 0.1338 | 2.5138 | 154.11 |
| 3 | 0.1430 | 1.9083 | 132.89 |
| 4 | 0.1574 | 1.9568 | 124.03 |
| 5 | 0.1466 | 1.7826 | 115.59 |
| 6 | 0.1665 | 1.7038 | 127.36 |
| 7 | 0.1477 | 1.8262 | 120.42 |
| 8 | 0.1265 | 1.8685 | 104.82 |
| 9 | 0.1466 | 1.6773 | 126.63 |
| 10 | 0.1742 | 1.4025 | 149.46 |

Se seleccionó **k = 6** por equilibrio entre métricas, tamaño de los clusters e interpretabilidad. El máximo silhouette fue k = 10, pero generó grupos pequeños y menos estables.

### Comparación antes vs. después

| Configuración | Silhouette | Davies-Bouldin | Calinski-Harabasz |
|---|---:|---:|---:|
| Antes: k=3, 16 variables | 0.1209 | 2.1254 | 110.91 |
| Después: k=6, PCA 12 componentes | 0.1665 | 1.7038 | 127.36 |

### Perfil de clusters

| Cluster | n | Media | Año | Hora | Longitud | Forma principal |
|---|---:|---:|---:|---:|---:|---|
| C0 | 456 | 34.6 % | 2009 | 14.8 | 51 | Geometric |
| C1 | 67 | 34.3 % | 2011 | 16.7 | 99 | Geometric |
| C2 | 75 | 46.7 % | 2015 | 15.4 | 91 | Geometric |
| C3 | 357 | 37.8 % | 2007 | 14.9 | 116 | Geometric |
| C4 | 191 | 30.4 % | 2008 | 15.4 | 99 | Geometric |
| C5 | 18 | 33.3 % | 2003 | 14.8 | 87 | Geometric |

## Interpretación

La estructura de los datos es débil: un silhouette de 0.1665 indica separación limitada. Los clusters se diferencian más por año y longitud del texto que por la forma del avistamiento, que es mayoritariamente geométrica en todos los grupos. PCA mejora las tres métricas respecto a la configuración inicial, pero la visualización en 2D solo explica 21.81 % de la varianza; los clusters se calcularon en 12 dimensiones.

## Nota

Las implementaciones son didácticas y no sustituyen bibliotecas optimizadas. El dataset es público y se conserva en su versión original, limpia y transformada para reproducibilidad.
