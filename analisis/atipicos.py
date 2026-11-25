import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from analisis import utilidades as u

PCT = 0.05
NEG = ["not", "no", "never", "nunca", "nada", "sin", "jamas", "tampoco", "without", "nor"]


def detectar(df, emb, fuente, fuentes):
    df["if"], df["lof"] = False, False
    df["x"], df["y"], df["rareza"] = 0.0, 0.0, 0.0
    for f in fuentes:
        i = df.index[df[fuente] == f]
        x = emb[i]
        m_if = IsolationForest(contamination=PCT, random_state=1).fit(x)
        lof = LocalOutlierFactor(n_neighbors=20, metric="cosine", contamination=PCT)
        df.loc[i, "lof"] = lof.fit_predict(x) == -1
        df.loc[i, "if"] = m_if.predict(x) == -1
        r_if = pd.Series(-m_if.score_samples(x)).rank(pct=True).values
        r_lof = pd.Series(-lof.negative_outlier_factor_).rank(pct=True).values
        df.loc[i, "rareza"] = ((r_if + r_lof) / 2).round(3)
        x50 = PCA(n_components=min(50, len(i) - 1), random_state=1).fit_transform(x)
        df.loc[i, ["x", "y"]] = TSNE(n_components=2, init="pca", random_state=1).fit_transform(x50)
    df["atipico"] = df["if"] & df["lof"]
    return df


def graf_metodos(res, fuentes, ruta):
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(fuentes))
    grupos = [("if", "Isolation Forest", "#9ecae1", ""), ("lof", "LOF", "#4a90c2", "//"),
              ("atipico", "Ambos métodos (atípicos)", "#1d2b45", "xx")]
    for k, (c, et, cc, h) in enumerate(grupos):
        b = ax.bar(x + (k - 1) * 0.28, res[c] / res["total"] * 100, 0.26,
                   color=cc, hatch=h, edgecolor="white", label=et)
        ax.bar_label(b, labels=res[c].astype(int), fontsize=8, padding=3)
    ax.set_xticks(x, [u.nombre(f) for f in fuentes])
    ax.set_ylabel("% de textos")
    ax.set_title("Textos atípicos por fuente (número de textos sobre cada barra)", pad=35)
    ax.legend(frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    ax.spines[["top", "right"]].set_visible(False)
    u.guardar(fig, ruta)


def graf_ngramas(at, salida):
    filas = []
    tit = {2: "Frases de dos palabras (bigramas)", 3: "Frases de tres palabras (trigramas)"}
    for n, color in [(2, "#cc79a7"), (3, "#d55e00")]:
        cv = CountVectorizer(ngram_range=(n, n), token_pattern=r"\b[a-zñ]{3,}\b", stop_words=NEG)
        m = cv.fit_transform(at["texto_limpio"])
        c = pd.Series(np.asarray(m.sum(axis=0)).ravel(), index=cv.get_feature_names_out())
        c = c[c >= 2].nlargest(15)
        filas.append(pd.DataFrame({"n": n, "frase": c.index, "menciones": c.values}))
        fig, ax = plt.subplots(figsize=(8, 6.5))
        ax.barh(c.index[::-1], c.values[::-1], color=color)
        ax.set_xlabel("Número de menciones")
        ax.set_title(f"{tit[n]} en textos atípicos")
        ax.spines[["top", "right"]].set_visible(False)
        u.guardar(fig, f"{salida}/2_ngramas_{n}.png")
    pd.concat(filas).to_csv(f"{salida}/2_ngramas.csv", index=False)


def graf_mapa(df, fuente, fuentes, ruta):
    n = len(fuentes)
    fig, axs = plt.subplots(1, n, figsize=(5 * n, 5), squeeze=False)
    for ax, f in zip(axs[0], fuentes):
        d = df[df[fuente] == f]
        k = d["atipico"].values
        ax.scatter(d["x"][~k], d["y"][~k], s=6, c="#bfc3c9", alpha=0.6, label=f"Normal ({(~k).sum()})")
        ax.scatter(d["x"][k], d["y"][k], s=40, c="#d55e00", marker="X", edgecolors="black",
                   linewidths=0.4, label=f"Atípico ({k.sum()})")
        ax.set_title(u.nombre(f))
        ax.set_xticks([]); ax.set_yticks([])
        ax.legend(fontsize=9, loc="upper right")
    fig.suptitle("Atípicos (t-SNE: textos cercanos hablan de lo mismo)")
    u.guardar(fig, ruta)


def analizar(df, emb, fuente, dmo, calificacion, salida):
    out = f"{salida}/atipicos"
    os.makedirs(out, exist_ok=True)
    df["texto_limpio"] = df["texto_limpio"].fillna("")
    fuentes = u.orden(df, fuente, dmo)
    df = detectar(df, emb, fuente, fuentes)
    reseñas = df.groupby(fuente)[calificacion].transform(lambda s: s.notna().any()) if calificacion else False
    df["usar"] = ~df["atipico"] | reseñas
    res = df.groupby(fuente)[["if", "lof", "atipico"]].sum().reindex(fuentes)
    res["total"] = df[fuente].value_counts().reindex(fuentes)
    res["siguen"] = df.groupby(fuente)["usar"].sum().reindex(fuentes)
    res.to_csv(f"{out}/1_resumen.csv")
    graf_metodos(res, fuentes, f"{out}/1_metodos.png")
    graf_ngramas(df[df["atipico"]], out)
    graf_mapa(df, fuente, fuentes, f"{out}/3_mapa.png")
    df[df["atipico"]].sort_values("rareza", ascending=False)[[fuente, "rareza", "texto_base"]].to_csv(
        f"{out}/4_muestra.csv", index=False)
    df[[fuente, "x", "y", "atipico", "rareza"]].to_csv(f"{out}/3_mapa.csv", index=False)
    return df.drop(columns=["x", "y"])