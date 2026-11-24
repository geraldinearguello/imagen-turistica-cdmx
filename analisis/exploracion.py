import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud
from analisis import utilidades as u

NEG = {"not", "no", "never", "nunca", "nada", "sin", "jamas", "tampoco", "without", "nor", "ni"}
DOMINIO = {"cdmx", "mexico", "ciudad", "city", "df"}
NOMBRES = {1: "Palabras (unigramas)", 2: "Bigramas", 3: "Trigramas"}


def valido(ng, n):
    w = set(ng.split())
    return not (w & (NEG | DOMINIO)) if n == 1 else not (w & DOMINIO)


def ngramas(df, fuente, fuentes, n, top=15):
    filas = []
    for f in fuentes:
        t = df.loc[df[fuente] == f, "texto_limpio"].fillna("")
        cv = CountVectorizer(ngram_range=(n, n), binary=True, token_pattern=r"\b[a-zñ]{2,}\b")
        m = cv.fit_transform(t)
        c = pd.Series(np.asarray(m.sum(axis=0)).ravel(), index=cv.get_feature_names_out())
        c = c[[valido(x, n) for x in c.index]].nlargest(top)
        filas.append(pd.DataFrame({fuente: f, "n": n, "ngrama": c.index, "textos": c.values,
                                   "porcentaje": (c.values / len(t) * 100).round(1)}))
    return pd.concat(filas)


def graf_ngramas(d, fuente, fuentes, n, paleta, ruta):
    col, tra = u.colores(fuentes, paleta), u.tramas(fuentes)
    fig, axs = plt.subplots(1, len(fuentes), figsize=(5.5 * len(fuentes), 6), squeeze=False)
    for ax, f in zip(axs[0], fuentes):
        x = d[d[fuente] == f]
        ax.barh(x["ngrama"][::-1], x["porcentaje"][::-1], color=col[f], hatch=tra[f], edgecolor="white")
        ax.set_title(u.nombre(f))
        ax.set_xlabel("% de textos que lo mencionan")
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"{NOMBRES[n]} más frecuentes por fuente")
    fig.tight_layout()
    u.guardar(fig, ruta)


def graf_nubes(df, texto, fuente, fuentes, paleta, ruta):
    fig, axs = plt.subplots(2, len(fuentes), figsize=(6 * len(fuentes), 7), squeeze=False)
    for j, f in enumerate(fuentes):
        d = df[df[fuente] == f]
        for i, (c, et) in enumerate([(texto, "antes"), ("texto_limpio", "después")]):
            t = " ".join(d[c].fillna("").astype(str))
            wc = WordCloud(width=800, height=450, background_color="white", colormap=paleta,
               max_words=80, collocations=False,
               stopwords=DOMINIO if i == 1 else set()).generate(t or "vacio")
            axs[i, j].imshow(wc)
            axs[i, j].set_title(f"{u.nombre(f)} · {et} de la limpieza")
            axs[i, j].axis("off")
    u.guardar(fig, ruta)


def explorar(df, texto, fuente, dmo, paleta, salida):
    out = f"{salida}/exploracion"
    os.makedirs(out, exist_ok=True)
    fuentes = u.orden(df, fuente, dmo)
    todos = []
    for n in [1, 2, 3]:
        d = ngramas(df, fuente, fuentes, n)
        todos.append(d)
        graf_ngramas(d, fuente, fuentes, n, paleta, f"{out}/{n}_{NOMBRES[n].split()[0].lower()}.png")
    pd.concat(todos).to_csv(f"{out}/ngramas.csv", index=False)
    graf_nubes(df, texto, fuente, fuentes, paleta, f"{out}/4_nubes.png")