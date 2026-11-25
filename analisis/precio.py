import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from analisis import utilidades as u

MODELO = "paraphrase-multilingual-MiniLM-L12-v2"
UMBRAL = 0.40
FRASES = [
    "es muy caro", "los precios son muy altos", "cobran demasiado", "nos cobraron de más",
    "precios inflados para turistas", "es barato", "precios accesibles", "buena relación calidad precio",
    "la entrada es gratuita", "entrada libre", "el costo del boleto", "cuánto cuesta la entrada",
    "hay que pagar para entrar", "vale lo que cuesta", "no vale lo que cobran", "gastamos mucho dinero",
    "too expensive", "very cheap", "overpriced", "the ticket price", "free admission",
    "worth the money", "good value for money", "they charged us extra", "it costs a lot of money",
]


def concepto(m):
    e = m.encode(FRASES, normalize_embeddings=True)
    c = e.mean(axis=0)
    return c / np.linalg.norm(c)


def ventanas(t, tam=30, paso=15):
    p = str(t).split()
    if len(p) <= tam:
        return [" ".join(p)]
    return [" ".join(p[i:i + tam]) for i in range(0, len(p) - tam + paso, paso)]


def similitud(textos, ruta):
    if os.path.exists(ruta):
        z = np.load(ruta, allow_pickle=True)
        if len(z["s"]) == len(textos):
            return z["s"], z["f"]
    m = SentenceTransformer(MODELO)
    c = concepto(m)
    frag, dueño = [], []
    for i, t in enumerate(textos):
        v = ventanas(t)
        frag += v
        dueño += [i] * len(v)
    v = m.encode(frag, batch_size=64, normalize_embeddings=True, show_progress_bar=True) @ c
    dueño, frag = np.array(dueño), np.array(frag, dtype=object)
    s = np.full(len(textos), -1.0)
    np.maximum.at(s, dueño, v)
    mejor = pd.Series(v).groupby(dueño).idxmax().reindex(range(len(textos)))
    f = np.array([frag[int(i)] for i in mejor], dtype=object)
    np.savez(ruta, s=s, f=f)
    return s, f


def resumen(df, fuente, dmo, fuentes):
    grupos = [(f, df[df[fuente] == f]) for f in fuentes]
    if len(fuentes) > 2:
        grupos.append(("visitantes", df[df[fuente] != dmo]))
    filas = []
    for f, d in grupos:
        p = d[d["precio"]]
        filas.append({"fuente": f, "textos": len(d), "hablan_de_precio": len(p),
                      "pct_precio": round(d["precio"].mean() * 100, 1),
                      "pct_negativo_precio": round((p["sentimiento"] == "negativo").mean() * 100, 1) if len(p) else np.nan,
                      "pct_negativo_general": round((d["sentimiento"] == "negativo").mean() * 100, 1)})
    return pd.DataFrame(filas)


def contexto(df, fuente, dmo, fuentes):
    d = df[df["precio"]].copy()
    pos = d["sentimiento"] == "positivo"
    d["tema"] = np.where(pos, "Positivo · " + d["nombre_positivos"], "Negativo · " + d["nombre_negativos"])
    d = d[np.where(pos, d["tema_positivos"], d["tema_negativos"]) >= 0]
    n = d[fuente].value_counts()
    cols = [f for f in fuentes if n.get(f, 0) >= 30]
    p = pd.crosstab(d["tema"], d[fuente], normalize="columns").reindex(columns=cols).fillna(0) * 100
    if len(fuentes) > 2:
        v = d[d[fuente] != dmo]
        p["visitantes"] = v["tema"].value_counts(normalize=True).reindex(p.index).fillna(0) * 100
        n["visitantes"] = len(v)
    ref = p["visitantes"] if "visitantes" in p else pd.Series(0, index=p.index)
    p = p[(ref >= 2) | (p[cols] >= 5).any(axis=1)]
    return p.sort_values(p.columns[-1], ascending=False).round(1), n


def graf_frecuencia_tono(r, fuentes, paleta, ruta):
    col = {**u.colores(fuentes, paleta), "visitantes": "#8b8f98"}
    tra = {**u.tramas(fuentes), "visitantes": "oo"}
    c_neg = plt.get_cmap(paleta)(0.15)
    fig, axs = plt.subplots(1, 2, figsize=(14, 0.8 * len(r) + 1.8))
    y = np.arange(len(r))[::-1]
    for i, x in enumerate(r.itertuples()):
        axs[0].barh(y[i], x.pct_precio, color=col[x.fuente], hatch=tra[x.fuente], edgecolor="white")
        axs[0].text(x.pct_precio, y[i], f" {x.pct_precio:.1f}% ({x.hablan_de_precio:,})", va="center", fontsize=9)
        if not np.isnan(x.pct_negativo_precio):
            axs[1].barh(y[i] + 0.2, x.pct_negativo_precio, 0.4, color=c_neg, hatch="//", edgecolor="white",
                        label="Al hablar de precio" if i == 0 else None)
            axs[1].text(x.pct_negativo_precio, y[i] + 0.2, f" {x.pct_negativo_precio:.0f}%", va="center", fontsize=9)
        axs[1].barh(y[i] - 0.2, x.pct_negativo_general, 0.4, color="#bfc3c9",
                    label="En todos sus textos" if i == 0 else None)
        axs[1].text(x.pct_negativo_general, y[i] - 0.2, f" {x.pct_negativo_general:.0f}%", va="center", fontsize=9)
    for ax in axs:
        ax.set_yticks(y, [u.nombre(f) for f in r["fuente"]])
        ax.spines[["top", "right"]].set_visible(False)
    axs[0].set_xlim(0, r["pct_precio"].max() * 1.35)
    axs[0].set_xlabel("% de textos que hablan de precio")
    axs[0].set_title("Frecuencia: ¿cuánto se habla de precio?")
    axs[1].set_xlim(0, 100)
    axs[1].set_xlabel("% de textos negativos")
    axs[1].set_title("Tono: ¿se habla peor del precio que en general?")
    axs[1].legend(frameon=False, loc="lower right")
    u.guardar(fig, ruta)


def graf_contexto(p, n, paleta, ruta):
    v = p.values
    tope = min(v.max(), 60)
    mx = v.argmax(axis=0)
    fig, ax = plt.subplots(figsize=(4 + 2 * p.shape[1], 0.6 * len(p) + 1.5))
    ax.imshow(v, cmap=paleta, aspect="auto", vmin=0, vmax=tope)
    for j in range(v.shape[1]):
        for i in range(v.shape[0]):
            ax.text(j, i, f"{v[i, j]:.0f}%", ha="center", va="center",
                    color="black" if v[i, j] > tope * 0.6 else "white",
                    fontweight="bold" if i == mx[j] else "normal")
    ax.set_xticks(range(p.shape[1]), [f"{u.nombre(c)}\n(n={n[c]:,})" for c in p.columns])
    ax.xaxis.tick_top()
    ax.set_yticks(range(len(p)), p.index)
    ax.set_title("Temas de los comentarios sobre precio", pad=40)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    u.guardar(fig, ruta)


def analizar(df, fuente, dmo, paleta, salida):
    out = f"{salida}/precio"
    os.makedirs(out, exist_ok=True)
    fuentes = u.orden(df, fuente, dmo)

    df["sim_precio"], df["fragmento_precio"] = similitud(df["texto_base"].tolist(), f"{out}/cache_fragmentos.npz")
    df["precio"] = df["sim_precio"] >= UMBRAL

    r = resumen(df, fuente, dmo, fuentes)
    r.to_csv(f"{out}/1_frecuencia_tono.csv", index=False)
    graf_frecuencia_tono(r, fuentes, paleta, f"{out}/1_frecuencia_tono.png")

    top = df.sort_values("sim_precio", ascending=False).groupby(fuente).head(5)
    top[[fuente, "sim_precio", "sentimiento", "fragmento_precio", "texto_base"]].to_csv(
        f"{out}/2_top5.csv", index=False)

    p, n = contexto(df, fuente, dmo, fuentes)
    if p.shape[1]:
        p.to_csv(f"{out}/3_contexto_temas.csv")
        graf_contexto(p, n, paleta, f"{out}/3_contexto_temas.png")
    return df, r