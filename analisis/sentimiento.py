import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from analisis import utilidades as u

MODELO = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
CP = ["p_neg", "p_neu", "p_pos"]


def probabilidades(textos):
    import torch
    from transformers import pipeline
    disp = "mps" if torch.backends.mps.is_available() else -1
    clf = pipeline("text-classification", model=MODELO, top_k=None, truncation=True,
                   max_length=512, device=disp)
    res = []
    for j in range(0, len(textos), 256):
        for r in clf(textos[j:j + 256], batch_size=32):
            p = {x["label"].lower(): x["score"] for x in r}
            res.append([p.get("negative", 0), p.get("neutral", 0), p.get("positive", 0)])
        print(f"  {min(j + 256, len(textos)):,} / {len(textos):,}")
    return np.array(res)


def con_cache(df, fuente, usa, ruta):
    if os.path.exists(ruta):
        prev = pd.read_csv(ruta).drop_duplicates([fuente, "texto_base"])
        df = df.merge(prev, on=[fuente, "texto_base"], how="left")
    else:
        for c in CP:
            df[c] = np.nan
    falta = usa & df["p_neg"].isna().values
    if falta.any():
        df.loc[falta, CP] = probabilidades(df.loc[falta, "texto_base"].astype(str).tolist())
    df.loc[df["p_neg"].notna(), [fuente, "texto_base"] + CP].to_csv(ruta, index=False)
    return df


def graf_validacion(mc, acc, ruta):
    rec = mc.div(mc.sum(axis=1), axis=0)
    fig, ax = plt.subplots(figsize=(6, 4.8))
    ax.imshow(rec.values, cmap="viridis_r", vmin=0, vmax=1)
    for a in range(2):
        for b in range(2):
            v = rec.values[a, b]
            ax.text(b, a, f"{v:.0%}\n({mc.values[a, b]})", ha="center", va="center",
                    color="white" if v > 0.5 else "black", fontsize=12)
    ax.set_xticks([0, 1], ["Negativo", "Positivo"])
    ax.set_yticks([0, 1], ["Negativo\n(1-2 estrellas)", "Positivo\n(4-5 estrellas)"])
    ax.set_xlabel("Lo que dice el modelo")
    ax.set_ylabel("Lo que dijo la persona")
    ax.set_title(f"Aciertos del modelo ({acc:.0%})")
    u.guardar(fig, ruta)


def graf_fuentes(p, n, paleta, ruta):
    cm = plt.get_cmap(paleta)
    fig, ax = plt.subplots(figsize=(10, 0.8 * len(p) + 1.5))
    y = np.arange(len(p))[::-1]
    ax.barh(y, p["positivo"], color=cm(0.75), label="Positivo")
    ax.barh(y, p["negativo"], left=p["positivo"], color=cm(0.15), hatch="//",
            edgecolor="white", label="Negativo")
    for k, f in enumerate(p.index):
        ax.text(p.loc[f, "positivo"] / 2, y[k], f"{p.loc[f, 'positivo']:.0f}%", ha="center", va="center")
        if p.loc[f, "negativo"] >= 6:
            ax.text(p.loc[f, "positivo"] + p.loc[f, "negativo"] / 2, y[k], f"{p.loc[f, 'negativo']:.0f}%",
                    ha="center", va="center", color="white")
    ax.set_yticks(y, [f"{u.nombre(f)}\n(n={n[f]:,})" for f in p.index])
    ax.set_xlim(0, 100)
    ax.set_xlabel("% de textos")
    ax.set_title("Tono de los textos por fuente", pad=30)
    ax.legend(frameon=False, ncol=2, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    ax.spines[["top", "right"]].set_visible(False)
    u.guardar(fig, ruta)


def calcular(df, fuente, dmo, calificacion, paleta, salida):
    out = f"{salida}/sentimiento"
    os.makedirs(out, exist_ok=True)
    fuentes = u.orden(df, fuente, dmo)

    cal = df[calificacion] if calificacion else pd.Series(np.nan, index=df.index)
    estrellas = cal.isin([1, 2, 4, 5])
    val = df[estrellas].sample(min(2000, estrellas.sum()), random_state=1).index
    usa = (~estrellas | df.index.isin(val)).values

    df = con_cache(df, fuente, usa, f"{out}/probabilidades.csv")
    df["modelo"] = np.where((df["p_neg"] > df["p_neu"]) & (df["p_neg"] > df["p_pos"]), "negativo", "positivo")
    df["sentimiento"] = np.where(estrellas, np.where(cal >= 4, "positivo", "negativo"), df["modelo"])
    df["origen"] = np.where(estrellas, "estrellas", "modelo")

    if len(val):
        t = df.loc[val]
        real = np.where(t["sentimiento"] == "positivo", "positivo", "negativo")
        mc = pd.crosstab(pd.Series(real, name="real"), pd.Series(t["modelo"].values, name="modelo")).reindex(
            index=["negativo", "positivo"], columns=["negativo", "positivo"], fill_value=0)
        acc = (real == t["modelo"].values).mean()
        mc.to_csv(f"{out}/1_validacion.csv")
        graf_validacion(mc, acc, f"{out}/1_validacion.png")
        print(f"  Validación con estrellas: {acc:.0%} de acierto")

    n = df[fuente].value_counts()
    p = pd.crosstab(df[fuente], df["sentimiento"], normalize="index").reindex(fuentes).fillna(0) * 100
    p.round(1).to_csv(f"{out}/2_fuentes.csv")
    graf_fuentes(p, n, paleta, f"{out}/2_fuentes.png")

    o = df[df["origen"] == "modelo"]
    mues = pd.concat([g.sample(min(5, len(g)), random_state=1) for _, g in o.groupby([fuente, "sentimiento"])])
    mues[[fuente, "sentimiento"] + CP + ["texto_base"]].to_csv(f"{out}/3_muestra.csv", index=False)
    return df, p