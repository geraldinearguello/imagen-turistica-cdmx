import os
import re
import unicodedata
import pandas as pd
import spacy

MODELOS = {"es": "es_core_news_sm", "en": "en_core_web_sm",
           "fr": "fr_core_news_sm", "pt": "pt_core_news_sm"}
NEG = {"no", "ni", "nunca", "jamas", "jamás", "nada", "tampoco", "sin",
       "not", "never", "nor", "nothing", "without"}

emo = re.compile("[\U0001F000-\U0001FAFF\u2600-\u27BF\uFE0F\u200B-\u200F\u2060-\u206F]")
web = re.compile(r"https?://\S+|www\.\S+"
                 r"|https?\s+(www\s+)?(\S+\s+){0,3}(com|mx|org|net|edu|gov|it|be|gl|dev|ie|io|co)\b")


def sin_acentos(t):
    t = unicodedata.normalize("NFD", t)
    return "".join(c for c in t if unicodedata.category(c) != "Mn")


def hashtags(t):
    return " ".join(sin_acentos(x.lower()) for x in re.findall(r"#(\w+)", t))


def base(t):
    t = web.sub(" ", t)
    t = re.sub(r"[@#]\w+", " ", t)
    t = emo.sub(" ", t)
    t = re.sub(r"\b(\w+)n t\b", r"\1 not", t)
    return re.sub(r"\s+", " ", t).strip()


def cargar_modelos(idioma):
    idiomas = sorted({idioma, "en"})
    return {i: spacy.load(MODELOS[i], disable=["parser", "ner"]) for i in idiomas}


def detectar(t, sw, idioma):
    w = re.findall(r"\w+", sin_acentos(t.lower()))
    return max(sw, key=lambda i: (sum(x in sw[i] for x in w), i == idioma))


def lematizar(docs, nlp, sw_todas):
    out = []
    for d in nlp.pipe(docs, batch_size=500):
        w = []
        for tk in d:
            if tk.lower_ in NEG:
                w.append(tk.lower_)
            elif tk.is_alpha and not tk.is_stop:
                lem = sin_acentos(tk.lemma_.lower())
                w += [p for p in lem.split() if p not in sw_todas and len(p) > 2]
        t = re.sub(r"\b(\w+)(?: \1\b){2,}", " ", " ".join(w))
        out.append(re.sub(r"\s+", " ", sin_acentos(t)).strip())
    return out


def limpiar(df, texto, fuente, idioma, salida):
    nlp = cargar_modelos(idioma)
    sw = {i: {sin_acentos(x) for x in m.Defaults.stop_words} for i, m in nlp.items()}
    sw_todas = set().union(*sw.values()) | {"el", "ve", "ll", "re", "vos"}

    df = df.dropna(subset=[texto]).copy()
    df[texto] = df[texto].astype(str)
    df["hashtags"] = df[texto].apply(hashtags)
    df["texto_base"] = df[texto].apply(base)
    df["idioma"] = df["texto_base"].apply(lambda t: detectar(t, sw, idioma))

    df["texto_limpio"] = ""
    for i in nlp:
        m = df["idioma"] == i
        df.loc[m, "texto_limpio"] = lematizar(df.loc[m, "texto_base"].tolist(), nlp[i], sw_todas)

    n0 = df[fuente].value_counts()
    df = df.drop_duplicates(subset=[fuente, "texto_base"])
    n1 = df[fuente].value_counts()
    df = df[df["texto_limpio"].str.split().str.len() >= 3]
    n2 = df[fuente].value_counts()

    os.makedirs(f"{salida}/limpieza", exist_ok=True)
    res = pd.DataFrame({"inicial": n0, "duplicados": n0 - n1,
                        "cortos": n1 - n2, "final": n2}).fillna(0).astype(int)
    res.to_csv(f"{salida}/limpieza/resumen_limpieza.csv")
    df.to_csv(f"{salida}/base_limpia.csv", index=False)
    return df.reset_index(drop=True)