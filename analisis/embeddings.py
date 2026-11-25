import os
import numpy as np
from sentence_transformers import SentenceTransformer

MODELO = "paraphrase-multilingual-MiniLM-L12-v2"


def pedazos(t, n=100):
    w = t.split()
    return [" ".join(w[i:i + n]) for i in range(0, max(len(w), 1), n)]


def calcular(df, salida):
    ruta = f"{salida}/embeddings.npy"
    if os.path.exists(ruta):
        emb = np.load(ruta)
        if len(emb) == len(df):
            return emb

    trozos, dueño = [], []
    for i, t in enumerate(df["texto_base"].astype(str)):
        p = pedazos(t)
        trozos += p
        dueño += [i] * len(p)

    e = SentenceTransformer(MODELO).encode(trozos, batch_size=64, show_progress_bar=True)
    dueño = np.array(dueño)
    emb = np.zeros((len(df), e.shape[1]))
    np.add.at(emb, dueño, e)
    emb /= np.bincount(dueño)[:, None]
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)

    np.save(ruta, emb)
    return emb