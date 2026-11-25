import os
import pandas as pd
from analisis import utilidades as u

PP = 4


def generar(br, img, pr, al, dmo, salida):
    out = f"{salida}/oportunidades"
    os.makedirs(out, exist_ok=True)
    filas = []

    def agregar(tipo, hallazgo, sugerencia):
        filas.append({"tipo": tipo, "hallazgo": hallazgo, "sugerencia": sugerencia})

    b = br.get("positivos")
    if b is not None:
        for tema, r in b[b["diferencia"] >= PP].sort_values("diferencia", ascending=False).iterrows():
            agregar("Comunicar más",
                    f"Los visitantes valoran «{tema}» más que {u.nombre(dmo)} ({r.pct_visitantes:.0f}% vs. {r.pct_dmo:.0f}%)",
                    "Incluir este tema y testimonios de visitantes en la promoción")
        for tema, r in b[b["diferencia"] <= -PP].sort_values("diferencia").iterrows():
            agregar("Revisar mensaje",
                    f"{u.nombre(dmo)} promueve «{tema}» más de lo que lo mencionan los visitantes ({r.pct_dmo:.0f}% vs. {r.pct_visitantes:.0f}%)",
                    "Reforzar la experiencia o ajustar el mensaje")

    neg = img.get("negativos")
    if neg is not None:
        for f in [c for c in neg.columns if not c.startswith("visitantes")]:
            tema, v = neg[f].idxmax(), neg[f].max()
            if v >= 10:
                agregar("Atender quejas", f"Queja principal en {u.nombre(f)}: «{tema}» ({v:.0f}% de sus negativos)",
                        "Revisar estas quejas con los responsables del servicio")

    p = pr.set_index("fuente")
    vis = "visitantes" if "visitantes" in p.index else [f for f in p.index if f != dmo][0]
    if p.loc[vis, "pct_precio"] > 2 * p.loc[dmo, "pct_precio"]:
        agregar("Valor por el dinero",
                f"Los visitantes hablan de precio más que {u.nombre(dmo)} ({p.loc[vis, 'pct_precio']:.1f}% vs. {p.loc[dmo, 'pct_precio']:.1f}%)",
                "Comunicar opciones gratuitas y la relación calidad-precio")
    for f, r in p.drop(index=[dmo, "visitantes"], errors="ignore").iterrows():
        if r.hablan_de_precio >= 30 and r.pct_negativo_precio > r.pct_negativo_general + 5:
            agregar("Valor por el dinero",
                    f"{u.nombre(f)} es más negativa al hablar de precio ({r.pct_negativo_precio:.0f}% vs. {r.pct_negativo_general:.0f}% en general)",
                    "Revisar la percepción de precios en esta fuente")

    a = al[~al["fuente"].str.startswith("visitantes")].sort_values("alineacion")
    if len(a) > 1:
        r = a.iloc[0]
        agregar("Mensaje por plataforma",
                f"El mensaje de {u.nombre(dmo)} se parece menos a {u.nombre(r.fuente)} ({r.alineacion:.2f})",
                "Adaptar el contenido al tipo de conversación de esa plataforma")

    res = pd.DataFrame(filas)
    res.to_csv(f"{out}/oportunidades.csv", index=False)
    with open(f"{out}/oportunidades.txt", "w", encoding="utf-8") as fh:
        fh.write("Áreas de oportunidad (lo que sugieren los datos)\n\n")
        for k, r in enumerate(res.itertuples(), 1):
            fh.write(f"{k}. [{r.tipo}] {r.hallazgo}\n   → {r.sugerencia}\n\n")
    return res