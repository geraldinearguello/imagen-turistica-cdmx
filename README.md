# Imagen turística: DMO vs. visitantes

Compara la **imagen proyectada** por una organización de destino (DMO), como SECTUR CDMX, con la **imagen percibida** por los visitantes en plataformas como TripAdvisor, Reddit e Instagram.

El programa limpia los textos, detecta publicaciones atípicas, clasifica el sentimiento, encuentra temas y compara qué tanto coincide lo que dice la DMO con lo que dicen los visitantes. Los resultados se ven en un tablero interactivo.

## Estructura

```
proyecto_imagen_cdmx/
├── main.py            # corre todo el análisis
├── tablero.py         # tablero interactivo (Streamlit)
├── unificar.py        # paso 0: junta varias bases en una
├── requirements.txt
├── analisis/          # un archivo por etapa del análisis
│   ├── limpieza.py
│   ├── exploracion.py
│   ├── embeddings.py
│   ├── atipicos.py
│   ├── sentimiento.py
│   ├── temas.py
│   ├── comparacion.py
│   ├── precio.py
│   ├── imagen.py
│   ├── oportunidades.py
│   └── utilidades.py
├── datos/             # bases (solo se sube ejemplo.csv)
├── salidas/           # resultados (se crea al correr)
└── .streamlit/        # tema oscuro del tablero
```

## Instalación

Requiere **Python 3.13**. Es opcional, pero recomendable, crear un entorno virtual:

```
python3 -m venv .venv
source .venv/bin/activate
```

Instalar librerías y modelos de spaCy:

```
pip install -r requirements.txt
python -m spacy download es_core_news_sm
python -m spacy download en_core_web_sm
```

La primera vez que se corre, el programa descarga de Hugging Face los modelos de embeddings y de sentimiento, así que necesita internet. Después ya quedan guardados en la computadora.

## Datos de entrada

Un CSV con, al menos:

| Columna | Contenido |
|---|---|
| texto | el texto de cada publicación o reseña |
| fuente | de dónde viene (ej. `sectur`, `tripadvisor`, `reddit`) |
| calificacion | *(opcional)* estrellas de 1 a 5 |

Debe haber **al menos 2 fuentes**, y una de ellas debe ser la DMO.

> Las bases completas no se incluyen. `datos/ejemplo.csv` es una muestra de 800 textos (200 por fuente), sin usuarios, menciones, enlaces ni correos, para fines académicos y de prueba.

Si las bases vienen en archivos separados, primero se juntan con:

```
python unificar.py
```

Esto crea `datos/base_unificada.csv`.

> `unificar.py` funciona con las cuatro bases originales, que no se incluyen.


## Uso

**1. Correr el análisis**

```
python main.py -i datos/base_unificada.csv -c texto -f fuente -d sectur -l es -t "Imagen turística de la CDMX" -p viridis --calificacion calificacion
```

| Parámetro | Qué es |
|---|---|
| `-i` | archivo CSV de entrada |
| `-c` | columna con el texto |
| `-f` | columna con la fuente |
| `-d` | nombre de la DMO dentro de la columna fuente |
| `-l` | idioma principal (`es`, `en`, `fr`, `pt`) |
| `-t` | título del tablero |
| `-p` | paleta: `viridis`, `cividis`, `plasma` o `inferno` (aptas para daltonismo) |
| `--calificacion` | *(opcional)* columna con estrellas |
| `-o` | *(opcional)* carpeta de resultados; por defecto `salidas` |

Para probar rápido con la base de ejemplo:

```
python main.py -i datos/ejemplo.csv -c texto -f fuente -d sectur -l es -t "Prueba" -p cividis --calificacion calificacion -o salidas_ejemplo
```

**2. Abrir el tablero**

```
streamlit run tablero.py
```

Si los resultados están en otra carpeta:

```
streamlit run tablero.py -- salidas_ejemplo
```

El tablero solo lee lo que hay en la carpeta de resultados, así que primero hay que correr `main.py`.

## Qué hace cada etapa

1. **Limpieza:** quita duplicados, enlaces, hashtags y textos vacíos; detecta idioma y lematiza.
2. **Exploración:** palabras, bigramas y trigramas más frecuentes, y nubes de palabras.
3. **Embeddings:** representa cada texto como un vector de 384 números (`paraphrase-multilingual-MiniLM-L12-v2`).
4. **Atípicos:** Isolation Forest + LOF; un texto es atípico si los dos métodos coinciden.
5. **Sentimiento:** modelo XLM-RoBERTa. Si hay estrellas, 1–2 cuenta como negativo y 4–5 como positivo.
6. **Temas:** TF-IDF + NMF. El número de temas se elige con coherencia NPMI.
7. **Comparación:** similitud coseno entre la DMO y cada fuente, y brechas por tema.
8. **Precio:** detecta textos que hablan de precio, costo o valor, y su tono.
9. **Imagen:** reparto de temas positivos y negativos por fuente.
10. **Oportunidades:** reglas simples que resumen los hallazgos principales.

## Nota importante: nombres de los temas

Los nombres de los temas (por ejemplo, "Museos y arte") **se ponen a mano** en el diccionario `NOMBRES` de `analisis/temas.py`, después de revisar las palabras clave y los comentarios de cada tema.

Con una base distinta, los temas que no estén en el diccionario se nombran automáticamente con sus 3 palabras clave principales.

## Autora

Rosario Geraldine Vasquez Arguello  
Ingeniería de Datos e Inteligencia Artificial, Universidad de Guanajuato  
Visualización de la Información

Contacto: [geralarguello17@gmail.com](mailto:geralarguello17@gmail.com)