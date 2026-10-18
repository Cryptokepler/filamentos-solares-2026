# Segmentación de filamentos solares 2026

Solución para la competición [Solar Filament Segmentation Challenge 2026](https://www.kaggle.com/competitions/filament-segmentation-2026) (Kaggle, IEEE BigData Cup).

## Idea
Imitar cómo trabajan los expertos: **observador → revisor → juez**.
1. **Observador:** propone todo lo que parece un filamento.
2. **Revisor:** varios modelos votan cada candidato.
3. **Juez:** reglas observadas en los datos (`docs/reglas.md`) deciden los casos dudosos.

## Estructura
- `filamentos/metrica.py`: Panoptic Quality como la define la competición, con diagnóstico de fragmentación y fusión.
- `filamentos/datos.py`: lectura de MAGFiLO y conversión polígono / máscara / RLE.
- `filamentos/clasico.py`: línea base sin aprendizaje (zonas oscuras + reglas).
- `visor/`: visor web de imágenes y anotaciones.
- `analisis/`: estudios de los datos.
- `docs/`: plan, reglas y [base de conocimiento](docs/conocimiento/README.md) con fuentes científicas.
- `notebooks/`: notebook de entrenamiento para Kaggle.

## Resultados locales
| Método | PQ | Datos |
|---|---|---|
| Experto contra experto | 0,340 | 296 imágenes con 2–3 anotadores |
| Línea base clásica | 0,098 | 100 imágenes de train |

## Datos
Descarga los datos con la API de Kaggle (hay que aceptar las reglas de la competición):
```
kaggle competitions download -c filament-segmentation-2026 && unzip filament-segmentation-2026.zip
```
Los datos no se incluyen en el repositorio.

## Visor
```
cd visor && python3 build_visor.py ../MAGFiLO_1.0_Kaggle_2026
cd .. && python3 -m http.server 8000   # abrir http://localhost:8000/visor/
```

## Entrenar el observador en Kaggle
`notebooks/entrenar_observador.ipynb` lleva el código incrustado (se regenera con `python3 notebooks/generar_notebook.py`).
1. En Kaggle: *Create → New Notebook → File → Import Notebook* y subir el `.ipynb`.
2. *Add Input*: la competición `filament-segmentation-2026`.
3. *Settings*: Accelerator **GPU T4 x2** o **P100**, Internet **On**.
4. *Run All*. Imprime la PQ de validación y deja `submission.csv` en *Output* (no se envía solo).
