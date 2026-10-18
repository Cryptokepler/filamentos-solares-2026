# Diario de experimentos

Un cambio por experimento. Nota = PQ en las 121 imágenes de examen (validación fija). Ver [roadmap](roadmap-entrenamiento.md).

| # | Fecha | Qué se cambió | PQ examen | TP / FP / FN | Partidos / fusionados | Qué se vio |
|---|---|---|---|---|---|---|
| 0 | 2026-10-18 | Línea base de reglas (sin red) | 0,098 | | | Muchas manchas oscuras que no son filamentos |
| 1 | 2026-10-18 | Primer entrenamiento: U-Net resnet34, 1024 px, 40 épocas | 0,363 red sola (época 40) · **0,384** con juez (umbral 0,6, área ≥ 330, sin unir) | 826 / 648 / 541 | 93 / 27 | Supera el 0,34 experto-vs-experto. Seguía mejorando en la época 40 (sin techo). Lo flojo es detectar (RQ 0,58) más que dibujar (SQ 0,66): inventa casi tantos como acierta. Parte ~3 veces más que fusiona. Ojo: época y juez elegidos mirando el mismo examen, la nota es algo optimista. **Kaggle público: 0,34** (envío 18 oct 22:57; líderes 0,55–0,64). Run de 4 h: 1 h 35 min preparando imágenes y 1 h prediciendo el test (filtro de mediana lento), solo 50 min entrenando. |
