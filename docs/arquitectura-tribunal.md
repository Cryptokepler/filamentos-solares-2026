# Arquitectura del tribunal (decidida con jhonatan, 20 oct 2026)

Idea de jhonatan: que el detector deje de decidir y solo proponga, y que la decisión la tomen jueces que se contrastan entre sí. Todo con una receta numérica, sin decisiones "a criterio".

## Las piezas

| Pieza | Qué hace | De dónde aprende |
|---|---|---|
| **Detector generoso** | Propone todo lo sospechoso: confianza desde 0,2 en vez de 0,8. | Lo que ya aprendió en la ronda 3 (dibujos de los expertos). No se reentrena. |
| **Clasificador (triaje)** | Separa los candidatos en obvios, basura y dudosos según la confianza del detector. | No aprende: es una **tabla medida**. En el paso 1 se mide cuánto acierta cada franja de confianza, y los cortes salen de esa tabla con la regla de abajo. |
| **Juez A, "el ojo"** | Da una nota de 0 a 100 mirando el recorte a 2048 alrededor del candidato. | Los píxeles. Maestro: cuántos expertos dibujaron ese candidato. |
| **Juez B, "la ficha"** | Da una nota de 0 a 100 mirando números: tamaño, oscuridad, forma, distancia al borde, si está en grupo, confianza del detector. | Los rasgos que elegimos con `docs/conocimiento` (guía de MAGFiLO y física). Mismo maestro. |

## La receta: de dónde sale el umbral (matemática, no suposición)

La nota es PQ = suma de IoU de los aciertos / (aciertos + ½ inventos + ½ escapados). Cada predicción se compara con cada experto de la imagen. Frente a cada experto, añadir un candidato tiene este efecto:
- **Si acierta** (probabilidad p): suma su IoU (q) arriba. Abajo suma +1 acierto y −1 escapado, es decir, **+½**.
- **Si falla:** arriba no suma nada. Abajo suma +1 invento, es decir, **+½**.

Abajo siempre suma ½, y arriba suma p·q de media. La nota sube si y solo si **p·q / ½ > PQ actual**, es decir:

> **Se aprueba un candidato si p > PQ / (2·q)**

Con las cifras medidas en la ronda 3 (PQ 0,44; IoU medio de los aciertos q = 0,68, igual en todos los tamaños: 0,68-0,73):

> **Umbral = 0,44 / (2 × 0,68) = 0,32 → 32 sobre 100**

Es como una apuesta: cada candidato es un boleto que cuesta ½ y paga 0,68 si acierta. Compensa comprarlo si acierta más de 32 veces de cada 100. El umbral se recalcula cuando cambien PQ o q: si mejora el borde (q sube), el umbral baja y entran más candidatos.

Comprobación con lo ya visto: lo que añadió el especialista de la ronda 4 acertaba el 34 %, casi en la raya de 32, y por eso la nota no se movió.

## Tabla de puntuación

**Nota de cada juez (0-100)** = probabilidad estimada de que un experto dibuje ese candidato. Antes de usarla se comprueba que está **calibrada**: de los candidatos a los que un juez pone 60, deben acertar ~60 de cada 100 en el examen. Si no, se corrige con la tabla de calibración.

**Nota del tribunal** = media de A y B. Si uno de los dos resulta más fiable en el examen, se pondera, pero el peso sale de la medición.

| Nota del tribunal | Decisión | Por qué |
|---|---|---|
| ≥ 70 | **Sí** | Muy por encima del umbral |
| 32-69 | **Sí** | Suma nota en promedio |
| < 32 | **No** | Resta nota en promedio |
| A y B difieren en más de 40 puntos | Se decide con la media, pero se apunta | Para revisar a mano qué ve uno y no el otro |

**Cortes del clasificador (triaje)**, con la tabla medida en el paso 1:
- **Obvio:** franjas de confianza que aciertan ≥ 75 %. Ningún juez razonable las bajaría de 32. Pasan directo.
- **Basura:** franjas que aciertan ≤ 10 %. Ningún juez razonable las subiría de 32. Fuera.
- **Dudoso:** el resto. Va al tribunal.

## Las rondas 3 y 4 medidas con la fórmula (20 oct 2026)

**La fórmula.** Cada franja de candidatos tiene un % de acierto: aciertos que aporta / (aciertos + inventos que aporta). La franja suma nota si ese % supera el **umbral = PQ / (2·SQ)**. Las cifras salen de los logs de Kaggle (examen de 121 imágenes, uniendo solapes).

**Ronda 3, detector general** (antes del refinador: PQ 0,389, SQ 0,640 → umbral **30 %**):

| Franja de confianza | Aciertos que aporta | Inventos que aporta | % acierto | ¿Supera el 30 %? | Lo que pasó en la realidad |
|---|---|---|---|---|---|
| 0,9 o más | 793 | 426 | 65 % | Sí | Se usa |
| 0,85-0,9 | 59 | 62 | 49 % | Sí | Quitarla bajaba la nota (0,380 → 0,369) |
| 0,8-0,85 | 56 | 70 | 44 % | Sí | Quitarla bajaba la nota (0,389 → 0,380) |
| 0,7-0,8 | 60 | 130 | 32 % | Raya | Añadirla dejaba la nota igual (0,389 → 0,387) |
| 0,6-0,7 | 9 | 124 | 7 % | No | Añadirla bajaba la nota (0,387 → 0,370) |

La fórmula explica sola por qué el mejor corte fue siempre 0,8: es justo donde el % de acierto cruza el umbral.

**Ronda 4, especialista** (lo que añade sobre el general; umbral 30 %):

| Franja de confianza | Aciertos | Inventos | % acierto | ¿Supera el 30 %? |
|---|---|---|---|---|
| 0,9 o más | 22 | 27 | 45 % | Sí |
| 0,8-0,9 | 34 | 86 | 28 % | Raya |
| 0,7-0,8 | 36 | 74 | 33 % | Raya |
| 0,6-0,7 | 15 | 90 | 17 % | No |
| 0,5-0,6 | 21 | 114 | 18 % | No |

El especialista tenía muy poco por encima de la raya: solo su franja más alta (49 candidatos) sumaba de verdad. Por eso subió +0,004 antes del refinador y nada después.

## Las perillas (lo que se ajusta)

| Perilla | Qué controla | Valor inicial | Cómo se fija |
|---|---|---|---|
| Confianza mínima del detector | Cuánto dispara | 0,2 | Lo más baja posible sin que el tribunal se ahogue en basura |
| Corte "obvio" | Qué pasa sin juicio | franja con ≥ 75 % de acierto | Tabla del paso 1 |
| Corte "basura" | Qué se tira sin juicio | franja con ≤ 10 % de acierto | Tabla del paso 1 |
| Pesos de los jueces A y B | Cuánto cuenta cada opinión | 50 / 50 | Calibración en el examen |
| Umbral de aprobación | Sí o no | PQ / (2·SQ) ≈ 32 | Se recalcula solo cuando cambian PQ o SQ |
| Umbral por tamaño (opcional) | Exigir más a los pequeños si se dibujan peor | igual para todos | Con el SQ medido por tamaño |

**Lo importante:** una vez guardados los candidatos del paso 1, mover estas perillas **no gasta GPU**. Se prueban miles de combinaciones aquí en segundos, midiendo la nota del examen con cada una. Solo se vuelve a Kaggle con la combinación ganadora. (Ojo: probar muchas combinaciones en el mismo examen infla la nota. Se separa el examen en dos mitades: con una se ajusta y con la otra se comprueba.)

## Plan por pasos (uno a la vez, midiendo)

1. **Medir** (solo predicción, ~30-40 min de GPU, sin entrenar). Pasar el detector de la ronda 3 en modo generoso por las imágenes del examen y guardar cada candidato con su ficha, su recorte y cuántos expertos lo dibujaron. Sale la tabla de acierto por franja de confianza y cuántos de los 547 escapados hay entre los candidatos.
2. **Juez B** (la ficha), entrenado sin GPU con validación cruzada por imagen. Medir.
3. **Juez A** (el ojo), solo si el paso 2 deja margen. Medir el tribunal A+B.
4. Si faltan candidatos honestos para entrenar al juez A: 2 detectores, cada uno con la mitad de las imágenes (~5 h de GPU).
