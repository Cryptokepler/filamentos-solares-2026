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

## Resultado del paso 1 (20 oct 2026, Kaggle `filamentos-tribunal-paso-1`, 78 min de GPU)

**Comprobación:** la cadena de la ronda 3 vuelve a dar **PQ 0,4405** (TP 820, FP 350, FN 547). Pesos y datos están bien cargados.

**Examen (121 imágenes), con refinador.** Cada candidato se compara con cada experto. Se descartan 1.411 candidatos que repetían a otro de más confianza.

| Franja de confianza | Candidatos | Aciertos | % acierto | IoU medio | Triaje (receta) |
|---|---|---|---|---|---|
| 0,10-0,20 | 679 | 22 | 3,2 % | 0,64 | Basura |
| 0,20-0,30 | 387 | 27 | 7,0 % | 0,70 | Basura |
| 0,30-0,40 | 298 | 32 | 10,7 % | 0,67 | Raya basura |
| 0,40-0,50 | 274 | 37 | 13,5 % | 0,67 | Dudoso |
| 0,50-0,60 | 232 | 38 | 16,4 % | 0,66 | Dudoso |
| 0,60-0,70 | 275 | 62 | 22,5 % | 0,66 | Dudoso |
| 0,70-0,80 | 261 | 81 | 31,0 % | 0,66 | Dudoso (en la raya del 32 %) |
| 0,80-0,85 | 154 | 69 | 44,8 % | 0,65 | Dudoso |
| 0,85-0,90 | 142 | 73 | 51,4 % | 0,67 | Dudoso |
| 0,90-0,95 | 325 | 196 | 60,3 % | 0,67 | Dudoso |
| 0,95-1,00 | 608 | 505 | 83,1 % | 0,69 | **Obvio** |

- **Al alcance:** 1.139 de los 1.367 filamentos reales (83 %) los dibuja algún candidato con IoU > 0,5. La ronda 3 acertaba 820, así que hay **319 más al alcance** (con la silueta del detector sin refinar serían solo 1.043; el refinador ayuda también aquí).
- **Dónde están:** entre 0,3 y 0,8 hay 1.340 candidatos con 250 aciertos (19 %). En bloque restan, por eso el juez de 0,8 era lo mejor sin tribunal. El trabajo del tribunal es encontrar dentro de ese montón los que superan el 32 %, y además quitar los malos de 0,8-0,95 (621 candidatos, 54 % de acierto).
- **Techo con un juez perfecto** (cuenta aproximada, IoU 0,66): aprobar los 250 aciertos de 0,3-0,8 sin ningún invento llevaría el examen de 0,44 a ~0,52. Un juez real quedará por debajo.

**Estudio (586 imágenes, el detector ya las conoce):** en las franjas bajas acierta casi igual que en el examen (0,5-0,6: 16,1 % frente a 16,4 %; 0,6-0,7: 23,7 % frente a 22,5 %; 0,7-0,8: 33,4 % frente a 31,0 %). Solo las franjas altas salen algo mejor (0,95-1: 88 % frente a 83 %). **Consecuencia:** los candidatos dudosos de estudio parecen honestos, así que el juez B podría aprender de 586 imágenes y no solo de 121. Se comprobará midiendo siempre en el examen. Si se confirma, no hace falta el paso 4 (2 detectores, ~5 h de GPU).

## Paso 2: juez B, "la ficha" (21 oct 2026, sin GPU)

Scripts: `analisis/tribunal_sim.py` (etiqueta los candidatos y simula la nota) y `analisis/juez_b.py`.

**El simulador es fiel.** Con la receta de la ronda 3 (confianza ≥ 0,8, uniendo solapes) da PQ 0,4409 frente al 0,4405 real (TP 821 frente a 820). Desde aquí cualquier receta se prueba en segundos.

**Techo con un juez perfecto** (sabe cuántos expertos dibujaron cada candidato): **PQ 0,535-0,540**. Lo reparte así:
- solo añadir los buenos de 0,3-0,8: 0,488;
- solo quitar los malos de 0,8-0,95: 0,479.

El montón de candidatos sí esconde ~+0,10. Entre 0,3 y 0,8 hay 183 candidatos (de 877) que superan la raya, y aciertan el 84 %.

**El juez B (árboles de decisión con 18 rasgos de la ficha)** aprende de los candidatos de estudio y se mide en el examen:
- **Calibración buena.** Cuando dice 35, aciertan el 34 %; cuando dice 55, el 56 %; cuando dice 70, el 65 %. Su nota es una probabilidad de verdad.
- **AUC 0,75 en los dudosos** (0,5 = azar, 1 = perfecto).
- **Pero no sube la nota.** El mejor umbral da 0,443-0,444 frente a 0,441 de base, dentro del ruido (±0,015). Con el umbral elegido en una mitad del examen y medido en la otra: 0,439-0,447 frente a 0,441-0,441.
- **Por qué no:** de los 877 dudosos bajos (0,3-0,8), la ficha solo encuentra 55 con más del 40 % de acierto (y aciertan el 41 %). Los 183 buenos de verdad no se distinguen por tamaño, oscuridad, forma, posición ni vecinos. La confianza del detector ya contiene casi todo lo que dice la ficha: el juez B "sin confianza" se queda en AUC 0,74, casi igual que con ella.

**Conclusión:** el margen existe (+0,10 con un juez perfecto), pero no está en los números de la ficha. Si está en algún sitio, está en los píxeles. Siguiente pieza: **juez A, "el ojo"**, una red pequeña que mire el recorte de cada candidato.

## Paso 3: juez A, "el ojo", y el tribunal A+B (21 oct 2026, sin GPU)

Scripts: `analisis/cajas_juez_a.py`, `analisis/juez_a.py` y `analisis/tribunal_ab.py`.

**Montaje del juez A.**
- **Datos:** recortes de 128 px (imagen + silueta) de los 9.253 candidatos con confianza ≥ 0,3. Las imágenes son las del Mac de jhonatan, y los recortes son idénticos a los del notebook.
- **Red:** 5 bloques convolucionales entrenados desde cero en CPU, durante 14 vueltas (45 min).
- **Entrenamiento y medida:** aprende con los 7.583 candidatos de estudio y se mide con los 1.670 del examen.

| Juez | AUC en los dudosos | Calibración | Mejor PQ del examen | Umbral elegido en una mitad y medido en la otra |
|---|---|---|---|---|
| Base (ronda 3) | — | — | 0,4409 | 0,4405 / 0,4413 |
| A (el ojo) | 0,70 | Se pasa de optimista (dice 69, acierta 50) | 0,433 | 0,423 / 0,435 (peor) |
| B (la ficha) | 0,75 | Buena | 0,443 | 0,441 / 0,441 (igual) |
| Tribunal: media de A y B | 0,76 | Buena | 0,444 | 0,445 / 0,438 (igual) |

- A y B coinciden poco: correlación 0,54, y en 93 de 1.295 dudosos discrepan en más de 40 puntos. Ven cosas distintas, como queríamos, pero juntos solo suben el AUC de 0,75 a 0,76.
- **Ningún juez mueve la nota fuera del ruido (±0,015).**

**Por qué el techo de 0,54 no se alcanza.** El juez perfecto sabe qué dibujó *el experto concreto de esa imagen*. Pero los expertos discrepan entre ellos: experto contra experto da 0,34 de PQ, y en los pequeños el 61 % lo marca uno solo. Un juez que mira la imagen solo puede saber qué dibujaría *un experto cualquiera*. Ese techo está mucho más bajo y lo mide el AUC: con 0,75, la separación es parcial. Lo que queda en el montón de dudosos es, sobre todo, la duda humana.

**Conclusión:** el tribunal está montado y medido, y el simulador queda listo para cualquier idea futura. Con los jueces de ahora no supera a la ronda 3. Palancas que quedan, ya medidas como posibles:
- **Mejorar el borde (SQ):** cada +0,01 de SQ suma ~+0,0065 de PQ.
- **Elegir a qué estilo de experto apuntar:** según con qué experto nos comparen, la nota va de 0,41 a 0,62.
- **Un juez A más fuerte** (red preentrenada, con GPU): sin garantía. El AUC de la ficha ya es mejor que el del ojo entrenado desde cero.
