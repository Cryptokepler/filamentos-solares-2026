# Resumen de las rondas: qué funcionó y qué no

Escrito el 20 oct 2026, después de la ronda 4; puesto al día el 25 oct 2026, después del jurado ronda 5 + ronda 7 (Kaggle 0,39). Junta en un sitio los datos del [diario](experimentos.md) para pensar en las siguientes soluciones. Todas las notas son PQ en el mismo examen fijo de 121 imágenes de train (nunca del test). El examen tiene un ruido de ±0,015, así que una diferencia menor que eso entre rondas no significa nada.

## 1. Las rondas de un vistazo

| Ronda | Qué probamos | PQ examen final | TP / FP / FN | Partidos / fusionados | Kaggle público | GPU |
|---|---|---|---|---|---|---|
| 0 | Reglas sin red (zonas oscuras) | 0,098 | — | — | — | 0 |
| 1 | U-Net (segmentación) a 1024 + juez | 0,384 | 826 / 648 / 541 | 93 / 27 | **0,34** | 4 h |
| 2 | Mask R-CNN (detector) a 1024 + juez + refinador a 2048 | **0,445** | 827 / 341 / 540 | 18 / 43 | no enviado | 2 h 40 |
| 3 | Lo mismo, con el detector entrenado hasta la meseta, reglas 4 y 5 y más umbrales | 0,4405 | 820 / 350 / 547 | 19 / 51 | **0,38** | 5 h 16 |
| 4 | + detector especialista en pequeños a 2048 por ventanas | 0,4391 | 903 / 524 / 464 | 24 / 54 | no se envía | 2 h 47 |
| 5 | YOLO11m-seg (detector de 2024) en lugar de Mask R-CNN + refinador de la ronda 3, umbral 0,2 | **0,4532** | 922 / 481 / 445 | — | no enviado | 2 h 41 |
| Curva | Ronda 5 con el 25 / 50 / 75 % de las imágenes de estudio (146 / 293 / 440) | 0,428 / 0,447 / 0,440 | 811/416/556 · 846/363/521 · 844/412/523 | — | no se envía | 5 h 18 |
| 5 (umbral 0,25) | La ronda 5 con el umbral del juez en 0,25 (sin GPU) | 0,4560 | 878 / 381 / 489 | — | no enviado | 0 |
| 5 + TTA | La ronda 5 mirando la foto volteada de 4 formas | 0,4485 | 873 / 404 / 494 | — | no se envía | 17 min |
| 7 | YOLO con la foto vista de tres formas (original, contraste local, "solo lo oscuro") | 0,4557 | 871 / 366 / 496 | — | no enviado | 2 h 52 |
| **Jurado 5 + 7** | Los YOLO de las rondas 5 y 7 votan: pasa lo que ven los dos con confianza ≥ 0,2 (sin GPU) | **0,4607** (honesto) | 892 / 382 / 475 | — | **0,39** | 0 |

Referencias fijas: experto contra experto 0,34; techo por resolución a 1024 px 0,98.

**En una frase:** el único salto grande fue pasar de U-Net a detector + refinador (ronda 2, +0,06). Después, YOLO (ronda 5, +0,01) y el jurado ronda 5 + ronda 7 (+0,005 más) han ido sumando poco a poco: Kaggle 0,34 → 0,38 → 0,39.

## 2. Pieza por pieza

| Pieza | Qué probamos | Resultado medido | Veredicto |
|---|---|---|---|
| **Tipo de modelo** | U-Net (ronda 1) → Mask R-CNN (ronda 2) | 0,384 → 0,445. Los inventados bajan casi a la mitad (648 → 341) y se parte mucho menos (93 → 18 partidos). | ✅ Funcionó |
| **Dibujar las máscaras a 2048** | Antes se dibujaban a 1024 con PIL y salían engordadas | Con dibujos perfectos: PQ 0,82 → 0,89. Entró junto con el detector en la ronda 2, así que no se puede separar su efecto. | ✅ Corregido |
| **Refinador** (U-Net de recortes a 2048) | Redibuja el borde de cada filamento aprobado | Suma siempre lo mismo: +0,054 (ronda 2), +0,052 (ronda 3), +0,050 (ronda 4). Sube la forma (SQ 0,64 → 0,68) y rescata ~50 aciertos (771 → 820). | ✅ Funciona y es estable |
| **Umbral del refinador** | 0,3 / 0,4 / 0,5 | 0,435 / 0,439 / 0,4405 | ➖ Da casi igual (se queda 0,5) |
| **Entrenar más el detector** | 12 vueltas fijas (ronda 2) → hasta la meseta, 28 vueltas (ronda 3) | Mejor examen 0,370 en las dos. La curva se aplana hacia la vuelta 10-16. | ❌ Sin efecto: ya estaba en su techo |
| **Confianza del juez** | De 0,5 a 0,9 | Gana siempre 0,8 (0,389). Con 0,85 quita 70 inventados pero también 56 aciertos; con 0,9 baja a 0,369. | ➖ 0,8 es el punto justo; más no ayuda |
| **Unir solapes en el juez** | Unir o no | Con confianza 0,8: 0,389 uniendo frente a 0,375 sin unir. Con 0,5: 0,352 frente a 0,306. | ✅ Unir siempre |
| **Tamaño mínimo 330 px** | 0 o 330 px | Diferencia de ±0,001 | ➖ Da igual |
| **Regla 4** (descartar lo muy negro: mancha solar) | Ronda 3 | No quitó nada: el detector no propone manchas solares | ➖ No actúa |
| **Regla 5** (descartar hilos de más de 10:1) | Ronda 3 | −0,012: quitó 31 inventados y 48 aciertos | ❌ Empeora |
| **Especialista con lupa a 2048** | Ronda 4: ventanas de 1024 sobre la imagen original, centradas en pequeños | Antes del refinador: 0,389 → 0,393. Después: 0,4390 → 0,4391. Encuentra **88** filamentos que el general no veía, pero **inventa ~170**. Su examen no mejoró de la vuelta 2 (0,382) a la 14 (0,385). | ❌ Ve más, pero no distingue lo real |
| **YOLO11-seg en lugar de Mask R-CNN** | Ronda 5 | 0,4405 → 0,4560 (umbral 0,25). Ve más (922 aciertos con umbral 0,2) y su confianza está mejor calibrada. | ✅ Funcionó, poco |
| **Más imágenes del mismo tipo** | Curva de aprendizaje: 25 / 50 / 75 / 100 % | 0,428 / 0,447 / 0,440 / 0,453: se aplana hacia ~300 imágenes | ❌ No es la palanca |
| **Jurado de YOLO iguales** | YOLO entrenados con 25-100 % de las fotos, "2 de 3" | 0,454-0,458 honesto: fallan en los mismos casos | ❌ Sin efecto |
| **Borde promedio entre expertos** | Medido sin GPU | Nuestro SQ 0,682 ya supera el parecido entre expertos (0,643-0,650); ±1 px baja a ~0,43 | ❌ Sin margen |
| **Mirar más fino (1536 px / modelo grande)** | Estudio de los escapados sin GPU | Solo 101 de 489 no se ven nada y solo 3 son finos; techo +0,034 sin inventar | ❌ No se lanza |
| **Post-procesado (unir trozos, área mínima)** | Propuesta externa "ronda 8", medida sin GPU | Unir a 5-10 px igual; a 20-40 px peor; área mínima ≥ 300 px peor | ❌ Sin margen |
| **Foto volteada al predecir (TTA)** | 4 vistas volteadas, media de confianzas | 0,4485: +23 inventados | ❌ Empeora |
| **Tres vistas de la foto (idea de jhonatan)** | Ronda 7: original + contraste local + "solo lo oscuro" como canales | Sola 0,4557 (igual). Pero crea un YOLO que se equivoca en sitios distintos | ➖ Sola no; útil en jurado |
| **Jurado ronda 5 + ronda 7** | Pasa lo que ven los dos (conf ≥ 0,2) | 0,4607 honesto (+0,009), **Kaggle 0,39**. Con un tercero (YOLO 50 %, "3 de 3") 0,4630. Añadir el Mask R-CNN de la ronda 3 con el mismo umbral empeora (0,449): su confianza va en otra escala | ✅ Funcionó, poco |

## 3. Dónde están los errores que quedan (ronda 3, la mejor enviada)

**Lo que se escapa (FN 547), por tamaño del filamento real:**

| Tamaño | % encontrado | Escapados del todo |
|---|---|---|
| < 330 px | 24 % | 52 |
| 330-1k px | 51 % | 160 |
| 1k-3k px | 68 % | 80 |
| 3k-10k px | 71 % | 26 |
| > 10k px | 72 % | 3 |

El 85 % de lo que se escapa mide menos de 3.000 px. La ronda 4 atacó justo esto y comprobó que **verlos no basta**: al buscarlos con lupa aparecen el doble de falsos.

**Lo que se escapa, por otros rasgos:**
- **Oscuridad:** los casi invisibles (más del 95 % del brillo del fondo) se encuentran solo en el 33 % de los casos; los oscuros normales, en el 69 %.
- **Zona del disco:** cerca del borde (80-90 % del radio) se encuentran el 51 %; en el resto, el 61 %.
- **Telescopio:** entre 53 % (Cerro Tololo) y 69 % (Udaipur). No es una causa grande.

**Lo que se inventa (FP 350):** 194 son "casi" (el detector vio un filamento real pero lo partió, lo unió o lo dibujó mal) y 156 son inventos puros, sobre todo de 330 a 3.000 px y de oscuridad media.

## 4. Comparado con lo que hacen los expertos entre sí

Analizado filamento a filamento (`analisis/casi_ronda3.py`):

| Qué le pasa al filamento real | Ronda 3 | Un experto frente a otro |
|---|---|---|
| Acierto | 59 % | 47 % |
| Escapado del todo | 23 % | 21 % |
| Fusionado con otro | 7,0 % | 6,7 % |
| Mal dibujado | 9,7 % | 22 % |
| Partido | 1,2 % | 3,0 % |

- En partir, unir y dibujar **ya estamos al nivel de un experto**, o mejor.
- En los filamentos pequeños los expertos coinciden poco: en los de menos de 330 px, otro experto coincide solo el 41 % de las veces. En los de más de 4.000 px, el 70 %.
- **Según el experto con quien nos comparen, la nota cambia mucho:** con los "prudentes" sacamos 0,52-0,62; con los "generosos", 0,41. Uno de ellos dibuja 11,8 filamentos por imagen, y nosotros 7,0. Nuestro modelo dibuja unos 6 por imagen y los expertos 7-8 de media.

## 5. Conclusiones con los datos

1. **El detector general está en su techo** (0,370 con el juez básico, 0,389 con el juez a 0,8) desde la ronda 2. Ni más entrenamiento ni más resolución lo mueven.
2. **El refinador funciona** y suma +0,05 de forma estable.
3. **Las reglas a mano del juez no ayudan.** La seguridad del detector tampoco separa bien lo real de lo inventado: al subir la confianza caen aciertos e inventados casi al mismo ritmo.
4. **Lo que falta no es vista, es criterio.** Los pequeños y los débiles son justo donde los expertos dudan. Un modelo entrenado con todos los expertos a la vez aprende una "media" de criterios que no coincide del todo con ninguno.
5. **Nuestro examen local predice bien la dirección de Kaggle:** 0,384 → 0,4405 en local; 0,34 → 0,38 en Kaggle (unos 0,04-0,06 menos).

## 6. El mejor envío: jurado ronda 5 + ronda 7 (Kaggle 0,39, 25 oct 2026)

Cómo se hace, paso a paso:

1. **Dos detectores YOLO11m-seg a 1024 px**, entrenados igual con las 586 imágenes de estudio:
   - ronda 5: la foto normal (tres copias iguales en los canales de color);
   - ronda 7: la foto vista de tres formas (`filamentos/yolo.py`, `tres_vistas`): original, contraste local (CLAHE) y "solo lo oscuro y fino" (black-hat).
2. Cada YOLO propone todo lo que ve desde confianza 0,1, y cada propuesta se redibuja a 2048 con el **refinador** de la ronda 3 (`filamentos/candidatos.py`, `procesar`).
3. **Jurado** (`analisis/jurado_envio.py`, misma regla que `analisis/jurado_yolo.py`): las propuestas de los dos que se solapan (IoU > 0,5) son "el mismo caso". Pasa si **los dos** lo ven con confianza ≥ 0,2. Se queda con la silueta del más seguro, y se unen los que se tocan.

| | Examen | Aciertos | Inventados | Escapados | Kaggle |
|---|---|---|---|---|---|
| Ronda 5 sola | 0,4560 (0,452 honesto) | 878 | 381 | 489 | — |
| Ronda 7 sola | 0,4557 | 871 | 366 | 496 | — |
| **Jurado 5 + 7** | **0,4607 honesto** | 892 | 382 | 475 | **0,39** |

Por qué funciona: dos YOLO casi iguales fallan en los mismos sitios, y votar no sirve de nada. El de tres vistas mira la foto de otra manera, así que sus errores son algo distintos, y lo que ven los dos es más fiable. Así el umbral puede bajar a 0,2 (más aciertos) sin que se disparen los inventados. Examen → Kaggle: −0,07, como en la ronda 3 (0,4405 → 0,38).

Notebooks de Kaggle: `filamentos-ronda-5-yolo` y `filamentos-ronda-5-tta-envio` (candidatos del test de la ronda 5), y `filamentos-ronda-7-tres-vistas`. Archivo enviado: `submission_jurado_r5_r7.csv` (1.212 filamentos en 177 de 180 imágenes).

## 7. Lo que queda por probar (25 oct 2026, para discutir)

| Idea | Estado | Qué ataca | Dato que la apoya |
|---|---|---|---|
| Más jueces que miren distinto (otra "vista", otro tamaño de modelo, otra arquitectura con su propio umbral) | Pendiente | Inventos y dudosos | El jurado solo suma si los jueces fallan en sitios distintos (5 + 7 sí; YOLO iguales no). Con un tercero ya da 0,4630 |
| Reentrenar al final con las 707 imágenes (también las del examen) | Pendiente | Un 20 % más de datos | La curva dice que suma poco, pero es lo último antes de cerrar; no se puede medir en local |
| Datos externos: magnetogramas y otros telescopios GONG, solo para entrenar | Pendiente | Débiles y dudosos | Los filamentos están sobre la línea donde el campo magnético cambia de signo. En el test solo se puede usar la foto H-alfa |
| Preentrenar con fotos H-alfa sin etiquetar | Pendiente | Que el modelo "conozca" mejor el Sol | Permitido por los organizadores |
| Juez que aprende | Probado (tribunal, 21 oct) | — | Sin mejora: 0,444 |
| Estilo de un experto | Descartado | — | No se puede adivinar qué grupo anotó cada imagen del test |
| Lupa de contraste | Probado (ronda 7) | — | Sola no sube; útil en el jurado |
| Varios detectores que votan | Probado | — | Iguales no suman; distintos sí (jurado 5 + 7) |
| Usar las 2 GPU T4 | Pendiente | Velocidad | Ahora se usa 1 de 2 |
| Informe final (4 páginas) y repo público | Pendiente | 30 % de la nota | Cierre el 15 nov 2026 |
