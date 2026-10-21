# Diario de experimentos

Un cambio por experimento. Nota = PQ en las 121 imágenes de examen (validación fija). Ver [roadmap](roadmap-entrenamiento.md).

| # | Fecha | Qué se cambió | PQ examen | TP / FP / FN | Partidos / fusionados | Qué se vio |
|---|---|---|---|---|---|---|
| 0 | 2026-10-18 | Línea base de reglas (sin red) | 0,098 | | | Muchas manchas oscuras que no son filamentos |
| 1 | 2026-10-18 | Primer entrenamiento: U-Net resnet34, 1024 px, 40 épocas | 0,363 red sola (época 40) · **0,384** con juez (umbral 0,6, área ≥ 330, sin unir) | 826 / 648 / 541 | 93 / 27 | Supera el 0,34 experto-vs-experto. Seguía mejorando en la época 40 (sin techo). Lo flojo es detectar (RQ 0,58) más que dibujar (SQ 0,66): inventa casi tantos como acierta. Parte ~3 veces más que fusiona. Ojo: época y juez elegidos mirando el mismo examen, la nota es algo optimista. **Kaggle público: 0,34** (envío 18 oct 22:57; líderes 0,55–0,64). Run de 4 h: 1 h 35 min preparando imágenes y 1 h prediciendo el test (filtro de mediana lento), solo 50 min entrenando. |
| 2 | 2026-10-19 | Cambio de método: detector Mask R-CNN (1024 px, un ejemplo por experto, 12 épocas) + refinador U-Net resnet18 en recortes de 2048 (4 vistas). Máscaras de entrenamiento dibujadas a 2048 (antes PIL las engordaba). Notebook `entrenar_detector.ipynb`, Kaggle `filamentos-detector-ronda-2` | 0,391 detector solo (confianza ≥ 0,8, unir solapes) · **0,445** con refinador (umbral 0,5) | 827 / 341 / 540 | 18 / 43 | +0,061 sobre la ronda 1. El refinador sube forma (SQ 0,64 → 0,68) y detección (RQ 0,61 → 0,65). Inventa la mitad que la ronda 1 (341 vs 648). Ahora fusiona más de lo que parte (al revés que la ronda 1). Seguía mejorando en la época 12, y la confianza y el umbral del refinador ganadores están en el borde de lo probado (0,8 y 0,5): hay que ampliar la búsqueda. Lo que se le escapa: pequeños (< 330 px: 25 % acierto; 330-1k: 51 %) y débiles (casi fondo > 0,95: 28 %). Por telescopio no hay diferencias grandes (57-69 %). Run de 160 min: detector 12 × 7,6 min, juez 35 min, refinador 8 min. Envío a Kaggle pendiente de decisión. |
| 3 | 2026-10-19 | Detector hasta la meseta (antes 12 vueltas fijas) + reglas 4 y 5 en el juez + juez probado más allá del borde. Refinador igual. Detalle en [Ronda 3](#ronda-3-qué-corrige-de-la-ronda-2-y-qué-añade) | 0,389 detector solo (confianza 0,8, unir solapes; reglas 4 y 5 no se usan) · **0,4405** con refinador (umbral 0,5) | 820 / 350 / 547 | 19 / 51 | **No mejora** (0,445 en la ronda 2). Estudiar más no subió el detector: su mejor examen fue 0,370 en la vuelta 16, igual que la ronda 2 en la 12. Los exámenes saltan ±0,015 de una vez a otra, así que la ronda 2 ya estaba en su techo. La regla 4 no actuó y la regla 5 quitó más reales que inventados. Run de 316 min (detector 28 vueltas en 245 min). |
| 4 | 2026-10-20 | + Detector especialista en filamentos pequeños (Mask R-CNN sobre ventanas de 1024 de la imagen a 2048), con su propio juez. General y refinador de la ronda 3 sin reentrenar. Kaggle `filamentos-ronda-4-especialista` | 0,3887 general solo · 0,3931 + especialista · **0,4391** con refinador | 903 / 524 / 464 | 24 / 54 | **No mejora.** El especialista encuentra 88 filamentos más, pero inventa ~170. Run de 167 min; el kernel murió por memoria al final (sin submission). Ver [resumen de las 4 rondas](resumen-rondas.md). |


## Ronda 3: qué corrige de la ronda 2 y qué añade

Decidido con jhonatan el 19 oct 2026, después de desgranar la ronda 2. La cadena sigue con tres filtros: **detector → juez → refinador** (el juez actúa antes del refinador para no redibujar propuestas que se van a tirar).

**De dónde partimos (ronda 2, PQ 0,445).** El detector solo: los expertos marcaron 1.367 filamentos en el examen y el detector propuso 1.168. De esas propuestas, 775 eran reales (66 %) y 393 inventadas (34 %). De los reales encontró el 57 % y se le escaparon 592 (43 %). Peca más de ciego que de mentiroso.

| Filtro | Qué falló en la ronda 2 | Qué cambia en la ronda 3 | Por qué |
|---|---|---|---|
| Detector | Estudió 12 vueltas fijas y seguía subiendo (0,300 → 0,370). Los pasos de aprendizaje se achicaron en fechas fijas (vueltas 9 y 11). | **Estudia hasta la meseta.** Examen cada 2 vueltas. Si en 3 exámenes seguidos no mejora al menos 0,005, achica el paso ×0,1. En la siguiente meseta se para. Tope de 40 vueltas o 6,5 h. Se guarda siempre la mejor versión. | Idea de jhonatan: "le faltó entrenamiento". Hay que dejarlo hasta que la nota se aplane. |
| Juez | Ganó la confianza 0,8, que era el tope de lo probado. Solo miraba seguridad, tamaño, borde y solapes. | **Prueba la confianza hasta 0,9.** Añade la **regla 4**: si su punto más oscuro baja del 44 % del brillo del fondo, es mancha solar. Añade la **regla 5**: más de 10 veces más largo que ancho es un hilo, no un filamento. Cada regla se usa solo si sube la nota del examen. | Así piensa un anotador: oscuro a medias y alargado, no negrísimo ni un hilo. Ataca los 393 inventados. Comprobado en 64 filamentos de expertos: la regla 4 no quita ninguno y la regla 5 quita 2 (3 %). |
| Refinador | El mejor umbral fue 0,5, que era el mínimo probado. | **Prueba también 0,3 y 0,4.** El entrenamiento no cambia (10 vueltas). | jhonatan prefiere no complicar más la cadena: el refinador sigue haciendo su trabajo. |

**Resultado esperado (estimación antes de ver el resultado, 19 oct 2026).** Son estimaciones razonadas, no promesas. Sirven para comparar después y aprender si acertamos en la causa.

| Cambio | Nota de referencia en la ronda 2 | Esperado en la ronda 3 | Razonamiento |
|---|---|---|---|
| Detector hasta la meseta (examen con confianza 0,6) | 0,370 | **0,38–0,40** | La curva fue 0,300 → 0,344 → 0,345 → 0,353 → 0,364 → 0,370, subiendo ~0,01 cada 2 vueltas al final. Parte de esa subida vino de achicar el paso en las vueltas 9 y 11, así que la meseta debería llegar algo más arriba, pero cada vez sube menos. |
| Detector con el mejor juez básico (confianza hasta 0,9) | 0,391 | **0,40–0,42** | Lo que mejore el detector, más quizás +0,005 por probar una confianza más alta. |
| + Reglas 4 y 5 | — | **+0,000 a +0,010** | Atacan solo los inventados (393). La regla 4 seguramente casi no actúa, porque el detector propone pocas manchas solares. La regla 5 quita algunos hilos falsos, pero también algún filamento real. Si no ayudan, el notebook no las usa. |
| + Refinador (umbral desde 0,3) | 0,445 | **0,45–0,48** (centro ~0,46) | En la ronda 2 el refinador sumó +0,054. Esperamos algo parecido sobre un detector mejor. |
| Kaggle público (si se envía) | 0,34 (ronda 1) | **~0,41–0,44** | El examen local sale ~0,04 más alto que Kaggle. |

**Cómo leer el resultado.**
- Si el detector solo apenas sube (menos de +0,005), el problema no era cuánto estudiaba, sino qué ve. Habría que ir a la lupa de contraste.
- Si la meseta llega muy pronto (antes de la vuelta 20), la ronda 2 ya estaba casi en su techo.
- Lo que más falla, los filamentos débiles y pequeños que se escapan (592), no debería cambiar mucho en esta ronda. No es lo que ataca.

**Cómo sabremos qué ayudó.** El notebook mide cada filtro por separado. Primero da la nota del detector con el juez básico, que es comparable con el 0,391 de la ronda 2. Luego da lo que suman las reglas 4 y 5. Al final da lo que añade el refinador, que es comparable con el 0,445.

**Resultado real (20 oct 2026) frente a lo esperado.**

| Cambio | Ronda 2 | Esperado | Real | ¿Acertamos? |
|---|---|---|---|---|
| Detector hasta la meseta (confianza 0,6) | 0,370 | 0,38–0,40 | **0,370** (vuelta 16 de 28) | No. Estudiar más no ayudó. |
| Detector con el mejor juez | 0,391 | 0,40–0,42 | **0,389** (confianza 0,8 otra vez) | No. Una confianza más alta no ayuda: con 0,85 quita 70 inventados pero también 56 aciertos. |
| + Reglas 4 y 5 | — | +0,000 a +0,010 | **regla 4: 0; regla 5: −0,012** | Sí en lo pequeño. La regla 4 no encontró ninguna propuesta tan negra: el detector no propone manchas solares. La regla 5 quitó 31 inventados y 48 aciertos. No se usan. |
| + Refinador | 0,445 | 0,45–0,48 | **0,4405** | No. Igual que la ronda 2, dentro del ruido. |

**Lo que aprendimos.**
- **La ronda 2 ya estaba en el techo de este detector.** La curva (0,298 → 0,337 → 0,336 → 0,349 → 0,357 → 0,355 → 0,358 → **0,370** → 0,366 → 0,357 → 0,336 → 0,370 → 0,368 → 0,359) sube hasta la vuelta ~10 y luego solo oscila. Achicar el paso en la vuelta 22 no la subió.
- **El examen tiene ruido de ±0,015.** Diferencias menores que eso entre rondas no significan nada.
- **La seguridad del detector no separa bien lo real de lo inventado.** Al subir la confianza caen inventados y aciertos casi al mismo ritmo.
- **La mitad de los "inventados" no son inventos.** De los 350 inventados finales, 194 son "casi": el detector vio un filamento real pero lo partió, lo unió con otro o lo dibujó mal (IoU 0,1–0,5). Solo 156 son inventos puros. Los "casi" se concentran en filamentos de 1k–10k px (149), que es la regla 6 (los grandes, donde los expertos discrepan al partir y unir).
- **Inventos puros (156):** sobre todo de 330 a 3k px (133), de oscuridad media (0,8–0,95: 137) y repartidos por el disco. En la zona del 80–90 % del radio el detector acierta solo el 52 % de lo que propone (43 inventos puros).

**Descartado por ahora.** Partir el juez en dos (portero antes y juez final después del refinador), porque complica sin una ganancia clara. Dejar que el refinador estudie hasta la meseta. Para la ronda 4 quedan la "lupa de contraste" (para los débiles, la mayor bolsa de fallos), el juez que aprende de los expertos y el magnetograma (datos externos).

## Ronda 4: qué corrige de la ronda 3 y qué añade

Decidido con jhonatan el 20 oct 2026. Notebook nuevo `notebooks/entrenar_especialista.ipynb` (Kaggle `filamentos-ronda-4-especialista`). Idea de jhonatan: "un detector especialista en menos de 3000 px, así dividimos la carga", entrenado con la imagen a 2048 para "afinar muy bien el ojo" en lo que falla.

**De dónde partimos (ronda 3, PQ 0,4405).** El detector general ya tocó techo (0,37 en el examen de estudio, igual en las rondas 2 y 3). Con los filamentos grandes (más de 3.000 px) va bien: ve el 90 % (71 % acertados y 18 % "casi") y solo 10 de sus 275 propuestas de ese tamaño son inventos puros, al nivel de experto contra experto. El problema son los pequeños: **467 de los 547 filamentos que se escapan miden menos de 3.000 px** (menos de 330 px: 24 % encontrados; 330-1k: 51 %; 1k-3k: 68 %).

| Pieza | Ronda 3 | Ronda 4 | Por qué |
|---|---|---|---|
| Detector general | Entrenado hasta la meseta (vuelta 16 de 28) | **El mismo, sin volver a entrenarlo** (se cargan sus pesos) | Ya está en su techo; reentrenarlo gasta GPU sin ganar nada. |
| Detector especialista | — | **Nuevo.** Mismo Mask R-CNN, pero mira la imagen original de **2048 px** por ventanas de 1024 (cada filamento se ve el doble de grande). Arranca de lo que aprendió el general. En cada vuelta, 8 de cada 10 ventanas de estudio se centran en un filamento pequeño. Ve todos los filamentos de la ventana, también trozos de los grandes. Estudia hasta la meseta como en la ronda 3 (tope 4,5 h). | Es "la lupa": a 1024 un filamento de 500 px queda en 125 px. Si le escondiéramos los grandes, aprendería que un trozo de filamento grande es fondo y se confundiría. En el examen y el test mira 9 ventanas solapadas; cada detección cuenta solo en la ventana que tiene su centro. |
| Juez | Uno (confianza 0,8, unir solapes) | **Dos.** El del general, igual que en la ronda 3. El del especialista: solo deja pasar filamentos de menos de 3.000 px y prueba su propia confianza (0,5-0,9), unir o no, y tamaño mínimo. | Cada detector se equivoca distinto, cada uno con su juez. |
| Juntar | — | Si más del 20 % de un filamento del especialista cae sobre uno del general, se descarta (gana el general). Si es menos, se le quitan esos píxeles. | El general es fiable en lo que ya ve; el especialista solo debe **añadir** lo que falta. |
| Refinador | Entrenado 10 vueltas, umbral 0,5 | **El mismo, sin volver a entrenarlo**, sobre todo lo que pasa los jueces. Umbral 0,3-0,5. | No es lo que falla. |

Revisado antes de decidir (20 oct): el notebook público del 0,70 usa el mismo detector que nosotros y no tiene especialista en pequeños. Su autor propone mirar a 2048 por trozos como siguiente paso.

**Errores de la ronda 3 y qué esperamos que cambie en la ronda 4.** Cifras del examen final de la ronda 3 (detector + juez + refinador: TP 820, FP 350, FN 547). Cada fila dice si la ronda 4 la ataca o no.

| Error de la ronda 3 | Cuántos | ¿Lo ataca la ronda 4? | Esperado en la ronda 4 |
|---|---|---|---|
| Se escapan, menos de 330 px | 61 (solo 24 % encontrados) | Sí, el especialista | ~50 (recupera ~10). Son los más dudosos: muchos solo los marca un experto. |
| Se escapan, 330-1k px | 240 (51 % encontrados) | Sí, el especialista | ~160 (recupera ~80). Es la bolsa más grande y donde la lupa más debería ayudar. |
| Se escapan, 1k-3k px | 166 (68 % encontrados) | Sí, el especialista | ~105 (recupera ~60) |
| Se escapan, más de 3k px | 80 (71 % encontrados) | No | ~80, sin cambio |
| Inventados "casi" (vio uno real pero lo partió, unió o dibujó mal) | 194 | No | ~194, sin cambio. Es el trabajo de la ronda siguiente. |
| Inventados puros | 156 | Puede empeorar | ~215. El especialista también inventará algo, sobre todo de 330 a 3k px; su juez debe frenarlo. |
| **Total** | **TP 820 · FP 350 · FN 547** | | **TP ~970 · FP ~410 · FN ~395** |

Si el especialista no cumple esa tabla (recupera pocos o inventa tantos como recupera), el notebook no lo usa y la nota se queda en ~0,44.

**Resultado esperado (estimación antes de ver el resultado, 20 oct 2026).**

| Etapa | Ronda 3 | Esperado en la ronda 4 | Razonamiento |
|---|---|---|---|
| General solo con su juez | 0,389 (TP 771, FP 399, FN 596) | **0,389** | Es el mismo detector: sirve de control. Si sale distinto, algo está mal cargado. |
| + Especialista | — | **0,41-0,43** | Si encuentra un tercio de los pequeños que se escapan (~150 aciertos más) con ~60 inventados nuevos: TP ~920, FP ~460, FN ~450, RQ 0,61 → 0,67. La forma (SQ) puede bajar un poco porque los pequeños se dibujan peor. |
| + Refinador | 0,4405 | **0,46-0,49** (centro ~0,47) | En las rondas 2 y 3 el refinador sumó unos +0,05. |

**Cómo leer el resultado.**
- Si el especialista suma aciertos pero casi el mismo número de inventados, el juez del especialista lo dejará fuera (el notebook solo lo usa si sube la nota). Entonces los pequeños que faltan no se distinguen bien ni con lupa, y habría que ir a la lupa de contraste o al juez que aprende.
- Si sube aciertos claramente, la siguiente palanca serían los "casi" grandes (partir y unir).
- Ojo con el ruido: el examen salta ±0,015.

**Cómo sabremos qué ayudó.** El notebook imprime la nota del general solo, la del general + especialista con cuántos aciertos e inventados añade el especialista, y la de cada combinación con el refinador.

**Dónde están los puntos que faltan (para que 0,46-0,49 no sea el techo).** Cálculo con las cifras de la ronda 3, suponiendo que cada arreglo sale perfecto. Sirve para ordenar las rondas, no es una promesa.

| Arreglo | Ronda que lo ataca | PQ si saliera perfecto |
|---|---|---|
| Ronda 3 tal cual | — | 0,44 |
| Arreglar los "casi" (partir y unir bien los grandes: −194 FP, −226 FN) | Siguiente | ~0,55 |
| + Encontrar 150 pequeños más | Ronda 4 (especialista) | ~0,58 |
| + Mejor borde (forma SQ de 0,68 a 0,75) | Refinador mejor | ~0,64 |

La ronda 4 ataca la bolsa de pequeños. La bolsa más grande que queda son los "casi" de los filamentos grandes (regla 6 de jhonatan).

**Corrección tras mirar los "casi" uno por uno (20 oct 2026, `analisis/casi_ronda3.py` sobre `predicciones_examen.json` de la ronda 3).** La tabla anterior es demasiado optimista: muchos "casi" no son errores claros, sino casos donde los propios expertos discrepan. Cada filamento real se compara con cada experto (1.367 casos) y, como referencia, cada experto con otro experto en las imágenes con 2-3 expertos (1.190 casos):

| Qué le pasa al filamento real | Ronda 3 | Experto contra experto |
|---|---|---|
| Acierto (IoU > 0,5) | 809 (59 %) | 564 (47 %) |
| Escapado del todo | 313 (23 %) | 246 (21 %) |
| Fusionado con otro en una sola predicción | 96 (7,0 %) | 80 (6,7 %) |
| Mal dibujado (una sola predicción, IoU 0,1-0,5) | 132 (9,7 %) | 264 (22 %) |
| Partido en varios trozos | 17 (1,2 %) | 36 (3,0 %) |

- **Une más de lo que parte** (96 reales fusionados frente a 17 partidos), pero fusiona al mismo ritmo que un experto frente a otro, y parte y dibuja mal menos que ellos.
- **42 predicciones unen varios reales** (2 de media, sobre todo de 3k-10k px). En 13 de las 26 que caen en imágenes con varios expertos, otro experto dibujó exactamente eso como un solo filamento: es la regla 6 (los expertos discrepan al partir y unir).
- **Mal dibujados:** la mitad son más grandes que el real (68) y la otra mitad más pequeños (64). No hay un sesgo fijo que corregir.
- Las cifras salen ligeramente distintas del notebook (809 aciertos frente a 820) porque las predicciones se guardaron como polígonos simplificados.

Conclusión: en partir, unir y dibujar ya estamos al nivel de un experto frente a otro. Arreglar los "casi" no llevaría a ~0,55; esa estimación se descarta.

**Resultado real (20 oct 2026, versión 2 del notebook, 167 min de GPU) frente a lo esperado.** El kernel se cayó (falta de memoria, casi seguro) al probar el refinador con umbral 0,5, después de dar todas las cifras de abajo. Por eso no hay `submission.csv` ni JSON de predicciones; sí se guardó `especialista_mejor.pt`.

| Etapa | Ronda 3 | Esperado | Real | ¿Acertamos? |
|---|---|---|---|---|
| General solo con su juez | 0,389 | 0,389 | **0,3887** (TP 771, FP 399, FN 596) | Sí: control correcto, los pesos se cargaron bien. |
| + Especialista | — | 0,41-0,43 | **0,3931** (TP 859, FP 569, FN 508; su juez: confianza 0,7, unir, mínimo 330 px) | No. +0,004, dentro del ruido. |
| + Refinador (umbral 0,4) | 0,4405 | 0,46-0,49 | **0,4391** (general solo + refinador: 0,4390) | No. El especialista no suma nada. |

| Error de la ronda 3 | Esperado en la ronda 4 | Real (antes del refinador) |
|---|---|---|
| Se escapan (FN) | 596 → ~450 | 596 → **508** (recupera 88) |
| Inventados (FP) | 399 → ~460 | 399 → **569** (+170) |

**Lo que aprendimos.**
- **La lupa sí ve más, pero ve de todo.** El especialista encontró 88 filamentos pequeños que el general no veía, pero por cada uno inventó unos dos. En PQ, un acierto con dos inventados al lado deja la nota igual.
- **Estudiar más no lo afinó.** Su mejor examen fue en la vuelta 2 (0,382) y en la 14 seguía igual (0,385), aunque la pérdida bajaba (0,52 → 0,47). Aprendió a dibujar mejor lo que ya veía, no a distinguir lo real de lo que no.
- **Confirma el hallazgo de la duda humana.** En los filamentos pequeños los expertos coinciden poco (menos de 330 px: 41 %). Muchos "inventados" del especialista son manchas que algún experto podría haber marcado y otro no. El límite no es la resolución (el ojo), sino a qué estilo de experto apuntamos.
- **El examen de estudio ya lo avisaba:** general + especialista con confianza 0,6 nunca superó al general solo (0,389).

**Envío a Kaggle (20 oct 2026).** Se envió el `submission.csv` de la ronda 3 (no el de la ronda 4, que no mejora): **público 0,38** frente a 0,34 de la ronda 1. En local la subida fue de 0,384 a 0,4405; Kaggle da unas 0,04-0,06 menos que nuestro examen, pero sube en la misma dirección.

**Decisión pendiente con jhonatan.** No relanzar el especialista tal cual. Siguiente palanca candidata: elegir el estilo de experto (p. ej. entrenar con un solo experto por imagen) o un juez que aprenda qué pequeños marcan los expertos.

## Ronda 5 (plan): detector YOLO11-seg

Decidido con jhonatan el 21 oct 2026. Tras el tribunal (los jueces no suben la nota) y el análisis de estilo de experto (sin palanca), la pregunta es si un detector más moderno que Mask R-CNN ve mejor los pequeños que se escapan. Candidato elegido: **YOLO11-seg** (Ultralytics, 2024).

**De dónde partimos (ronda 3, examen):** PQ **0,4405** = TP 820 · FP 350 · FN 547, borde (SQ) 0,682.
- De los 547 escapados, 313 no los toca ninguna predicción y 234 están vistos a medias.
- En modo generoso, el detector actual propone algo encima de 319 de los 547 (paso 1 del tribunal).

| Pieza | Ronda 3 | Ronda 5 | Por qué |
|---|---|---|---|
| Detector | Mask R-CNN v2 (torchvision), 1024 px | **YOLO11-seg** (tamaño m), 1024 px, preentrenado en COCO y reentrenado con las 586 imágenes de estudio. Cada experto como ejemplo aparte, como siempre. Estudia hasta que el examen de estudio deje de mejorar (tope ~3 h). | Arquitectura de 2024 y "ojo" distinto. Es la familia más usada hoy en Kaggle para segmentar objetos. |
| Juez | Confianza ≥ 0,8 y unir solapes | El mismo tipo de juez. Su umbral se busca **sin GPU** con el simulador, a partir de los candidatos guardados (confianza ≥ 0,1). | La confianza de YOLO no está en la misma escala que la de Mask R-CNN. |
| Refinador | U-Net ResNet-18 de la ronda 3 | **El mismo, sin reentrenar.** | No es lo que falla. YOLO dibuja las máscaras algo más bastas, y el refinador las redibuja. |
| Votación | — | **Sin GPU:** juntar los candidatos de YOLO con los de la ronda 3 en el simulador (un filamento entra si lo apoyan los dos, o uno con confianza alta). | Si YOLO empata, sirve igual como segundo modelo que vota. |

**Qué guarda el notebook:**
- nota del examen de YOLO solo y de YOLO + refinador, con el desglose TP/FP/FN y los escapados por tamaño;
- `candidatos_yolo_examen.json` y `candidatos_yolo_estudio.json`, con el mismo formato que el paso 1;
- los pesos `yolo_mejor.pt`.

No genera envío hasta que veamos los números.

**Errores de la ronda 3 y qué esperamos en la ronda 5** (estimación antes de lanzar):

| Error de la ronda 3 | Cuántos | Esperado con YOLO + refinador | Razonamiento |
|---|---|---|---|
| Escapados < 1.000 px | ~300 | ~270 | Una arquitectura con mejor detección de objetos pequeños puede rescatar algunos, pero gran parte es duda humana. |
| Escapados 1k-3k px | 166 | ~150 | Igual. |
| Escapados > 3k px | 80 | ~80 | Ya va bien. |
| Inventados | 350 | ~350-380 | Un detector nuevo con pocas imágenes suele inventar algo más al principio. |
| **Total** | **TP 820 · FP 350 · FN 547** | **TP ~860 · FP ~365 · FN ~505** | |

**Resultado esperado:**

| Etapa | Ronda 3 | Esperado en la ronda 5 |
|---|---|---|
| Detector solo con su juez | 0,389 | **0,37-0,41**: máscaras más bastas, pero quizá más aciertos |
| + refinador | 0,4405 | **0,43-0,46** (centro 0,445) |
| Votación YOLO + ronda 3 (sin GPU) | — | **0,44-0,47** (centro 0,455) |

**Cómo leer el resultado:**
- Si YOLO + refinador da ≥ 0,455, el ojo moderno ayuda y merecería la pena probar el siguiente (Mask2Former o un ojo moderno en Mask R-CNN).
- Si empata pero la votación sube, nos quedamos con dos modelos que votan.
- Si nada sube, la fuga de los pequeños es duda humana y no un problema de "ojo". Entonces solo quedaría pulir: reentrenar con las 707 imágenes al final.
- Ruido del examen: ±0,015.

**Requisitos en Kaggle:** GPU T4, **Internet On** (instala `ultralytics` y descarga los pesos de partida), y como Inputs la competición más `filamentos-detector-ronda-2` Version 4 (refinador). Licencia de Ultralytics: AGPL-3.0. Encaja con la regla de repo público al final.

### Ronda 5: resultado real (21 oct 2026, Kaggle filamentos-ronda-5-yolo, 161 min de GPU)

YOLO11m-seg entrenó 88 vueltas y se paró por el tope de 2,5 h, no por meseta. El mAP del examen estaba plano desde la vuelta ~60 (0,67). Generó 1.708 candidatos en el examen (confianza ≥ 0,1).

| Etapa | Ronda 3 | Esperado | Ronda 5 real (mejor umbral) |
|---|---|---|---|
| Detector solo | 0,389 | 0,37-0,41 | **0,352** (umbral 0,3; TP 696 · FP 443 · FN 671) |
| + refinador | 0,4405 | 0,43-0,46 | **0,4532** (umbral 0,2; TP 922 · FP 481 · FN 445) |

| Error | Ronda 3 | Esperado | Ronda 5 (umbral 0,2) |
|---|---|---|---|
| Aciertos | 820 | ~860 | **922** (+102) |
| Inventados | 350 | ~365 | **481** (+131) |
| Escapados | 547 | ~505 | **445** (−102) |
| Escapados < 1.000 px | ~300 | ~270 | 236 |
| Escapados 1k-3k px | 166 | ~150 | 142 |
| Escapados > 3k px | 80 | ~80 | 67 |

Cuánto cuesta cada grupo (puntos de PQ que se ganarían si desapareciera): inventados 0,095 (ronda 3: 0,07); escapados 0,087 (ronda 3: 0,16).

- Dentro de lo esperado (+0,013 sobre la ronda 3, justo en el borde del ruido ±0,015).
- Cambia el reparto: YOLO ve 102 filamentos más (sobre todo pequeños) pero inventa 131 más. Es un detector más "generoso".
- YOLO solo dibuja peor (SQ 0,63 vs 0,68 con refinador): el refinador es imprescindible.
- Su confianza está bien calibrada: la franja 0,2-0,3 acierta el 34 % (justo el punto de equilibrio ~1/3); la 0,1-0,2 el 20 % (no compensa).
- Al alcance de algún candidato: 1.063 de 1.367 (78 %), menos que Mask R-CNN (1.139), pero acierta más de los que alcanza.

**Votación YOLO + ronda 3 (sin GPU, `analisis/votacion_r5.py`).** Se emparejan los candidatos de los dos modelos (IoU > 0,5); se prueban ~180 recetas. "Honesto" = receta elegida en una mitad del examen y medida en la otra (media de los dos cruces).

| Receta | Mejor en el examen | TP / FP / FN | Honesto |
|---|---|---|---|
| Solo ronda 3 (conf ≥ 0,8) | 0,4409 | 821 / 352 / 546 | 0,438 |
| Solo YOLO (conf ≥ 0,25) | 0,4560 | 878 / 381 / 489 | 0,452 |
| Unión (YOLO ≥ 0,25 o ronda 3 ≥ 0,95) | 0,4592 | 885 / 377 / 482 | 0,451 |
| Acuerdo (los dos lo ven) | 0,4525 | 889 / 426 / 478 | 0,449 |
| Media ponderada (0,2·ronda 3 + 0,8·YOLO ≥ 0,35) | **0,4604** | 884 / 369 / 483 | **0,457** |

- La votación apenas suma: +0,005 honesto sobre YOLO solo, dentro del ruido.
- La silueta de YOLO refinada gana siempre a la de Mask R-CNN refinada en las parejas.
- Lo que la ronda 3 ve y YOLO no, casi nunca es real: Mask R-CNN no aporta filamentos nuevos.
- Conclusión: el salto viene del detector nuevo, no de la votación. YOLO solo con umbral 0,2-0,25 es el candidato a envío.

## Curva de aprendizaje (plan, 21 oct 2026): ¿más imágenes suben la nota?

Decidido con jhonatan tras la ronda 5. Él cree que faltan imágenes; antes de buscar más (fotos vecinas de GONG, otros telescopios) medimos si la cantidad es la palanca.

**Diseño:** mismo sistema que la ronda 5 (YOLO11m-seg a 1024 + refinador de la ronda 3, juez de umbral + unir) y mismo examen de 121 imágenes. Solo cambia la parte de las 586 imágenes de estudio que ve YOLO: 25 %, 50 % y 75 %, anidadas (el 25 % está dentro del 50 %…), orden al azar con semilla 0. Tope 1,75 h por paso, paciencia 25. El 100 % es la ronda 5 (0,4532). Notebook: `notebooks/curva_aprendizaje.ipynb` (generado por `generar_notebook_curva.py`), ~6 h de GPU.

**Esperado (antes de lanzar):** en detección de objetos con pocos datos la nota suele crecer con el logaritmo de las imágenes. Estimación: 25 % ≈ 0,40-0,42; 50 % ≈ 0,43-0,44; 75 % ≈ 0,445-0,45.

**Regla de decisión (fijada antes de lanzar):**
- 50 % → 100 % sube **≥ 0,02**: más imágenes pagan; doblar debería dar otro escalón parecido → fase 2 (fotos vecinas de GONG con las etiquetas copiadas, permitido por los organizadores solo para entrenar).
- Sube **< 0,01**: dentro del ruido; la cantidad no es la palanca.
- Entre medias: decidir juntos mirando escapados por tamaño.

## Análisis sin GPU

- **La tabla pública de Kaggle está rota (20 oct 2026).** El notebook público "Please fix the Leaderboard" demuestra que enviar máscaras vacías puntúa entre 0,70 y 1,00. Los organizadores dicen en el foro que se toman "muy en serio" un 0,41 y que revisarán a mano todas las notas altas. Los participantes honestos del foro van por 0,40-0,46. Conclusión: la meta de "alcanzar al líder (0,64)" no vale; la brújula es nuestro examen local.
- **¿Hace falta ver a 2048? (18 oct 2026).** Se encogen los dibujos de los expertos a menos píxeles y se reamplían a 2048 para medir la nota máxima posible por falta de nitidez (25 imágenes de muestra, 2.072 filamentos). Techo de PQ: 0,91 a 512 px, 0,98 a 1024 px y 1,00 a 1536 px. A 1024 no se pierde ningún filamento. Casi todos miden 9 px o más de grosor (mediana 16 px). Conclusión: la resolución no limita el **dibujo**. Queda por saber si limita **ver** los filamentos débiles; eso lo dirá el informe de errores.
- **Notebooks públicos (19 oct 2026).** El bloque de 0,55 de la tabla sale de un notebook que reenvía un CSV ya calculado, sin modelo. El mejor método público honesto (LB 0,70, de Fususu, basado en el notebook de Nomannic19) es un detector Mask R-CNN a 1024 px más un refinador U-Net sobre recortes de cada filamento a 2048. Encaja con nuestro diagnóstico: lo que falla es decidir qué es filamento y partir o unir, no el borde. Decisión de jhonatan: pasar ya al detector (ronda 2 = `notebooks/entrenar_detector.ipynb`).
- **Dibujar las máscaras a baja resolución con PIL las engorda (19 oct 2026).** Con máscaras de experto perfectas (15 imágenes), dibujarlas a 1024 con PIL y reampliarlas da PQ 0,82. Dibujarlas a 2048 con pycocotools, como la métrica oficial, y reducirlas da 0,89. La ronda 1 usó PIL para el objetivo, así que la U-Net aprendió bordes algo gruesos. El detector ya usa la forma buena.
