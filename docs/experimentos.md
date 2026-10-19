# Diario de experimentos

Un cambio por experimento. Nota = PQ en las 121 imágenes de examen (validación fija). Ver [roadmap](roadmap-entrenamiento.md).

| # | Fecha | Qué se cambió | PQ examen | TP / FP / FN | Partidos / fusionados | Qué se vio |
|---|---|---|---|---|---|---|
| 0 | 2026-10-18 | Línea base de reglas (sin red) | 0,098 | | | Muchas manchas oscuras que no son filamentos |
| 1 | 2026-10-18 | Primer entrenamiento: U-Net resnet34, 1024 px, 40 épocas | 0,363 red sola (época 40) · **0,384** con juez (umbral 0,6, área ≥ 330, sin unir) | 826 / 648 / 541 | 93 / 27 | Supera el 0,34 experto-vs-experto. Seguía mejorando en la época 40 (sin techo). Lo flojo es detectar (RQ 0,58) más que dibujar (SQ 0,66): inventa casi tantos como acierta. Parte ~3 veces más que fusiona. Ojo: época y juez elegidos mirando el mismo examen, la nota es algo optimista. **Kaggle público: 0,34** (envío 18 oct 22:57; líderes 0,55–0,64). Run de 4 h: 1 h 35 min preparando imágenes y 1 h prediciendo el test (filtro de mediana lento), solo 50 min entrenando. |
| 2 | 2026-10-19 | Cambio de método: detector Mask R-CNN (1024 px, un ejemplo por experto, 12 épocas) + refinador U-Net resnet18 en recortes de 2048 (4 vistas). Máscaras de entrenamiento dibujadas a 2048 (antes PIL las engordaba). Notebook `entrenar_detector.ipynb`, Kaggle `filamentos-detector-ronda-2` | 0,391 detector solo (confianza ≥ 0,8, unir solapes) · **0,445** con refinador (umbral 0,5) | 827 / 341 / 540 | 18 / 43 | +0,061 sobre la ronda 1. El refinador sube forma (SQ 0,64 → 0,68) y detección (RQ 0,61 → 0,65). Inventa la mitad que la ronda 1 (341 vs 648). Ahora fusiona más de lo que parte (al revés que la ronda 1). Seguía mejorando en la época 12, y la confianza y el umbral del refinador ganadores están en el borde de lo probado (0,8 y 0,5): hay que ampliar la búsqueda. Lo que se le escapa: pequeños (< 330 px: 25 % acierto; 330-1k: 51 %) y débiles (casi fondo > 0,95: 28 %). Por telescopio no hay diferencias grandes (57-69 %). Run de 160 min: detector 12 × 7,6 min, juez 35 min, refinador 8 min. Envío a Kaggle pendiente de decisión. |
| 3 | 2026-10-19 | Detector hasta la meseta (antes 12 vueltas fijas) + reglas 4 y 5 en el juez + juez probado más allá del borde. Refinador igual. Detalle en [Ronda 3](#ronda-3-qué-corrige-de-la-ronda-2-y-qué-añade) | pendiente | | | Preparado; se lanza cuando jhonatan haga Save & Run All. |


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

**Descartado por ahora.** Partir el juez en dos (portero antes y juez final después del refinador), porque complica sin una ganancia clara. Dejar que el refinador estudie hasta la meseta. Para la ronda 4 quedan la "lupa de contraste" (para los débiles, la mayor bolsa de fallos), el juez que aprende de los expertos y el magnetograma (datos externos).

## Análisis sin GPU

- **¿Hace falta ver a 2048? (18 oct 2026).** Se encogen los dibujos de los expertos a menos píxeles y se reamplían a 2048 para medir la nota máxima posible por falta de nitidez (25 imágenes de muestra, 2.072 filamentos). Techo de PQ: 0,91 a 512 px, 0,98 a 1024 px y 1,00 a 1536 px. A 1024 no se pierde ningún filamento. Casi todos miden 9 px o más de grosor (mediana 16 px). Conclusión: la resolución no limita el **dibujo**. Queda por saber si limita **ver** los filamentos débiles; eso lo dirá el informe de errores.
- **Notebooks públicos (19 oct 2026).** El bloque de 0,55 de la tabla sale de un notebook que reenvía un CSV ya calculado, sin modelo. El mejor método público honesto (LB 0,70, de Fususu, basado en el notebook de Nomannic19) es un detector Mask R-CNN a 1024 px más un refinador U-Net sobre recortes de cada filamento a 2048. Encaja con nuestro diagnóstico: lo que falla es decidir qué es filamento y partir o unir, no el borde. Decisión de jhonatan: pasar ya al detector (ronda 2 = `notebooks/entrenar_detector.ipynb`).
- **Dibujar las máscaras a baja resolución con PIL las engorda (19 oct 2026).** Con máscaras de experto perfectas (15 imágenes), dibujarlas a 1024 con PIL y reampliarlas da PQ 0,82. Dibujarlas a 2048 con pycocotools, como la métrica oficial, y reducirlas da 0,89. La ronda 1 usó PIL para el objetivo, así que la U-Net aprendió bordes algo gruesos. El detector ya usa la forma buena.
