# Roadmap de entrenamiento

Del 18 de octubre al 15 de noviembre de 2026. Cada paso tiene una meta medible y algo que mirar con los ojos antes de pasar al siguiente.

## Cómo repartimos las 707 imágenes

Entrenar **no gasta** las imágenes: la red las mira muchas veces y siguen ahí. Lo que sí se "gasta" es la **confianza en el examen**. Si la red estudia con una imagen, esa imagen ya no sirve para examinarla. Y si ajustamos demasiadas cosas mirando siempre el mismo examen, acabamos aprendiéndonos el examen.

| Grupo | Imágenes | Para qué | Quién lo ve |
|---|---|---|---|
| **Estudio** | 586 | La red aprende con ellas | La red |
| **Examen** | 121 (49 con 2–3 expertos) | Nota local (PQ) después de cada experimento | Nadie durante el entrenamiento |
| **Test de Kaggle** | 180 | La nota oficial | Solo Kaggle tiene las respuestas |

- El examen es siempre el mismo grupo de 121 imágenes (elegido por el nombre del archivo, así que se repite igual cada vez). Así las notas de distintos experimentos se pueden comparar.
- No hay que subir imágenes a ningún sitio: los datos ya están en Kaggle y el notebook los lee desde allí.
- Al final, cuando ya no queden decisiones por tomar, la red aprenderá con **las 707** para dar la respuesta definitiva.

## Reglas del juego

1. **Un cambio por experimento.** Si cambiamos dos cosas a la vez, no sabemos cuál ayudó.
2. **Todo se apunta** en `docs/experimentos.md`: qué se cambió, la nota y qué se vio. Esto también sirve para el informe final.
3. **No se envía nada a Kaggle** hasta que la nota local supere 0,34 (el acuerdo entre expertos). Si nos quedamos cerca, lo hablamos.
4. Kaggle da unas 30 horas de GPU gratis a la semana. Cada experimento tarda 1–2 h, así que caben unos 10–15 por semana.

## Paso 1: primer entrenamiento (esta semana)

- **Qué:** ejecutar `notebooks/entrenar_observador.ipynb` tal cual.
- **Meta:** superar el 0,10 de la línea base de reglas.
- **Qué mirar:** la nota y el desglose. Se cuentan los aciertos, los falsos (inventados), los que se le escapan, los partidos (1 real → varios trozos) y los fusionados (varios reales → 1 mancha).
- **Ya preparado (a partir del 2.º entrenamiento):** el notebook guarda `predicciones_examen.json` para ver en el visor lo que predice la red junto a los dibujos de los expertos.

## Paso 2: diagnóstico (jhonatan + visor)

- **Qué:** mirar en el visor (opción *Examen de la red*) las 121 imágenes del examen con la predicción encima, ordenadas de peor a mejor nota.
- **Pregunta clave:** ¿dónde pierde más puntos? ¿Inventa, se le escapan, parte o fusiona?
- **Resultado:** una lista ordenada de los 2–3 problemas más gordos. Esa lista decide el paso 3.

## Paso 3: mejoras de una en una (semanas 2–3)

Candidatas. El orden lo decide el diagnóstico:

| Problema | Posible mejora |
|---|---|
| Se le escapan filamentos finos | Más resolución (1024 → 1536 o 2048) |
| Inventa manchas | Juez más estricto: tamaño, oscuridad, forma (reglas 3–5) |
| Parte filamentos largos | Unir trozos cercanos que siguen la misma dirección |
| Fusiona filamentos vecinos | Enseñarle los bordes entre filamentos |
| Aprende poco en general | Más épocas o una red base más grande |
| Fallos cerca del borde del Sol | Regla 2 (nada más allá del 90–95 % del radio) |

Después de cada mejora: nota nueva, comparar con la anterior y quedarse con lo que suba.

## Paso 4: el revisor (semana 3)

- **Qué:** entrenar 5 redes, cada una con un examen distinto (validación cruzada). Así cada imagen hace de examen una vez y todas sirven para estudiar.
- **Cómo vota:** se suman las 5 opiniones y solo queda lo que la mayoría ve.
- **Meta:** que la votación suba la nota respecto a una sola red.

## Idea de jhonatan: consenso entre telescopios

GONG tiene 6 telescopios (la letra al final del nombre de la imagen: B, M, L, U, T, C). En el test cada imagen es de un solo telescopio, pero las reglas permiten datos externos y GONG publica en abierto las imágenes de todos.

- **Paso previo (gratis):** medir con las predicciones del examen si la red falla más en algún observatorio.
- **La idea:** para cada imagen, descargar las de los otros telescopios a la misma hora, alinearlas, pasar la red por todas y quedarse con lo que ven varios. Un filamento visto por 3–4 telescopios es real; una mancha en uno solo suele ser una nube o un defecto de cámara.
- **Cuándo:** después del paso 4, si la red ya está en buen nivel. Es el revisor con telescopios en vez de modelos.

## Paso 5: juez final y primer envío (semana 4)

- Ajustar las reglas del juez sobre la votación del revisor.
- Si la nota local supera 0,34: primer envío a Kaggle y comparar la nota oficial con la local. Si se parecen, nuestro examen es fiable.
- Entrenar con las 707 imágenes para el envío definitivo.

## Paso 6: entrega (antes del 15 de noviembre)

- Informe técnico de 4 páginas (plantilla Overleaf).
- Repo público con `requirements.txt` y el notebook completo.
- Formulario de Google de la competición.
