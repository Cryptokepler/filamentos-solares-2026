# Datos: red GONG y dataset MAGFiLO

## GONG (Global Oscillation Network Group)
- Red de 6 telescopios repartidos por el mundo para observar el Sol las 24 horas [Tlatov y Pevtsov]:
  Big Bear (California), Mauna Loa (Hawái), Learmonth (Australia), Udaipur (India), Teide (Canarias) y Cerro Tololo (Chile).
- Cada sede toma una imagen H-alfa de disco completo **cada 60 s**; desfasadas 20 s, la red da una imagen cada 20 s [Tlatov y Pevtsov].
- La letra antes de la `h` final del nombre del archivo indica la sede: B, C, L, M, T, U (**inferido** del nombre de las sedes; coincide con las 6 letras que hay en los datos).
- Al ser telescopios en tierra, las imágenes tienen nubes, turbulencia y diferencias de calibración entre sedes: es uno de los retos que nombra la competición.

## MAGFiLO v1.0 [MAGFiLO]
- *Manually Annotated GONG Filaments in H-alpha Observations*, publicado en *Scientific Data* (2024).
- 10 244 filamentos en 1 593 observaciones de GONG entre 2011 y 2022.
- Por cada filamento: contorno (segmentación), caja mínima, espina y quiralidad.
- Más de 1 000 horas de trabajo con revisión doble ciego; acuerdo entre anotadores **Kappa = 0,66**.

## La versión de la competición **[medido]**
- 707 imágenes de entrenamiento (JPEG 2048 × 2048) y 180 de test sin respuestas.
- 1 154 registros de anotación: 411 imágenes con 1 anotador, 145 con 2 y 151 con 3; 8 199 filamentos.
- Categorías: Left 2 535, Right 2 590, Unidentifiable 3 074.
- Área de un filamento: mediana 1 228 px; el 95 % supera ~330 px.
- Años: sobre todo 2011–2017. Las 180 imágenes de test son de 176 días distintos: no hay secuencias temporales.
