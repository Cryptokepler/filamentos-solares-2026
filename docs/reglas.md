# Reglas observadas

Cada regla con la imagen donde se vio y, si existe, el número que la respalda. Luego se convierte en código y se mide con la PQ local.

Datos medidos con 100 imágenes de entrenamiento (1 de cada 7) y sus 1060 filamentos dibujados. Script: `/mnt/project-files/analisis/manchas.py`, resultados en `/mnt/project-files/analisis/manchas.csv`.

| # | Regla / premisa | De dónde sale | Dato | Estado |
|---|---|---|---|---|
| 1 | No toda mancha oscura es un filamento. | jhonatan, imagen 20110109104734Ch | Solo ~19 % de las manchas oscuras de más de 60 px están dibujadas. | Confirmada |
| 2 | Cerca del borde del Sol no se dibujan filamentos. | Medición | Solo 0,8 % de los filamentos dibujados está más allá del 90 % del radio, y ninguno más allá del 95 %. Si estuvieran repartidos al azar, habría ~19 %. | Confirmada, por convertir en código |
| 3 | Hay un tamaño mínimo. | Medición | El 95 % de los filamentos dibujados mide más de ~330 px (en imagen de 2048). De las manchas oscuras de 60–200 px solo se dibuja el 13 %. | Por fijar el umbral exacto |
| 4 | Lo más negro no es filamento (probablemente manchas solares). | Medición | De las manchas cuyo punto más oscuro baja del 44 % del brillo del fondo, se dibujan el 0 %. Los filamentos son oscuros "a medias". | Por confirmar en el visor |
| 6 | Los filamentos pequeños son ruido entre expertos; los grandes se pierden por la forma o el corte, no por no verlos. | Medición en las 296 imágenes con 2–3 anotadores (4993 filamentos) | Menos de 330 px: otro experto lo dibuja igual solo el 41 % y nadie más lo dibuja el 38 %. Más de 4000 px: otro experto lo toca el 96 %, pero solo coincide con IoU > 0,5 el 70 %. | Confirmada |
| 5 | Forma: alargada pero no un hilo larguísimo. | Medición | Se dibuja ~50 % de las manchas con proporción largo/ancho entre 2,4 y 6. Por encima de 10, solo el 4 %. | Por confirmar en el visor |
