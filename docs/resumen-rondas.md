# Resumen de las 4 rondas: qué funcionó y qué no

Escrito el 20 oct 2026, después de la ronda 4. Junta en un sitio los datos del [diario](experimentos.md) para pensar en las siguientes soluciones. Todas las notas son PQ en el mismo examen fijo de 121 imágenes de train (nunca del test). El examen tiene un ruido de ±0,015, así que una diferencia menor que eso entre rondas no significa nada.

## 1. Las rondas de un vistazo

| Ronda | Qué probamos | PQ examen final | TP / FP / FN | Partidos / fusionados | Kaggle público | GPU |
|---|---|---|---|---|---|---|
| 0 | Reglas sin red (zonas oscuras) | 0,098 | — | — | — | 0 |
| 1 | U-Net (segmentación) a 1024 + juez | 0,384 | 826 / 648 / 541 | 93 / 27 | **0,34** | 4 h |
| 2 | Mask R-CNN (detector) a 1024 + juez + refinador a 2048 | **0,445** | 827 / 341 / 540 | 18 / 43 | no enviado | 2 h 40 |
| 3 | Lo mismo, con el detector entrenado hasta la meseta, reglas 4 y 5 y más umbrales | 0,4405 | 820 / 350 / 547 | 19 / 51 | **0,38** | 5 h 16 |
| 4 | + detector especialista en pequeños a 2048 por ventanas | 0,4391 | 903 / 524 / 464 | 24 / 54 | no se envía | 2 h 47 |

Referencias fijas: experto contra experto 0,34; techo por resolución a 1024 px 0,98.

**En una frase:** el único salto de verdad fue pasar de U-Net a detector + refinador (ronda 2, +0,06). Las rondas 3 y 4 no movieron la nota.

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

## 6. Lo que aún no hemos probado (ideas para discutir, sin decidir)

| Idea | Qué ataca | Dato que la apoya |
|---|---|---|
| Entrenar apuntando a un estilo de experto (p. ej. un solo experto por imagen, el que más dibujó, como el notebook público del 0,70) | El criterio en los dudosos | La nota cambia de 0,41 a 0,62 según el experto |
| Juez que aprende (un modelo pequeño que decide qué propuestas aprobar, entrenado con los aciertos e inventados del examen de estudio) | Separar lo real de lo inventado mejor que la confianza | Las reglas a mano no funcionaron; la confianza sola tampoco |
| Varios detectores que votan (el "revisor" de la cascada original, nunca construido) | Inventos: un invento rara vez lo ven todos | Idea de jhonatan del 18 oct, pendiente |
| Lupa de contraste (realzar los filamentos débiles antes de mirar) | Los casi invisibles (33 % encontrados) | Es la bolsa con peor acierto |
| Reentrenar al final con las 707 imágenes (también las del examen) | Más datos: +20 % | Lo hace el notebook público del 0,70 |
| Datos externos (otros telescopios GONG a la misma hora, magnetograma) | Débiles y dudosos | Aparcado por decisión de jhonatan hasta que lo oficial se estanque, y ya se ha estancado |
| Usar las 2 GPU T4 | Velocidad (rondas más baratas) | Ahora se usa 1 de 2 |
