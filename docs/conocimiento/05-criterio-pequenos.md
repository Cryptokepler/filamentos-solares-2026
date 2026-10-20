# Criterio para los filamentos pequeños (20 oct 2026)

Pregunta de jhonatan tras la ronda 4: la vista ya está bien, nos falta **criterio** para decidir qué filamento pequeño es de verdad. Dos fuentes: nuestros propios datos (dónde coinciden los expertos) y lo publicado (cómo se anotó MAGFiLO y qué hicieron otros).

## 1. Lo que dicen nuestros datos: dónde coinciden los expertos

Script: `analisis/consenso_pequenos.py`. Usa las 296 imágenes de train con 2-3 expertos (4.993 filamentos). Para cada filamento de un experto se mira si los demás dibujaron el mismo (IoU > 0,5):
- **todos:** lo dibujaron todos los demás expertos;
- **alguno:** solo parte de ellos;
- **solo:** ningún otro.

La oscuridad y la distancia al centro solo se pueden medir en las imágenes que tenemos en local: 34 imágenes, 471 filamentos pequeños.

**Por tamaño:**

| Tamaño | todos | alguno | solo | n |
|---|---|---|---|---|
| < 330 px | 24 % | 15 % | **61 %** | 263 |
| 330-1k | 38 % | 17 % | 44 % | 1.648 |
| 1k-3k | 50 % | 19 % | 32 % | 2.031 |
| 3k-10k | 54 % | 16 % | 30 % | 904 |
| > 10k | 56 % | 12 % | 32 % | 147 |

Incluso en los grandes, un 30 % lo dibuja un solo experto. Ahí la causa es sobre todo partir y unir distinto. De los pequeños marcados por uno solo, el 59 % lo tocó otro experto con un dibujo distinto, y el 41 % nadie más lo vio.

**Pequeños (< 3.000 px): qué rasgo separa los unánimes de los de un solo experto.**

| Rasgo | Valor | todos | solo | ¿Separa? |
|---|---|---|---|---|
| **Oscuridad** (brillo mediano / fondo) | < 0,85 | **59 %** | 28 % | **Sí, es el rasgo más fuerte** |
| | 0,85-0,9 | 55 % | 30 % | |
| | 0,9-0,95 | 35 % | 44 % | |
| | > 0,95 | **10 %** | **66 %** | |
| **Distancia al centro** (radio) | < 0,8 | 46-47 % | 33-37 % | Sí, cerca del borde |
| | 0,8-0,9 | 36 % | 52 % | |
| | > 0,9 | 17 % | 83 % (n=6) | |
| **En grupo de pequeños** (otro pequeño a < 150 px) | no | 47 % | 36 % | Sí, moderado |
| | sí | 34 % | 47 % | |
| **Alargamiento** | < 2 | 34 % | 48 % | Algo |
| | > 5 | 48 % | 35 % | |
| Junto a un filamento grande (< 150 px) | — | 37 % frente a 43 % | — | No |
| Año (ciclo solar) | 2011-2022 | 34-50 % | — | No |

Oscuridad y forma combinadas: los pequeños oscuros (< 0,9) son unánimes el 56 % de las veces, sean alargados o no. Los pálidos son unánimes solo el 21 % si son redondeados y el 35 % si son alargados.

**Por experto:** hay expertos que marcan muchos pequeños que nadie más ve (010402, 010101: 53 % "solo") y otros que casi no (050102: 20 %). Encaja con los estilos "generoso" y "prudente" que ya habíamos visto.

## 2. Lo que dice lo publicado

Investigado con fuentes enlazadas. Fuente principal: Ahmadzadeh et al. 2024, *Scientific Data*, https://doi.org/10.1038/s41597-024-03876-y (el artículo de MAGFiLO).

- **Reglas de anotación de MAGFiLO:**
  - Se ignoran los filamentos "too small or too faint to exhibit any feature".
  - Se ignoran los **grupos de filamentos pequeños y cercanos**.
  - Se ignora todo lo que esté más allá de **±70° del meridiano central**.
  - No hay un mínimo en píxeles.
  - Los anotadores aprendieron a distinguir manchas solares.
- **Unir o partir:** se consultaba la evolución en el tiempo y se separaba si la quiralidad de los trozos se contradecía. El artículo admite que unos unían y otros partían.
- **No hubo consenso ni votación:** "we decided to leave the natural discrepancies". Las discrepancias entre expertos son a propósito.
- **Lo que significa para nosotros:** nuestros datos confirman las reglas. Lo pálido ("too faint"), los grupos de pequeños y lo cercano al borde son justo donde solo un experto marca.
- **Reglas de la competición:**
  - En la inferencia (test) solo se pueden usar las imágenes H-alfa del test.
  - Los datos externos sirven para entrenar, no para predecir.
  - Cada FP y cada FN cuestan lo mismo en PQ.
- **Trabajos previos con el mismo problema:**
  - Bernasconi 2005: descarta < 300 px y exige forma alargada.
  - Hao 2013: los muy pequeños solo valen si su relación de ejes es > 2.
  - Diercke 2024 (GONG, U-Net/YOLO): precisión de ~50 %, sobre todo por FP pequeños. Los etiquetadores discrepaban justo en las estructuras pequeñas.
  - Ninguno usa etiquetas de consenso.

## 3. Criterios candidatos para el juez

1. **Oscuridad:** pálido y pequeño = dudoso. Es la señal más fuerte (10 % frente a 59 % unánime).
2. **Grupo de pequeños:** varios pequeños juntos no se anotaban como filamentos (regla oficial; 34 % frente a 47 % unánime).
3. **Cerca del borde del disco** (radio > 0,8): dudoso (regla oficial de ±70°).
4. **Forma:** redondeado y pequeño = dudoso. Es el efecto más débil.

**Ojo:** el detector probablemente ya ha aprendido parte de esto. En la ronda 3 encuentra solo el 33 % de los filamentos pálidos, y sus inventos puros son sobre todo de oscuridad media. Hay que medir cuánto añade un juez que use estos rasgos explícitamente. Las reglas a mano de la ronda 3 (reglas 4 y 5) no ayudaron.

## 4. Prueba sin GPU con la ronda 3 (20 oct 2026)

Scripts: `analisis/prueba_juez_criterio.py` (sobre `predicciones_examen.json` y `errores_examen.csv` de la ronda 3) y `analisis/oraculo_consenso.py`.

**Quitar predicciones con los criterios no ayuda.** Base de la ronda 3: PQ 0,4405.

| Se quitan las predicciones pequeñas (< 3.000 px) que son... | Aciertos que se pierden | Inventos que se quitan | PQ |
|---|---|---|---|
| pálidas (brillo > 0,95) | 13 | 13 | 0,4385 |
| pálidas (> 0,93) | 59 | 46 | 0,4276 |
| cerca del borde (radio > 0,8) | 37 | 48 | 0,4353 |
| en grupo de pequeños | 103 | 36 | 0,4077 |
| redondeadas (alargamiento < 2) | 44 | 16 | 0,4270 |
| juez que aprende (árboles, validación cruzada por imagen, AUC 0,63) | 12-61 | 12-41 | 0,425-0,438 |

- Ni siquiera las predicciones más dudosas de la ronda 3 fallan más de la mitad de las veces. Las pequeñas pálidas aciertan el 50 %, y las oscuras el 81 %.
- **El detector ya aplica este criterio:** casi no propone pequeños pálidos (por eso solo encuentra el 33 % de los reales pálidos).

**La economía de la nota PQ.** Cada predicción se compara con cada experto de la imagen. Con la calidad de borde actual (SQ 0,68) y la nota 0,44, una predicción solo compensa si acierta más de ~1 de cada 3 veces:
- Por debajo de ese umbral resta, y conviene quitarla.
- Por encima suma.
- Lo que añadió el especialista de la ronda 4 acertaba el 34 %, justo en el umbral: por eso no cambió nada.

**Techo con dibujos perfectos** (296 imágenes con 2-3 expertos; la nota está inflada porque el dibujo copiado coincide al 100 % con el experto de donde sale):

| Estrategia | PQ |
|---|---|
| Copiar a un solo experto | 0,605 |
| Marcar solo lo que dibujan todos | 0,487 |
| Marcar lo que dibujan al menos 2 (mayoría) | 0,573 |
| Marcar todo lo que dibuja cualquiera (unión) | **0,658** |

Con esta métrica, **ser generoso paga más que ser estricto**: un filamento que marca un solo experto da un acierto pleno con él, y eso compensa los inventos frente a los demás. La conclusión no es quitar más, sino **encontrar más con al menos 1/3 de acierto y dibujarlos mejor** (cada +0,01 en SQ suma ~+0,0065 de PQ).
