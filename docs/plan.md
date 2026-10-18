# Reto Kaggle: Segmentación de filamentos solares 2026

Competición: https://www.kaggle.com/competitions/filament-segmentation-2026
Actualizado el 18 oct 2026 con la página de Overview que pegó jhonatan.

## Lo que está confirmado

**Fecha límite: 15 de noviembre de 2026, 23:59 CDT.** Quedan unas 4 semanas. Ganadores el 30 de noviembre.

**La tarea.** Imágenes H-alfa del Sol (red GONG, 2048 × 2048). Hay que generar una máscara **por cada filamento individual**, incluyendo estructuras finas (las "púas"), sin meter fondo. Se permite cualquier método, clásico o deep learning. Se pueden usar datos externos, pero en inferencia solo se usan las imágenes de test y no se pueden usar otros metadatos de verdad para entrenar.

**Los tres retos que señalan los organizadores**
1. Estructuras finas (púas).
2. Ruido y artefactos de telescopios en tierra.
3. Continuidad: no partir un filamento en "islas" ni juntar regiones oscuras cercanas.

**La métrica: Panoptic Quality (PQ).**
- Un filamento predicho y uno real se emparejan si su IoU > 0,5.
- PQ = (IoU medio de los aciertos) × (aciertos / (aciertos + ½ falsos positivos + ½ olvidos)).
- Penalizaciones extra por fragmentación y sobrefusión (correspondencias uno a muchos y muchos a uno), y por **eficiencia computacional**.
- El leaderboard público usa ~50 % del test; el resto es privado.
- Los organizadores (20 ago): "cualquier PQ por encima de 0,35 es de gran valor para nosotros". También avisan de que algunos participantes intentaron trucar el sistema, así que el leaderboard público no es fiable del todo.
- Hay un cuaderno oficial de autoevaluación: https://www.kaggle.com/code/azimahmadzadeh/self-evaluation-notebook

**Cómo se decide el ganador (rúbrica)**
- 70 % cuantitativo: PQ, distribución de Dice e IoU, distribución de casos uno a muchos y muchos a uno.
- 30 % cualitativo: descripción del pipeline, aspecto visual de las máscaras y **calidad del código** (modularidad, documentación).
- El premio (hasta 3.000 USD) se reparte entre los 3 primeros en proporción a la puntuación al cuadrado.

**Formato de envío.** Un CSV con una fila por filamento:
```
filament_id,segmentation_rle
20150125172714Mh_1,<RLE COCO compacto>
20150125172714Mh_2,<...>
```
- `filament_id` = id de la imagen + `_n` para que sea único.
- Solo los "counts" del RLE de `pycocotools`, sin comillas. Tamaño fijo 2048 × 2048.

**Entrega final (obligatoria para optar a premio), mediante un formulario de Google:**
- Informe técnico de 4 páginas en PDF con la plantilla de Overleaf.
- Repositorio Git **público** con `requirements.txt` (versiones fijas) y un notebook Jupyter que ejecute todo el pipeline y reproduzca las predicciones sin archivos adicionales.

**Datos (MAGFiLO).** Según repositorios de otros participantes (⚠️ confirmar en la pestaña Data): ~707 imágenes de entrenamiento, ~180 de test, anotaciones COCO en JSON con polígonos, espina, caja y quiralidad. Algunas imágenes tienen varios anotadores, que coinciden entre sí con un PQ de solo ~0,36.

## Estrategia central (idea de jhonatan, 18 oct)
Imitar cómo se contrastan los expertos: **proponer → contrastar → filtrar**.
1. Varios modelos distintos (3–5) proponen filamentos; se queda lo que ve la mayoría (ensamble con voto).
2. Un "supervisor" revisa cada candidato con las reglas de `reglas-filamentos.md` (borde, tamaño, oscuridad, forma) y descarta los dudosos.
No se usan agentes conversacionales: la competición puntúa la eficiencia y exige código reproducible.

## Plan en 4 semanas

**Semana 1 (18–25 oct): datos, visor y primer envío**
- Tú: aceptar las reglas y descargar los datos (o pasarme acceso).
- Yo: **visor** de imágenes con los filamentos superpuestos (cada uno con su color, cambiar de anotador, ver espina).
- Juntos: mirar 20–30 imágenes y anotar qué cuenta como un filamento, cuándo dos trozos son uno solo y cómo afectan el borde del disco y las nubes.
- Yo: PQ local idéntica a la oficial, validación separada por año/observatorio, y primer envío sencillo (umbral de zonas oscuras + componentes) para aprender el formato.

**Semana 2 (26 oct–1 nov): modelo de deep learning**
- U-Net con encoder preentrenado → máscara de filamento → separar instancias.
- Entrenamiento en GPU gratuita de Kaggle (~30 h/semana).

**Semana 3 (2–8 nov): atacar la métrica**
- Aquí manda tu razonamiento: reglas para unir fragmentos, separar filamentos pegados, tamaño mínimo y umbrales. Cada idea se mide con la PQ local.
- Ensamble ligero y TTA, sin olvidar que la eficiencia también puntúa.

**Semana 4 (9–15 nov): cierre**
- Congelar el modelo el 12 nov.
- Repositorio público ordenado: `requirements.txt`, notebook de punta a punta y README.
- Informe de 4 páginas en Overleaf (yo redacto, tú revisas).
- Enviar el formulario **antes del 15 nov**, sin dejarlo para la última hora.

## Reparto
- Tú: observar, formular hipótesis, decidir qué probar y revisar el informe.
- Yo: código, visor, métrica, experimentos y borradores.
- Registro de experimentos: qué probamos, qué PQ local dio y qué dio en el leaderboard.

## Fuentes
- Página Overview de Kaggle (pegada por jhonatan)
- https://solarnews.aas.org/2026/solar-filament-segmentation-challenge-2026/
- https://bigdataieee.org/BigData2026/cup/solar-filament-segmentation/
- https://github.com/us-llm-engineer/kaggle-computer-vision-benchmarks
- https://github.com/zhang0894/Solar-Filament-Segmentation-Challenge-2026-by-Anon-Tokyo
