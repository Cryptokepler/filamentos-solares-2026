# Métrica: Panoptic Quality (PQ)

Definida en [Kirillov 2019] y adoptada por la competición.

1. Cada filamento predicho se empareja con uno real si su **IoU > 0,5** (intersección / unión). Con ese umbral, cada filamento solo puede emparejarse con uno.
2. Aciertos = TP, inventados = FP, olvidados = FN.
3. **SQ** (calidad de forma) = IoU medio de los aciertos.
4. **RQ** (calidad de detección) = TP / (TP + ½ FP + ½ FN).
5. **PQ = SQ × RQ**, entre 0 y 1.

Partir un filamento en dos suele dejar ambos trozos por debajo de IoU 0,5 → 1 FN + 2 FP. Fusionar dos → 2 FN + 1 FP. Por eso la competición insiste en la fragmentación y la sobrefusión, y además las mide aparte en la rúbrica.

## Referencias de nivel **[medido]**
| Comparación | PQ |
|---|---|
| Un experto contra otro (296 imágenes, 598 pares) | 0,340 |
| Línea base clásica sin IA (100 imágenes) | 0,098 |

Los organizadores dicen que "cualquier PQ superior a 0,35 es de gran valor". Implementación: `filamentos/metrica.py`.
