# Qué es un filamento solar

## Definición
Un filamento es una nube de plasma relativamente frío y denso suspendida en la corona solar por el campo magnético, a alturas de hasta ~100 000 km sobre la cromosfera [Mackay 2010]. Visto sobre el disco solar en H-alfa aparece **oscuro**, porque absorbe la luz que llega de abajo. Visto en el borde (limbo) aparece **brillante** contra el cielo negro y se llama **protuberancia**: es el mismo objeto [Mackay 2010].

> Consecuencia para el reto: los expertos casi no dibujan filamentos cerca del borde. Solo el 0,8 % de los filamentos anotados está más allá del 90 % del radio y ninguno más allá del 95 % **[medido]**.

## Valores típicos [Parenti 2014]
| Magnitud | Valor |
|---|---|
| Temperatura | 7 000 – 20 000 K (núcleo ~8 200 K); la corona que lo rodea está a ~1 000 000 K |
| Densidad electrónica | 10⁹ – 10¹¹ cm⁻³, ~100 veces más que la corona |
| Longitud | ~30 000 – 110 000 km (varias veces el diámetro de la Tierra, 12 700 km) |
| Anchura | ~1 000 – 10 000 km; hilos internos de 150 – 450 km |
| Altura | ~26 000 km de media en filamentos largos |
| Masa | 10¹⁴ – 2·10¹⁵ g |
| Vida | De minutos u horas (zonas activas) a meses (filamentos quietos) |

A 2048 px por imagen, el disco solar mide ~1 870 px de diámetro **[medido]**, así que 1 px ≈ 750 km. Un filamento de 30 000 km ocupa ~40 px de largo.

## Dónde se forman
- Siempre sobre una **línea de inversión de polaridad**: la frontera donde el campo magnético de la superficie cambia de signo [Mackay 2010].
- Alrededor hay un **canal de filamento**, con fibrillas alineadas con esa línea [Mackay 2010].
- Tres tipos [Mackay 2010]:
  - **de región activa**: junto a manchas solares, pequeños y de vida corta;
  - **intermedios**: en el borde de las regiones activas;
  - **quietos**: en Sol tranquilo, a cualquier latitud, incluida la "corona polar".

## Partes de un filamento
- **Espina (spine):** el eje a lo largo de la parte superior. MAGFiLO la anota como una línea.
- **Púas (barbs):** prolongaciones finas que bajan desde la espina hacia la superficie. Son lo más difícil de segmentar: la página de la competición las menciona como primer reto.
- **Patas:** los dos extremos.
- Todo está hecho de hilos finos que cambian en minutos, aunque el conjunto dure días [Mackay 2010].

## Quiralidad (las categorías Left / Right del dataset)
- Un filamento es **dextral** o **sinistral** según hacia dónde apunte su campo magnético axial, visto desde el lado de polaridad positiva [Mackay 2010; Ouyang 2017].
- Regla de Martin: los dextrales tienen púas "hacia la derecha" (right-bearing) y los sinistrales "hacia la izquierda" (left-bearing) [Mackay 2010].
- **Regla hemisférica:** en el norte predominan los dextrales y en el sur los sinistrales; se cumple en ~92 % de 571 filamentos eruptivos estudiados [Ouyang 2017].
- En MAGFiLO, la quiralidad se anota como Left, Right o Unidentifiable [MAGFiLO]. **La competición no la puntúa**, pero la forma de las púas que la delata sí cuenta para la máscara.

## Por qué importan
Cuando el campo que sostiene un filamento se desestabiliza, el filamento puede entrar en erupción y formar parte de una **eyección de masa coronal (CME)** [Parenti 2014; Mackay 2010]. Una CME dirigida a la Tierra puede dañar redes eléctricas y satélites, alterar el GPS y suponer riesgo de radiación para tripulaciones y astronautas (página de la competición). Las imágenes H-alfa de GONG se usan para el pronóstico operativo del clima espacial en NOAA SWPC, la Fuerza Aérea de EE. UU., el Met Office británico y NICT en Japón [Tlatov y Pevtsov].
