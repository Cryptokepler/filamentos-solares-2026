"""Genera notebooks/guardar_candidatos.ipynb (paso 1 del tribunal) incrustando el código de filamentos/.

Solo predice, no entrena. Ejecutar desde la raíz del repositorio:
    python3 notebooks/generar_notebook_candidatos.py
"""
import os

import nbformat as nbf

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODULOS = ["__init__.py", "datos.py", "metrica.py", "clasico.py", "observador.py", "detector.py", "candidatos.py"]

celdas = []


def md(texto):
    celdas.append(nbf.v4.new_markdown_cell(texto.strip()))


def code(texto):
    celdas.append(nbf.v4.new_code_cell(texto.strip()))


md("""
# Paso 1 del tribunal: guardar todos los candidatos

Idea de jhonatan (20 oct 2026): el detector ya no decide, solo propone. Dispara "a discreción" y un tribunal
de jueces decide (ver `docs/arquitectura-tribunal.md`). Este notebook **no entrena nada**:

1. Pasa el detector de la ronda 3 en **modo generoso** (confianza desde 0,1 en vez de 0,8) por las 121 imágenes
   del examen y por las 586 de estudio.
2. De cada candidato guarda su silueta (del detector y del refinador), su ficha (confianza, tamaño, oscuridad,
   contraste con su entorno, forma, distancia al borde) y, en el examen, un recorte para el juez A.
3. Imprime la **tabla de acierto por franja de confianza** y cuántos filamentos reales están al alcance.
4. Comprueba que la cadena de la ronda 3 sigue dando ~0,4405 (si no, algo está mal cargado).

No genera `submission.csv`. Requisitos en Kaggle: *Accelerator* = GPU T4, y como *Input*: la competición y el
*Output* del notebook `filamentos-detector-ronda-2` (con `detector_mejor.pt` y `refinador.pt`).
""")

code("""
# Configuración. PRUEBA_RAPIDA=True reduce todo para comprobar que el notebook funciona.
PRUEBA_RAPIDA = False

RES = 1024              # resolución del detector general (la de la ronda 3)
CONF_MIN = 0.1          # el detector propone todo lo que tenga al menos esta confianza
UMBRAL_REF = 0.5        # umbral del refinador de la ronda 3
LADO_RECORTE = 128      # recortes para el juez A (solo examen)
CON_ESTUDIO = True      # también las 586 imágenes de estudio (sin recortes)
""")

code("""
import subprocess, sys
try:
    import segmentation_models_pytorch as smp
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "segmentation-models-pytorch"], check=True)
    import segmentation_models_pytorch as smp
try:
    import pycocotools
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pycocotools"], check=True)
""")

md("## Código del proyecto\nCopiado del repositorio `filamentos-solares-2026`.")
code("import os\nos.makedirs('filamentos', exist_ok=True)")
for nombre in MODULOS:
    with open(os.path.join(RAIZ, "filamentos", nombre)) as f:
        fuente = f.read()
    code(f"%%writefile filamentos/{nombre}\n{fuente or '# paquete del proyecto'}")

md("## Datos, separación estudio / examen y pesos de la ronda 3\nMisma separación que las rondas 1-4.")
code("""
import glob, hashlib, json, time
import numpy as np, pandas as pd, torch
from PIL import Image
from filamentos.datos import cargar_anotaciones, mascaras_de
from filamentos.metrica import Resultado, evaluar_imagen, resumen
from filamentos import detector as D, candidatos as C

disp = 'cuda' if torch.cuda.is_available() else 'cpu'
T0 = time.time()

def buscar(patron):
    for raiz in ('/kaggle/input', os.environ.get('MAGFILO', '.'), os.environ.get('PESOS', '.')):
        r = glob.glob(f'{raiz}/**/{patron}', recursive=True)
        if r:
            return r[0]
    return None

json_path = buscar('MAGFiLO_1.0_Annotations_kaggle2026_train.json')
assert json_path, 'No encuentro los datos: añade la competición como Input del notebook'
DATOS = os.path.dirname(os.path.dirname(json_path))
PESOS_DET, PESOS_REF = buscar('detector_mejor.pt'), buscar('refinador.pt')
if not PRUEBA_RAPIDA:
    assert PESOS_DET and PESOS_REF, ('No encuentro los pesos de la ronda 3: añade como Input el Output del '
                                     'notebook filamentos-detector-ronda-2. Hay en /kaggle/input: '
                                     + str(glob.glob('/kaggle/input/*/*')))
print('Datos en', DATOS, '| pesos:', PESOS_DET, PESOS_REF)

ANOT = cargar_anotaciones(DATOS)
nombres = sorted(n for n in ANOT if os.path.exists(f'{DATOS}/train/train_images/{n}'))

def es_val(n):
    return int(hashlib.md5(n.encode()).hexdigest(), 16) % 100 < 15
val = [n for n in nombres if es_val(n)]
tr = [n for n in nombres if not es_val(n)]
if PRUEBA_RAPIDA:
    tr, val = tr[:3], val[:3]
print(len(tr), 'imágenes de estudio,', len(val), 'de examen')

gen = D.construir_detector(False, RES).to(disp)
ref = smp.Unet('resnet18', encoder_weights=None, in_channels=3, classes=1).to(disp)
if PESOS_DET:
    gen.load_state_dict(torch.load(PESOS_DET, map_location=disp))
    ref.load_state_dict(torch.load(PESOS_REF, map_location=disp))
""")

md("## Candidatos del examen (con recortes) y comprobación de la ronda 3")
code("""
def reales_rle(n):
    return [[D.a_rle(m) for m in mascaras_de(anns)] for anns in ANOT[n].values()]

def gris2048(n):
    return np.asarray(Image.open(f'{DATOS}/train/train_images/{n}').convert('L'), np.uint8)

cand_val, recortes, indice, reales_val = {}, [], [], {}
r3 = Resultado()
t = time.time()
for i, n in enumerate(val):
    grande = gris2048(n)
    cands, recs, pred_r3 = C.procesar(gen, ref, grande, disp, RES, CONF_MIN, UMBRAL_REF, LADO_RECORTE)
    cand_val[n] = cands
    recortes.append(recs)
    indice += [(n, k) for k in range(len(cands))]
    reales_val[n] = reales_rle(n)
    for rles in reales_val[n]:
        r3 += evaluar_imagen([D.mask_utils.decode(r).astype(bool) for r in rles], pred_r3)
    if i % 20 == 0:
        print(f'{i + 1}/{len(val)} imágenes, {sum(map(len, cand_val.values()))} candidatos, {time.time() - t:.0f} s', flush=True)
print('Cadena de la ronda 3 (debe dar ~0,4405):', resumen(r3))
print(f'{sum(map(len, cand_val.values()))} candidatos en el examen ({time.time() - t:.0f} s)')

np.savez_compressed('recortes_examen.npz', recortes=np.concatenate(recortes),
                    imagen=np.array([a for a, _ in indice]), k=np.array([b for _, b in indice]))
with open('candidatos_examen.json', 'w') as fh:
    json.dump(cand_val, fh, separators=(',', ':'))
""")

md("""
## La tabla: acierto por franja de confianza
Cada candidato se compara con cada experto, como la métrica. **% acierto** se compara con la raya del tribunal:
PQ ÷ (2 × IoU medio) ≈ 32 %. **Alcanzables** = filamentos reales que algún candidato dibuja con IoU > 0,5: es el
techo de lo que el tribunal puede rescatar.
""")
code("""
pd.set_option('display.width', 200)
for clave, nombre in (('rle_ref', 'con refinador'), ('rle', 'silueta del detector')):
    filas, n_reales, alcanzables, n_dup = C.tabla_franjas(cand_val, reales_val, clave)
    print(f'\\n=== Examen, {nombre} ({n_dup} candidatos repetidos descartados) ===')
    print(pd.DataFrame(filas).to_string(index=False))
    print(f'Filamentos reales: {n_reales} | al alcance de algún candidato: {alcanzables} '
          f'({100 * alcanzables / n_reales:.0f} %) | la ronda 3 acertaba {r3.tp}')
""")

md("## Candidatos de las imágenes de estudio\nSin recortes. El detector las estudió, así que aquí lo verá todo más fácil: sirven para comparar, no para medir.")
code("""
if CON_ESTUDIO:
    cand_tr, t = {}, time.time()
    for i, n in enumerate(tr):
        cand_tr[n], _, _ = C.procesar(gen, ref, gris2048(n), disp, RES, CONF_MIN, UMBRAL_REF, 0)
        if i % 100 == 0:
            print(f'{i + 1}/{len(tr)} imágenes, {time.time() - t:.0f} s', flush=True)
    with open('candidatos_estudio.json', 'w') as fh:
        json.dump(cand_tr, fh, separators=(',', ':'))
    filas, n_reales, alcanzables, n_dup = C.tabla_franjas(cand_tr, {n: reales_rle(n) for n in tr})
    print('\\n=== Estudio, con refinador (el detector ya las conoce) ===')
    print(pd.DataFrame(filas).to_string(index=False))
    print(f'Filamentos reales: {n_reales} | al alcance: {alcanzables} ({100 * alcanzables / n_reales:.0f} %)')
print(f'Tiempo total: {(time.time() - T0) / 60:.0f} min')
""")

nb = nbf.v4.new_notebook()
nb["cells"] = celdas
nb["metadata"] = {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                  "language_info": {"name": "python"},
                  "accelerator": "GPU"}
salida = os.path.join(RAIZ, "notebooks", "guardar_candidatos.ipynb")
nbf.write(nb, salida)
print("Notebook generado:", salida)
