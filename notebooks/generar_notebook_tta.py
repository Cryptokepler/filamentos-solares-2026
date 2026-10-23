"""Genera notebooks/ronda5_tta_envio.ipynb: ronda 5 (YOLO + refinador) mirando la imagen volteada (TTA) y envío.

No entrena nada. Mide en el examen si el TTA mejora a la ronda 5 normal y genera `submission.csv` del test con la
mejor de las dos. El código de filamentos/ va empaquetado (tar.gz en base64) en una celda.
Ejecutar desde la raíz del repositorio:
    python3 notebooks/generar_notebook_tta.py
"""
import base64
import io
import os
import tarfile

import nbformat as nbf

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODULOS = ["__init__.py", "datos.py", "clasico.py", "detector.py", "candidatos.py", "yolo.py"]

celdas = []


def md(texto):
    celdas.append(nbf.v4.new_markdown_cell(texto.strip()))


def code(texto):
    celdas.append(nbf.v4.new_code_cell(texto.strip()))


def empaquetar():
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for nombre in MODULOS:
            tar.add(os.path.join(RAIZ, "filamentos", nombre), arcname=f"filamentos/{nombre}")
    return base64.b64encode(buf.getvalue()).decode()


md("""
# Ronda 5 + imagen volteada (TTA) y envío

Decidido con jhonatan el 23 oct 2026. No entrena nada: usa el YOLO de la ronda 5 (`yolo_mejor.pt`) y el refinador de
la ronda 3 (`refinador.pt`).

1. **Examen:** YOLO mira cada imagen 4 veces (normal, volteada izquierda-derecha, arriba-abajo y las dos), se juntan
   las respuestas y se mide la nota con el juez de la ronda 5. Referencia sin TTA, mismos pesos:
   **PQ 0,4560 con umbral 0,25 (aciertos 878 · inventados 381 · escapados 489)**.
2. **Test:** genera `submission.csv` con la mejor opción (TTA o normal) y su umbral. **No envía nada.**

Requisitos en Kaggle: GPU T4, **Internet On** y como *Input*: la competición, el notebook `filamentos-ronda-5-yolo`
(`yolo_mejor.pt`) y el notebook `filamentos-detector-ronda-2` Version 4 (`refinador.pt`).
""")

code("""
# Configuración. PRUEBA_RAPIDA=True reduce todo para comprobar que el notebook funciona.
PRUEBA_RAPIDA = False

RES = 1024
CONF_MIN = 0.1             # confianza mínima de YOLO en cada vista
UMBRAL_REF = 0.5           # umbral del refinador de la ronda 3
UMBRALES_TTA = (0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4)  # sobre la confianza media de las 4 vistas
REF_NORMAL = dict(PQ=0.4560, umbral=0.25, aciertos=878, inventados=381, escapados=489)  # ronda 5 sin TTA
""")

code("""
import subprocess, sys
for paquete, modulo in (('ultralytics', 'ultralytics'), ('segmentation-models-pytorch', 'segmentation_models_pytorch'),
                        ('pycocotools', 'pycocotools')):
    try:
        __import__(modulo)
    except ImportError:
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', paquete], check=True)
""")

md("## Código del proyecto\\nEmpaquetado del repositorio `filamentos-solares-2026` (los mismos ficheros de `filamentos/`).")
code(f"""
import base64, io, os, tarfile
tarfile.open(fileobj=io.BytesIO(base64.b64decode('{empaquetar()}')), mode='r:gz').extractall('.')
print(sorted(os.listdir('filamentos')))
""")

md("## Datos, pesos y separación estudio / examen\\nMisma separación que las rondas 1-5.")
code("""
import csv, glob, hashlib, json, time
import numpy as np, pandas as pd, torch
import segmentation_models_pytorch as smp
from PIL import Image
from ultralytics import YOLO
from filamentos.datos import cargar_anotaciones
from filamentos import candidatos as C, yolo as Y

disp = 'cuda' if torch.cuda.is_available() else 'cpu'
T0 = time.time()

def buscar(patron):
    for raiz in ('/kaggle/input', os.environ.get('MAGFILO', '.'), os.environ.get('PESOS', '.')):
        r = sorted(glob.glob(f'{raiz}/**/{patron}', recursive=True))
        if r:
            return r[0]
    return None

json_path = buscar('MAGFiLO_1.0_Annotations_kaggle2026_train.json')
assert json_path, 'No encuentro los datos: añade la competición como Input del notebook'
DATOS = os.path.dirname(os.path.dirname(json_path))
PESOS_REF = buscar('refinador.pt')
PESOS_YOLO = buscar('yolo_mejor.pt')
assert PESOS_REF, ('No encuentro refinador.pt: añade como Input el notebook filamentos-detector-ronda-2 (Version 4). '
                   'Hay en /kaggle/input: ' + str(glob.glob('/kaggle/input/*/*')))
if not PRUEBA_RAPIDA:
    assert PESOS_YOLO, ('No encuentro yolo_mejor.pt: añade como Input el notebook filamentos-ronda-5-yolo. '
                        'Hay en /kaggle/input: ' + str(glob.glob('/kaggle/input/*/*')))
print('Datos en', DATOS, '| refinador:', PESOS_REF, '| YOLO:', PESOS_YOLO)

ANOT = cargar_anotaciones(DATOS)
nombres = sorted(n for n in ANOT if os.path.exists(f'{DATOS}/train/train_images/{n}'))

def es_val(n):
    return int(hashlib.md5(n.encode()).hexdigest(), 16) % 100 < 15
val = [n for n in nombres if es_val(n)]
TEST = os.environ.get('TEST_DIR', f'{DATOS}/test/test_images')
test = sorted(os.listdir(TEST)) if os.path.isdir(TEST) else []
if PRUEBA_RAPIDA:
    val, test = val[:2], test[:2]
    PESOS_YOLO = PESOS_YOLO or 'yolo11n-seg.pt'
print(len(val), 'imágenes de examen,', len(test), 'de test')

ref = smp.Unet('resnet18', encoder_weights=None, in_channels=3, classes=1).to(disp)
ref.load_state_dict(torch.load(PESOS_REF, map_location=disp))
modelo = YOLO(PESOS_YOLO)
det_normal = Y.detector(modelo, RES, CONF_MIN)
det_tta = Y.detector_tta(modelo, RES, CONF_MIN)
reales_val = {n: Y.reales_rle(ANOT[n]) for n in val}
""")

md("""
## 1. Examen con la imagen volteada
Cada candidato guarda la confianza de cada una de las 4 vistas (`vistas`), para probar otras reglas sin GPU.
""")
code("""
t = time.time()
cand_tta = {}
for i, n in enumerate(val):
    grande = np.asarray(Image.open(f'{DATOS}/train/train_images/{n}').convert('L'), np.uint8)
    cand_tta[n], _, _ = C.procesar(None, ref, grande, disp, RES, 0.0, UMBRAL_REF, 0, detectar=det_tta)
    if i % 20 == 0:
        print(f'{i + 1}/{len(val)} imágenes, {time.time() - t:.0f} s', flush=True)
with open('candidatos_tta_examen.json', 'w') as fh:
    json.dump(cand_tta, fh, separators=(',', ':'))
print(f'{sum(map(len, cand_tta.values()))} candidatos en el examen ({(time.time() - t) / 60:.0f} min)')

filas = []
for u in UMBRALES_TTA:
    r = Y.nota_umbral(cand_tta, reales_val, u, 'rle_ref', por_tamano=True)
    filas.append(dict(umbral=u, PQ=round(r['pq'], 4), SQ=round(r['sq'], 3), aciertos=r['tp'], inventados=r['fp'],
                      escapados=r['fn'], esc_menos_1k=r['escapados_menos_1k'], esc_1k_3k=r['escapados_1k_3k'],
                      esc_mas_3k=r['escapados_mas_3k']))
tabla = pd.DataFrame(filas)
print('\\n=== Examen, YOLO + refinador CON imagen volteada (confianza media de 4 vistas) ===')
print(tabla.to_string(index=False))
mejor = tabla.loc[tabla.PQ.idxmax()]
print(f"\\nSin TTA (ronda 5): PQ {REF_NORMAL['PQ']} con umbral {REF_NORMAL['umbral']} | aciertos "
      f"{REF_NORMAL['aciertos']} inventados {REF_NORMAL['inventados']} escapados {REF_NORMAL['escapados']}")
print(f'Con TTA:           PQ {mejor.PQ} con umbral {mejor.umbral} | aciertos {int(mejor.aciertos)} '
      f'inventados {int(mejor.inventados)} escapados {int(mejor.escapados)}')
USAR_TTA = mejor.PQ > REF_NORMAL['PQ']
UMBRAL = float(mejor.umbral) if USAR_TTA else REF_NORMAL['umbral']
print(f"Para el envío: {'CON' if USAR_TTA else 'SIN'} imagen volteada, umbral {UMBRAL}")
""")

md("## 2. Test y `submission.csv`\nEl archivo se genera pero **no se envía**.")
code("""
t = time.time()
filas, cand_test = [], {}
det = det_tta if USAR_TTA else det_normal
for i, f in enumerate(test):
    grande = np.asarray(Image.open(f'{TEST}/{f}').convert('L'), np.uint8)
    cand_test[f], _, _ = C.procesar(None, ref, grande, disp, RES, 0.0, UMBRAL_REF, 0, detectar=det)
    img_id = os.path.splitext(f)[0]
    for k, r in enumerate(Y.seleccion(cand_test[f], UMBRAL), start=1):
        filas.append((f'{img_id}_{k}', r['counts'].decode('ascii') if isinstance(r['counts'], bytes) else r['counts']))
    if i % 30 == 0:
        print(f'{i + 1}/{len(test)} imágenes, {len(filas)} filamentos, {time.time() - t:.0f} s', flush=True)
with open('candidatos_test.json', 'w') as fh:
    json.dump(cand_test, fh, separators=(',', ':'))
with open('submission.csv', 'w', newline='') as fh:
    w = csv.writer(fh)
    w.writerow(['filament_id', 'segmentation_rle'])
    w.writerows(filas)

ids = [a for a, _ in filas]
assert len(ids) == len(set(ids))
if filas:
    from filamentos.datos import rle_a_mascara
    assert rle_a_mascara(filas[0][1]).shape == (2048, 2048)
else:
    print('AVISO: no hay filamentos en el test (¿falta la carpeta test?)')
print(len(filas), 'filamentos en', len(set(i.rsplit('_', 1)[0] for i in ids)), 'imágenes -> submission.csv')
print(f'Tiempo total: {(time.time() - T0) / 60:.0f} min')
""")

nb = nbf.v4.new_notebook()
nb["cells"] = celdas
nb["metadata"] = {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                  "language_info": {"name": "python"},
                  "accelerator": "GPU"}
salida = os.path.join(RAIZ, "notebooks", "ronda5_tta_envio.ipynb")
nbf.write(nb, salida)
print("Notebook generado:", salida, os.path.getsize(salida), "bytes")
