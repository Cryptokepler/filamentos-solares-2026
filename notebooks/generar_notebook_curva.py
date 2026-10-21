"""Genera notebooks/curva_aprendizaje.ipynb: YOLO11-seg entrenado con el 25 %, 50 % y 75 % de las imágenes de estudio.

El código de filamentos/ va empaquetado (tar.gz en base64) en una celda, para que el notebook quepa al subirlo.
Ejecutar desde la raíz del repositorio:
    python3 notebooks/generar_notebook_curva.py
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
# Curva de aprendizaje: ¿más imágenes suben la nota?

Decidido con jhonatan el 21 oct 2026 (plan en `docs/experimentos.md`, "Curva de aprendizaje"). Mismo sistema que la
ronda 5 (YOLO11m-seg a 1024 px + refinador de la ronda 3) y mismo examen de 121 imágenes; solo cambia cuántas
imágenes de estudio ve YOLO: 25 %, 50 % y 75 % (cada grupo contiene al anterior). El 100 % ya está medido en la
ronda 5: **PQ 0,4532** (umbral 0,2; TP 922 · FP 481 · FN 445).

Requisitos en Kaggle: GPU T4, **Internet On** y como *Input* la competición y `filamentos-detector-ronda-2`
Version 4 (`refinador.pt`). No genera envío.
""")

code("""
# Configuración. PRUEBA_RAPIDA=True reduce todo para comprobar que el notebook funciona.
PRUEBA_RAPIDA = False

FRACCIONES = (0.25, 0.5, 0.75)  # parte de las imágenes de estudio que ve YOLO en cada paso
RES = 1024
MODELO = 'yolo11m-seg.pt'
HORAS = 1.75               # tope de entrenamiento por paso (el 100 % ya no mejoraba desde ~1,7 h)
EPOCAS = 300
PACIENCIA = 25
LOTE = 6
CONF_MIN = 0.1
UMBRAL_REF = 0.5
UMBRALES = (0.15, 0.2, 0.25, 0.3, 0.4, 0.5)
SEMILLA = 0
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

md("## Código del proyecto\nEmpaquetado del repositorio `filamentos-solares-2026` (los mismos ficheros de `filamentos/`).")
code(f"""
import base64, io, os, tarfile
tarfile.open(fileobj=io.BytesIO(base64.b64decode('{empaquetar()}')), mode='r:gz').extractall('.')
print(sorted(os.listdir('filamentos')))
""")

md("## Datos, separación estudio / examen y refinador de la ronda 3\nMisma separación que las rondas 1-5.")
code("""
import glob, hashlib, json, shutil, time
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
        r = glob.glob(f'{raiz}/**/{patron}', recursive=True)
        if r:
            return r[0]
    return None

json_path = buscar('MAGFiLO_1.0_Annotations_kaggle2026_train.json')
assert json_path, 'No encuentro los datos: añade la competición como Input del notebook'
DATOS = os.path.dirname(os.path.dirname(json_path))
PESOS_REF = buscar('refinador.pt')
if not PRUEBA_RAPIDA:
    assert PESOS_REF, ('No encuentro refinador.pt: añade como Input el notebook filamentos-detector-ronda-2 '
                       '(Version 4). Hay en /kaggle/input: ' + str(glob.glob('/kaggle/input/*/*')))
print('Datos en', DATOS, '| refinador:', PESOS_REF)

ANOT = cargar_anotaciones(DATOS)
nombres = sorted(n for n in ANOT if os.path.exists(f'{DATOS}/train/train_images/{n}'))

def es_val(n):
    return int(hashlib.md5(n.encode()).hexdigest(), 16) % 100 < 15
val = [n for n in nombres if es_val(n)]
tr = [n for n in nombres if not es_val(n)]
if PRUEBA_RAPIDA:
    tr, val = tr[:8], val[:3]
    FRACCIONES, RES, MODELO, EPOCAS, HORAS, LOTE = (0.5,), 256, 'yolo11n-seg.pt', 1, 0.02, 4
# Orden fijo al azar: cada fracción es el principio de la lista, así el 25 % está dentro del 50 %, etc.
orden = list(np.random.default_rng(SEMILLA).permutation(tr))
print(len(tr), 'imágenes de estudio,', len(val), 'de examen')

ref = smp.Unet('resnet18', encoder_weights=None, in_channels=3, classes=1).to(disp)
if PESOS_REF:
    ref.load_state_dict(torch.load(PESOS_REF, map_location=disp))
reales_val = {n: Y.reales_rle(ANOT[n]) for n in val}
""")

md("""
## Los pasos de la curva
En cada paso: entrenar YOLO con esa parte, sacar los candidatos del examen y medir con el juez de la ronda 5
(confianza >= umbral, unir los que se tocan). Se guarda `candidatos_curva_<pct>.json` de cada paso.
""")
code("""
resumen = []
for frac in FRACCIONES:
    pct = int(round(100 * frac))
    parte = sorted(orden[:max(1, int(round(frac * len(orden))))])
    print(f'\\n########## Paso {pct} %: {len(parte)} imágenes de estudio ##########', flush=True)
    t = time.time()
    carpeta = f'/tmp/yolo_datos_{pct}'
    yaml = Y.preparar_datos(DATOS, ANOT, parte, val, carpeta, res=RES)
    modelo = YOLO(MODELO)
    modelo.train(data=yaml, imgsz=RES, epochs=EPOCAS, time=HORAS, patience=PACIENCIA, batch=LOTE, single_cls=True,
                 fliplr=0.5, flipud=0.5, degrees=0, hsv_h=0, hsv_s=0, hsv_v=0.3, mosaic=1.0, close_mosaic=10,
                 overlap_mask=False, cos_lr=True, project='/tmp/yolo_runs', name=f'c{pct}', exist_ok=True,
                 plots=False, workers=2, device=0 if disp == 'cuda' else 'cpu', verbose=False, seed=SEMILLA)
    hist = pd.read_csv(f'/tmp/yolo_runs/c{pct}/results.csv')
    hist.to_csv(f'yolo_historial_{pct}.csv', index=False)
    min_ent = (time.time() - t) / 60
    print(f'Entrenamiento: {min_ent:.0f} min, {len(hist)} vueltas', flush=True)

    det = Y.detector(YOLO(f'/tmp/yolo_runs/c{pct}/weights/best.pt'), RES, CONF_MIN)
    cand = {}
    for n in val:
        grande = np.asarray(Image.open(f'{DATOS}/train/train_images/{n}').convert('L'), np.uint8)
        cand[n], _, _ = C.procesar(None, ref, grande, disp, RES, CONF_MIN, UMBRAL_REF, 0, detectar=det)
    with open(f'candidatos_curva_{pct}.json', 'w') as fh:
        json.dump(cand, fh, separators=(',', ':'))
    filas = []
    for u in UMBRALES:
        r = Y.nota_umbral(cand, reales_val, u, 'rle_ref', por_tamano=True)
        filas.append(dict(umbral=u, PQ=round(r['pq'], 4), SQ=round(r['sq'], 3), aciertos=r['tp'], inventados=r['fp'],
                          escapados=r['fn'], esc_menos_1k=r['escapados_menos_1k'], esc_1k_3k=r['escapados_1k_3k'],
                          esc_mas_3k=r['escapados_mas_3k']))
    tabla = pd.DataFrame(filas)
    print(f'\\n=== Paso {pct} %: examen, YOLO + refinador ===')
    print(tabla.to_string(index=False), flush=True)
    mejor = tabla.loc[tabla.PQ.idxmax()]
    resumen.append(dict(paso=f'{pct} %', imagenes=len(parte), vueltas=len(hist), minutos=round(min_ent),
                        umbral=mejor.umbral, PQ=mejor.PQ, aciertos=int(mejor.aciertos),
                        inventados=int(mejor.inventados), escapados=int(mejor.escapados),
                        esc_menos_1k=int(mejor.esc_menos_1k)))
    pd.DataFrame(resumen).to_csv('curva_resumen.csv', index=False)
""")

md("## La curva\nEl 100 % es la ronda 5 (mismo sistema, medido en el mismo examen).")
code("""
resumen.append(dict(paso='100 % (ronda 5)', imagenes=len(tr), vueltas=88, minutos=153, umbral=0.2, PQ=0.4532,
                    aciertos=922, inventados=481, escapados=445, esc_menos_1k=236))
print(pd.DataFrame(resumen).to_string(index=False))
print(f'Tiempo total: {(time.time() - T0) / 60:.0f} min')
""")

nb = nbf.v4.new_notebook()
nb["cells"] = celdas
nb["metadata"] = {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                  "language_info": {"name": "python"},
                  "accelerator": "GPU"}
salida = os.path.join(RAIZ, "notebooks", "curva_aprendizaje.ipynb")
nbf.write(nb, salida)
print("Notebook generado:", salida, os.path.getsize(salida), "bytes")
