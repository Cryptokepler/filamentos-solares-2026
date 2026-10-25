"""Genera notebooks/juez_<vista>.ipynb: un juez más para el jurado, YOLO entrenado con otra forma de ver la foto.

Igual que la ronda 7, pero con la vista que se elija de filamentos/yolo.py `VISTA` (25 oct 2026: "venas", el juez
angiografía). El refinador es el de la ronda 3, sin cambios. Mide el examen, guarda los candidatos del examen y del
test (para el jurado, que se mide sin GPU con analisis/jurado_yolo.py) y deja `submission.csv` (no envía nada).
Ejecutar desde la raíz del repositorio:
    python3 notebooks/generar_notebook_juez.py venas
"""
import base64
import io
import os
import sys
import tarfile

import nbformat as nbf

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VISTA = sys.argv[1] if len(sys.argv) > 1 else "venas"
TITULOS = {"venas": "el juez angiografía (detector de venas)"}
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


md(f"""
# Tercer juez: {TITULOS.get(VISTA, VISTA)}

Decidido con jhonatan (25 oct 2026): el jurado ronda 5 + ronda 7 dio Kaggle 0,39 porque sus dos YOLO miran la foto
de forma distinta. Se prueban terceros jueces uno a uno. Este YOLO recibe la vista `{VISTA}` de `filamentos/yolo.py`.
Todo lo demás igual que las rondas 5 y 7: YOLO11m-seg a 1024 px, 586 imágenes de estudio, mismo refinador de la
ronda 3, mismo examen de 121.

1. Entrena YOLO (tope `HORAS`).
2. Examen: candidatos y nota por umbral, con aciertos, inventados y escapados. Compara con la ronda 5.
3. Test: `submission.csv` con el mejor umbral del examen. **No envía nada.**

Requisitos en Kaggle: GPU T4, **Internet On** y como *Input*: la competición y el notebook
`filamentos-detector-ronda-2` Version 4 (`refinador.pt`).
""")

code(f"VISTA = {VISTA!r}  # forma de ver la foto (filamentos/yolo.py, VISTA)")
code("""
# Configuración. PRUEBA_RAPIDA=True reduce todo para comprobar que el notebook funciona.
PRUEBA_RAPIDA = False

RES = 1024
MODELO = 'yolo11m-seg.pt'
HORAS = 2.5                # mismo tope que la ronda 5
EPOCAS = 150
PACIENCIA = 25
LOTE = 6
CONF_MIN = 0.1
UMBRAL_REF = 0.5
UMBRALES = (0.15, 0.2, 0.25, 0.3, 0.35, 0.4)
REF_R5 = dict(PQ=0.4560, umbral=0.25, aciertos=878, inventados=381, escapados=489)  # ronda 5, mismo examen
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

md("## Código del proyecto\\nEmpaquetado del repositorio `filamentos-solares-2026` (los ficheros de `filamentos/`).")
code(f"""
import base64, io, os, tarfile
tarfile.open(fileobj=io.BytesIO(base64.b64decode('{empaquetar()}')), mode='r:gz').extractall('.')
print(sorted(os.listdir('filamentos')))
""")

md("## Datos, refinador y separación estudio / examen\\nMisma separación que las rondas 1-6.")
code("""
import csv, glob, hashlib, json, shutil, time
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
assert PESOS_REF, ('No encuentro refinador.pt: añade como Input el notebook filamentos-detector-ronda-2 (Version 4). '
                   'Hay en /kaggle/input: ' + str(glob.glob('/kaggle/input/*/*')))
print('Datos en', DATOS, '| refinador:', PESOS_REF)

ANOT = cargar_anotaciones(DATOS)
nombres = sorted(n for n in ANOT if os.path.exists(f'{DATOS}/train/train_images/{n}'))

def es_val(n):
    return int(hashlib.md5(n.encode()).hexdigest(), 16) % 100 < 15
val = [n for n in nombres if es_val(n)]
tr = [n for n in nombres if not es_val(n)]
TEST = os.environ.get('TEST_DIR', f'{DATOS}/test/test_images')
test = sorted(os.listdir(TEST)) if os.path.isdir(TEST) else []
if PRUEBA_RAPIDA:
    tr, val, test = tr[:6], val[:3], test[:2]
    RES, MODELO, EPOCAS, HORAS, LOTE = 256, 'yolo11n-seg.pt', 1, 0.05, 4
print(len(tr), 'imágenes de estudio,', len(val), 'de examen,', len(test), 'de test')

ref = smp.Unet('resnet18', encoder_weights=None, in_channels=3, classes=1).to(disp)
ref.load_state_dict(torch.load(PESOS_REF, map_location=disp))
reales_val = {n: Y.reales_rle(ANOT[n]) for n in val}
""")

md("## La vista de una imagen\\nPara comprobar a ojo lo que ve YOLO: sus tres canales, uno al lado del otro.")
code("""
muestra = np.asarray(Image.open(f'{DATOS}/train/train_images/{val[0]}').convert('L').resize((RES, RES), Image.BILINEAR))
v = Y.VISTA[VISTA](muestra)
Image.fromarray(np.concatenate([v[..., i] for i in range(3)], 1)).save('vista.png')
from IPython.display import Image as Ver, display
display(Ver('vista.png', width=1200))
""")

md("## 1. Entrenar YOLO con esta vista\\nMismos ajustes que las rondas 5 y 7.")
code("""
t = time.time()
yaml = Y.preparar_datos(DATOS, ANOT, tr, val, '/tmp/yolo_datos', res=RES, vistas=VISTA)
print(f'Datos YOLO listos en {time.time() - t:.0f} s')
modelo = YOLO(MODELO)
modelo.train(data=yaml, imgsz=RES, epochs=EPOCAS, time=HORAS, patience=PACIENCIA, batch=LOTE, single_cls=True,
             fliplr=0.5, flipud=0.5, degrees=0, hsv_h=0, hsv_s=0, hsv_v=0.3, mosaic=1.0, close_mosaic=10,
             overlap_mask=False, cos_lr=True, project='/tmp/yolo_runs', name='juez', exist_ok=True,
             plots=False, workers=2, device=0 if disp == 'cuda' else 'cpu', verbose=False)
shutil.copy('/tmp/yolo_runs/juez/weights/best.pt', f'yolo_{VISTA}.pt')
shutil.copy('/tmp/yolo_runs/juez/results.csv', 'yolo_historial.csv')
print(pd.read_csv('yolo_historial.csv').iloc[:, :12].tail(8).to_string(index=False))
print(f'Entrenamiento: {(time.time() - t) / 60:.0f} min')
""")

md("## 2. Examen\\nYOLO en modo generoso + refinador de la ronda 3; se guardan todos los candidatos para analizar sin GPU.")
code("""
det = Y.detector(YOLO(f'yolo_{VISTA}.pt'), RES, CONF_MIN, vistas=VISTA)
t = time.time()
cand_val = {}
for i, n in enumerate(val):
    grande = np.asarray(Image.open(f'{DATOS}/train/train_images/{n}').convert('L'), np.uint8)
    cand_val[n], _, _ = C.procesar(None, ref, grande, disp, RES, CONF_MIN, UMBRAL_REF, 0, detectar=det)
    if i % 20 == 0:
        print(f'{i + 1}/{len(val)} imágenes, {time.time() - t:.0f} s', flush=True)
with open(f'candidatos_{VISTA}_examen.json', 'w') as fh:
    json.dump(cand_val, fh, separators=(',', ':'))

pd.set_option('display.width', 200)
for clave, nombre in (('rle', 'YOLO solo'), ('rle_ref', 'YOLO + refinador')):
    filas = []
    for u in UMBRALES:
        r = Y.nota_umbral(cand_val, reales_val, u, clave, por_tamano=True)
        filas.append(dict(umbral=u, PQ=round(r['pq'], 4), SQ=round(r['sq'], 3), aciertos=r['tp'], inventados=r['fp'],
                          escapados=r['fn'], esc_menos_1k=r['escapados_menos_1k'], esc_1k_3k=r['escapados_1k_3k'],
                          esc_mas_3k=r['escapados_mas_3k']))
    tabla = pd.DataFrame(filas)
    print(f'\\n=== Examen, {nombre}, vista {VISTA} ===')
    print(tabla.to_string(index=False))
mejor = tabla.loc[tabla.PQ.idxmax()]
UMBRAL = float(mejor.umbral)
print(f"\\nRonda 5 (una vista): PQ {REF_R5['PQ']} umbral {REF_R5['umbral']} | aciertos {REF_R5['aciertos']} "
      f"inventados {REF_R5['inventados']} escapados {REF_R5['escapados']}")
print(f'Juez {VISTA}: PQ {mejor.PQ} umbral {UMBRAL} | aciertos {int(mejor.aciertos)} '
      f'inventados {int(mejor.inventados)} escapados {int(mejor.escapados)}')
""")

md("## 3. Test y `submission.csv`\\nCon el mejor umbral del examen. El archivo se genera pero **no se envía**.")
code("""
t = time.time()
filas, cand_test = [], {}
for i, f in enumerate(test):
    grande = np.asarray(Image.open(f'{TEST}/{f}').convert('L'), np.uint8)
    cand_test[f], _, _ = C.procesar(None, ref, grande, disp, RES, CONF_MIN, UMBRAL_REF, 0, detectar=det)
    img_id = os.path.splitext(f)[0]
    for k, r in enumerate(Y.seleccion(cand_test[f], UMBRAL), start=1):
        filas.append((f'{img_id}_{k}', r['counts'].decode('ascii') if isinstance(r['counts'], bytes) else r['counts']))
    if i % 30 == 0:
        print(f'{i + 1}/{len(test)} imágenes, {len(filas)} filamentos, {time.time() - t:.0f} s', flush=True)
with open(f'candidatos_{VISTA}_test.json', 'w') as fh:
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
salida = os.path.join(RAIZ, "notebooks", f"juez_{VISTA}.ipynb")
nbf.write(nb, salida)
print("Notebook generado:", salida, os.path.getsize(salida), "bytes")
