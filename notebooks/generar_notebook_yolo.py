"""Genera notebooks/entrenar_yolo.ipynb (ronda 5: detector YOLO11-seg) incrustando el código de filamentos/.

Ejecutar desde la raíz del repositorio:
    python3 notebooks/generar_notebook_yolo.py
"""
import os

import nbformat as nbf

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODULOS = ["__init__.py", "datos.py", "metrica.py", "clasico.py", "observador.py", "detector.py", "candidatos.py", "yolo.py"]

celdas = []


def md(texto):
    celdas.append(nbf.v4.new_markdown_cell(texto.strip()))


def code(texto):
    celdas.append(nbf.v4.new_code_cell(texto.strip()))


md("""
# Ronda 5: detector YOLO11-seg

Decidido con jhonatan el 21 oct 2026 (plan en `docs/experimentos.md`, "Ronda 5"). Cambia **solo el detector**:
YOLO11-seg (Ultralytics, 2024) en lugar de Mask R-CNN, a 1024 px, entrenado con las 586 imágenes de estudio
(cada experto como ejemplo aparte). El refinador es el de la ronda 3, sin reentrenar.

1. Entrena YOLO hasta que deje de mejorar (tope de tiempo `HORAS`).
2. Pasa YOLO en modo generoso (confianza desde 0,1) por las 121 imágenes del examen y guarda cada candidato con su
   silueta (detector y refinador) y su ficha, igual que el paso 1 del tribunal: así la votación con la ronda 3 y
   el juez se ajustan después sin GPU.
3. Imprime la nota del examen por umbral de confianza, con aciertos, inventados y escapados (por tamaño).

No genera `submission.csv`. Requisitos en Kaggle: *Accelerator* = GPU T4, **Internet On** (instala `ultralytics`
y baja los pesos de partida) y como *Input*: la competición y el notebook `filamentos-detector-ronda-2`
Version 4 (de ahí sale `refinador.pt`).
""")

code("""
# Configuración. PRUEBA_RAPIDA=True reduce todo para comprobar que el notebook funciona.
PRUEBA_RAPIDA = False

RES = 1024                 # resolución del detector (la de la ronda 3)
MODELO = 'yolo11m-seg.pt'  # tamaño m: buen equilibrio entre fuerza y memoria de la T4
HORAS = 2.5                # tope de entrenamiento (YOLO para antes si deja de mejorar)
EPOCAS = 150
PACIENCIA = 25             # vueltas sin mejorar antes de parar
LOTE = 6
CONF_MIN = 0.1             # guarda todos los candidatos desde esta confianza
UMBRAL_REF = 0.5           # umbral del refinador de la ronda 3
UMBRALES = (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)
CON_ESTUDIO = False        # candidatos también de las imágenes de estudio (tarda ~1 h más)
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

md("## Código del proyecto\nCopiado del repositorio `filamentos-solares-2026`.")
code("import os\nos.makedirs('filamentos', exist_ok=True)")
for nombre in MODULOS:
    with open(os.path.join(RAIZ, "filamentos", nombre)) as f:
        fuente = f.read()
    code(f"%%writefile filamentos/{nombre}\n{fuente or '# paquete del proyecto'}")

md("## Datos, separación estudio / examen y refinador de la ronda 3\nMisma separación que las rondas 1-4.")
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
    tr, val = tr[:6], val[:3]
    RES, MODELO, EPOCAS, HORAS, LOTE = 256, 'yolo11n-seg.pt', 1, 0.05, 4
print(len(tr), 'imágenes de estudio,', len(val), 'de examen')

ref = smp.Unet('resnet18', encoder_weights=None, in_channels=3, classes=1).to(disp)
if PESOS_REF:
    ref.load_state_dict(torch.load(PESOS_REF, map_location=disp))
""")

md("## Entrenar YOLO11-seg\nCada imagen se repite una vez por experto, con sus dibujos. YOLO valida cada vuelta con el examen y guarda la mejor.")
code("""
t = time.time()
yaml = Y.preparar_datos(DATOS, ANOT, tr, val, '/tmp/yolo_datos', res=RES)
print(f'Datos YOLO listos en {time.time() - t:.0f} s')
modelo = YOLO(MODELO)
modelo.train(data=yaml, imgsz=RES, epochs=EPOCAS, time=HORAS, patience=PACIENCIA, batch=LOTE, single_cls=True,
             fliplr=0.5, flipud=0.5, degrees=0, hsv_h=0, hsv_s=0, hsv_v=0.3, mosaic=1.0, close_mosaic=10,
             overlap_mask=False, cos_lr=True, project='/tmp/yolo_runs', name='r5', exist_ok=True,
             plots=False, workers=2, device=0 if disp == 'cuda' else 'cpu', verbose=False)
shutil.copy('/tmp/yolo_runs/r5/weights/best.pt', 'yolo_mejor.pt')
shutil.copy('/tmp/yolo_runs/r5/results.csv', 'yolo_historial.csv')
print(pd.read_csv('yolo_historial.csv').iloc[:, :12].tail(8).to_string(index=False))
print(f'Entrenamiento: {(time.time() - t) / 60:.0f} min')
""")

md("## Candidatos del examen\nYOLO en modo generoso; cada candidato se redibuja con el refinador de la ronda 3 por separado.")
code("""
mejor = YOLO('yolo_mejor.pt')
det = Y.detector(mejor, RES, CONF_MIN)
cand_val, reales_val = {}, {}
t = time.time()
for i, n in enumerate(val):
    grande = np.asarray(Image.open(f'{DATOS}/train/train_images/{n}').convert('L'), np.uint8)
    cand_val[n], _, _ = C.procesar(None, ref, grande, disp, RES, CONF_MIN, UMBRAL_REF, 0, detectar=det)
    reales_val[n] = Y.reales_rle(ANOT[n])
    if i % 20 == 0:
        print(f'{i + 1}/{len(val)} imágenes, {sum(map(len, cand_val.values()))} candidatos, {time.time() - t:.0f} s', flush=True)
with open('candidatos_yolo_examen.json', 'w') as fh:
    json.dump(cand_val, fh, separators=(',', ':'))
print(f'{sum(map(len, cand_val.values()))} candidatos en el examen ({time.time() - t:.0f} s)')
""")

md("""
## La nota del examen
Juez de la ronda 3 (confianza >= umbral y unir los que se tocan) con cada umbral. Referencia de la ronda 3:
**PQ 0,4405 = TP 820, FP 350, FN 547** (escapados: ~300 de menos de 1.000 px, 166 de 1.000-3.000, 80 de más).
""")
code("""
pd.set_option('display.width', 200)
for clave, nombre in (('rle', 'YOLO solo'), ('rle_ref', 'YOLO + refinador')):
    filas = []
    for u in UMBRALES:
        r = Y.nota_umbral(cand_val, reales_val, u, clave, por_tamano=True)
        filas.append(dict(umbral=u, PQ=round(r['pq'], 4), SQ=round(r['sq'], 3), aciertos=r['tp'], inventados=r['fp'],
                          escapados=r['fn'], esc_menos_1k=r['escapados_menos_1k'], esc_1k_3k=r['escapados_1k_3k'],
                          esc_mas_3k=r['escapados_mas_3k']))
    print(f'\\n=== Examen, {nombre} ===')
    print(pd.DataFrame(filas).to_string(index=False))
filas, n_reales, alcanzables, n_dup = C.tabla_franjas(cand_val, reales_val, 'rle_ref')
print('\\n=== Acierto por franja de confianza (con refinador) ===')
print(pd.DataFrame(filas).to_string(index=False))
print(f'Filamentos reales: {n_reales} | al alcance de algún candidato: {alcanzables} ({100 * alcanzables / n_reales:.0f} %) '
      f'| ronda 3: 1139 al alcance, 820 acertados')
""")

md("## Candidatos de las imágenes de estudio (opcional)")
code("""
if CON_ESTUDIO:
    cand_tr, t = {}, time.time()
    for i, n in enumerate(tr):
        grande = np.asarray(Image.open(f'{DATOS}/train/train_images/{n}').convert('L'), np.uint8)
        cand_tr[n], _, _ = C.procesar(None, ref, grande, disp, RES, CONF_MIN, UMBRAL_REF, 0, detectar=det)
        if i % 100 == 0:
            print(f'{i + 1}/{len(tr)} imágenes, {time.time() - t:.0f} s', flush=True)
    with open('candidatos_yolo_estudio.json', 'w') as fh:
        json.dump(cand_tr, fh, separators=(',', ':'))
print(f'Tiempo total: {(time.time() - T0) / 60:.0f} min')
""")

nb = nbf.v4.new_notebook()
nb["cells"] = celdas
nb["metadata"] = {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                  "language_info": {"name": "python"},
                  "accelerator": "GPU"}
salida = os.path.join(RAIZ, "notebooks", "entrenar_yolo.ipynb")
nbf.write(nb, salida)
print("Notebook generado:", salida)
