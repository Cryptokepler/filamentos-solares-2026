"""Genera notebooks/entrenar_observador.ipynb incrustando el código de filamentos/.

Kaggle no puede leer nuestro repositorio privado, así que el notebook lleva el
código dentro (celdas %%writefile). Ejecutar desde la raíz del repositorio:
    python3 notebooks/generar_notebook.py
"""
import os

import nbformat as nbf

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODULOS = ["__init__.py", "datos.py", "metrica.py", "clasico.py", "observador.py"]

celdas = []


def md(texto):
    celdas.append(nbf.v4.new_markdown_cell(texto.strip()))


def code(texto):
    celdas.append(nbf.v4.new_code_cell(texto.strip()))


md("""
# Observador: U-Net para filamentos solares

Entrena una red que predice, píxel a píxel, la probabilidad de filamento. Después la convierte en
filamentos individuales, la mide con la **PQ oficial** en una validación propia y genera
`submission.csv`. **Este notebook no envía nada a la competición.**

Requisitos en Kaggle: *Accelerator* = GPU T4 o P100, *Internet* = activado (para instalar la librería
y bajar el encoder preentrenado), y la competición añadida como *Input*.
""")

code("""
# Configuración. PRUEBA_RAPIDA=True reduce todo para comprobar que el notebook funciona.
PRUEBA_RAPIDA = False

RES = 1024            # resolución de trabajo (la original es 2048)
ENCODER = "resnet34"
PESOS = "imagenet"   # encoder preentrenado (necesita Internet en Kaggle)
EPOCAS = 40
LOTE = 4
LR = 3e-4
VALIDAR_CADA = 5
SEMILLA = 42

if PRUEBA_RAPIDA:
    RES, EPOCAS, LOTE, VALIDAR_CADA, PESOS = 256, 2, 2, 1, None
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

md("## Datos y separación entrenamiento / validación")
code("""
import glob, hashlib, json, time, random
import numpy as np, torch
from multiprocessing import Pool
from filamentos.datos import cargar_anotaciones, mascaras_de, mascara_a_rle, rle_a_mascara
from filamentos.metrica import Resultado, evaluar_imagen, resumen
from filamentos import observador as O

random.seed(SEMILLA); np.random.seed(SEMILLA); torch.manual_seed(SEMILLA)

json_path = glob.glob('/kaggle/input/**/MAGFiLO_1.0_Annotations_kaggle2026_train.json', recursive=True)
if not json_path:  # ejecución fuera de Kaggle
    json_path = glob.glob(os.environ.get('MAGFILO', '.') + '/**/MAGFiLO_1.0_Annotations_kaggle2026_train.json', recursive=True)
DATOS = os.path.dirname(os.path.dirname(json_path[0]))
print('Datos en', DATOS)

ANOT = cargar_anotaciones(DATOS)
nombres = sorted(n for n in ANOT if os.path.exists(f'{DATOS}/train/train_images/{n}'))

# Validación fija: ~15 % de las imágenes, elegidas por hash del nombre (reproducible)
def es_val(n):
    return int(hashlib.md5(n.encode()).hexdigest(), 16) % 100 < 15
val = [n for n in nombres if es_val(n)]
tr = [n for n in nombres if not es_val(n)]
if PRUEBA_RAPIDA:
    tr, val = tr[:16], val[:6]
print(len(tr), 'entrenamiento,', len(val), 'validación')
""")

code("""
def preparar(n):
    x, r = O.preparar_imagen(f'{DATOS}/train/train_images/{n}', RES)
    return x, r, O.objetivo_suave(ANOT[n], RES)

t = time.time()
with Pool(os.cpu_count()) as p:
    cache = dict(zip(tr + val, p.map(preparar, tr + val)))
print(f'Preprocesado en {time.time() - t:.0f} s')
""")

md("## Entrenamiento")
code("""
disp = 'cuda' if torch.cuda.is_available() else 'cpu'
modelo = smp.Unet(ENCODER, encoder_weights=PESOS, in_channels=3, classes=1).to(disp)
opt = torch.optim.AdamW(modelo.parameters(), lr=LR, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=EPOCAS * max(1, len(tr) // LOTE))
escalador = torch.amp.GradScaler(enabled=disp == 'cuda')

dl = torch.utils.data.DataLoader(
    O.DatosEntreno([cache[n][0] for n in tr], [cache[n][2] for n in tr], aumentar=True),
    batch_size=LOTE, shuffle=True, drop_last=True, num_workers=2)

def evaluar(modelo, nombres, tta=False, **post):
    modelo.eval()
    tot = Resultado()
    for n in nombres:
        x, r, _ = cache[n]
        with torch.autocast(disp, enabled=disp == 'cuda'):
            prob = O.predecir_prob(modelo, x, disp, tta=tta)
        pred = O.a_instancias(prob, r, **post)
        for anns in ANOT[n].values():
            tot += evaluar_imagen(mascaras_de(anns), pred)
    return tot

mejor, historia = -1, []
for epoca in range(1, EPOCAS + 1):
    modelo.train(); t = time.time(); suma = 0
    for x, y in dl:
        x, y = x.to(disp), y.to(disp)
        with torch.autocast(disp, enabled=disp == 'cuda'):
            l = O.perdida(modelo(x), y)
        opt.zero_grad(); escalador.scale(l).backward(); escalador.step(opt); escalador.update(); sched.step()
        suma += l.item()
    linea = f'época {epoca}: pérdida {suma / len(dl):.4f} ({time.time() - t:.0f} s)'
    if epoca % VALIDAR_CADA == 0 or epoca == EPOCAS:
        res = evaluar(modelo, val)
        historia.append((epoca, res.pq))
        linea += ' | ' + resumen(res)
        if res.pq > mejor:
            mejor = res.pq
            torch.save(modelo.state_dict(), 'observador_mejor.pt')
    print(linea)
print('Mejor PQ de validación:', round(mejor, 4))
""")

md("## Ajuste del juez en validación\nUmbral de probabilidad, tamaño mínimo y unión de trozos cercanos.")
code("""
modelo.load_state_dict(torch.load('observador_mejor.pt', map_location=disp))
probs = {}
for n in val:
    x, r, _ = cache[n]
    with torch.autocast(disp, enabled=disp == 'cuda'):
        probs[n] = O.predecir_prob(modelo, x, disp, tta=True)

def pq_con(**post):
    tot = Resultado()
    for n in val:
        pred = O.a_instancias(probs[n], cache[n][1], **post)
        for anns in ANOT[n].values():
            tot += evaluar_imagen(mascaras_de(anns), pred)
    return tot

resultados = []
for umbral in (0.3, 0.4, 0.5, 0.6):
    for area_min in (150, 330, 600):
        for union_px in (0, 8):
            res = pq_con(umbral=umbral, area_min=area_min, union_px=union_px)
            resultados.append((res.pq, dict(umbral=umbral, area_min=area_min, union_px=union_px)))
            print(f'{umbral} {area_min} {union_px} -> {resumen(res)}')
MEJOR_POST = max(resultados, key=lambda t: t[0])[1]
print('Mejor post-proceso:', MEJOR_POST, 'PQ', round(max(resultados, key=lambda t: t[0])[0], 4))
""")

md("## Predicción del test y `submission.csv`\nEl archivo se genera pero **no se envía**.")
code("""
import csv
test_dir = f'{DATOS}/test/test_images'
filas = []
for f in sorted(os.listdir(test_dir)):
    x, r = O.preparar_imagen(f'{test_dir}/{f}', RES)
    with torch.autocast(disp, enabled=disp == 'cuda'):
        prob = O.predecir_prob(modelo, x, disp, tta=True)
    img_id = os.path.splitext(f)[0]
    for k, m in enumerate(O.a_instancias(prob, r, **MEJOR_POST), start=1):
        filas.append((f'{img_id}_{k}', mascara_a_rle(m)))
    if PRUEBA_RAPIDA and len(filas) > 20:
        break
with open('submission.csv', 'w', newline='') as fh:
    w = csv.writer(fh)
    w.writerow(['filament_id', 'segmentation_rle'])
    w.writerows(filas)

# Comprobación del formato: ids únicos y RLE que se decodifica a 2048x2048
ids = [a for a, _ in filas]
assert len(ids) == len(set(ids))
if filas:
    assert rle_a_mascara(filas[0][1]).shape == (2048, 2048)
else:
    print('AVISO: el modelo no encontró ningún filamento en el test')
print(len(filas), 'filamentos en', len(set(i.rsplit('_', 1)[0] for i in ids)), 'imágenes -> submission.csv')
""")

nb = nbf.v4.new_notebook()
nb["cells"] = celdas
nb["metadata"] = {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                  "language_info": {"name": "python"}}
salida = os.path.join(RAIZ, "notebooks", "entrenar_observador.ipynb")
nbf.write(nb, salida)
print("Notebook generado:", salida)
