"""Genera notebooks/entrenar_especialista.ipynb (ronda 4) incrustando el código de filamentos/.

Kaggle no puede leer nuestro repositorio privado, así que el notebook lleva el
código dentro (celdas %%writefile). Ejecutar desde la raíz del repositorio:
    python3 notebooks/generar_notebook_especialista.py
"""
import os

import nbformat as nbf

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODULOS = ["__init__.py", "datos.py", "metrica.py", "clasico.py", "observador.py", "detector.py", "especialista.py"]

celdas = []


def md(texto):
    celdas.append(nbf.v4.new_markdown_cell(texto.strip()))


def code(texto):
    celdas.append(nbf.v4.new_code_cell(texto.strip()))


md("""
# Ronda 4: detector especialista en filamentos pequeños

La ronda 3 mostró que el detector general (Sol entero a 1024 px) ya tocó techo: ve el 90 % de los filamentos
grandes, pero 467 de los 547 que se le escapan miden menos de 3000 px.

1. **Detector general**: el de la ronda 3, sin volver a entrenarlo (se cargan sus pesos).
2. **Detector especialista** (nuevo): mismo Mask R-CNN, pero mira la imagen original de **2048 px** por ventanas
   de 1024 (cada filamento se ve el doble de grande). Arranca de lo que aprendió el general y estudia con
   ventanas centradas casi siempre en filamentos pequeños, hasta la meseta como en la ronda 3.
3. **Dos jueces**: el del general (como en la ronda 3) y el del especialista, que solo deja pasar filamentos de
   menos de 3000 px que el general no haya encontrado ya (si se repiten, gana el general).
4. **Refinador**: el de la ronda 3, sin volver a entrenarlo, aplicado a todo lo que pasa los jueces.

Se mide por etapas en el mismo examen de 121 imágenes: general solo, general + especialista, y con refinador.
Genera `submission.csv` pero **no envía nada a la competición.**

Requisitos en Kaggle: *Accelerator* = GPU T4, *Internet* = activado, y como *Input*: la competición y el
*Output* del notebook `filamentos-detector-ronda-2` (versión de la ronda 3, con `detector_mejor.pt` y
`refinador.pt`).
""")

code("""
# Configuración. PRUEBA_RAPIDA=True reduce todo para comprobar que el notebook funciona.
PRUEBA_RAPIDA = False

RES = 1024            # resolución del detector general (la original es 2048)
# Especialista: estudia hasta que la nota se aplana (meseta), igual que el detector en la ronda 3
EPOCAS_MAX = 40       # tope de seguridad
LOTE_ESP = 2
LR_ESP = 0.005
P_PEQUENO = 0.8       # 8 de cada 10 ventanas de estudio se centran en un filamento pequeño
EXAMEN_CADA = 2
PACIENCIA = 3
MEJORA_MIN = 0.005
ACHIQUES = 1
HORAS_MAX_ESP = 4.5   # el general y el refinador ya vienen entrenados de la ronda 3
SEMILLA = 42

if PRUEBA_RAPIDA:
    EPOCAS_MAX, EXAMEN_CADA, PACIENCIA, MEJORA_MIN = 2, 1, 1, 1.0
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

md("## Datos, separación estudio / examen y pesos de la ronda 3\nMisma separación que las rondas 1-3.")
code("""
import glob, hashlib, json, time, random
import numpy as np, torch
from multiprocessing import Pool
from PIL import Image
from pycocotools import mask as mask_utils
from filamentos.datos import cargar_anotaciones, mascaras_de, mascara_a_rle, rle_a_mascara
from filamentos.metrica import Resultado, evaluar_imagen, resumen
from filamentos import clasico, detector as D, especialista as E, observador as O

random.seed(SEMILLA); np.random.seed(SEMILLA); torch.manual_seed(SEMILLA)
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
                                     'notebook filamentos-detector-ronda-2')
print('Datos en', DATOS, '| pesos:', PESOS_DET, PESOS_REF)

ANOT = cargar_anotaciones(DATOS)
nombres = sorted(n for n in ANOT if os.path.exists(f'{DATOS}/train/train_images/{n}'))

def es_val(n):
    return int(hashlib.md5(n.encode()).hexdigest(), 16) % 100 < 15
val = [n for n in nombres if es_val(n)]
tr = [n for n in nombres if not es_val(n)]
if PRUEBA_RAPIDA:
    tr, val = tr[:8], val[:3]
print(len(tr), 'imágenes de estudio,', len(val), 'de examen')
""")

code("""
def radio(img):
    \"\"\"Distancia al centro del disco solar, en radios, para cada píxel.\"\"\"
    cx, cy, R = clasico.disco_solar(img.astype(np.float32))
    yy, xx = np.indices(img.shape)
    return (np.hypot(xx - cx, yy - cy) / R).astype(np.float32)

def gris2048(n, carpeta='train/train_images'):
    return np.asarray(Image.open(f'{DATOS}/{carpeta}/{n}').convert('L'), np.uint8)

def cargar(n):
    grande = gris2048(n)
    img = np.asarray(Image.fromarray(grande).resize((RES, RES), Image.BILINEAR), np.uint8)
    return grande, img, radio(img)

def radio2048(r):
    return np.asarray(Image.fromarray(r).resize((2048, 2048), Image.BILINEAR))

t = time.time()
with Pool(os.cpu_count()) as p:
    cache = dict(zip(tr + val, p.map(cargar, tr + val)))
ejemplos_esp = [(cache[n][0], anns) for n in tr for anns in ANOT[n].values()]
print(f'{len(ejemplos_esp)} ejemplos de estudio (imagen x experto), cargados en {time.time() - t:.0f} s')

# Filamentos de los expertos en el examen, a 2048 y comprimidos (RLE) para no llenar la memoria
reales_val = {n: [[D.a_rle(m) for m in mascaras_de(anns)] for anns in ANOT[n].values()] for n in val}

def pq_examen(pred_por_imagen, nombres=None):
    \"\"\"pred_por_imagen: {imagen: [máscaras 2048]}. Compara contra cada experto, como en las rondas 1-3.\"\"\"
    tot = Resultado()
    for n in (nombres or val):
        pred = pred_por_imagen[n]
        for rles in reales_val[n]:
            tot += evaluar_imagen([mask_utils.decode(r).astype(bool) for r in rles], pred)
    return tot
""")

md("## Etapa 1: detector general de la ronda 3 y su juez\nDebe dar lo mismo que en la ronda 3 (0,389 detector solo).")
code("""
gen = D.construir_detector(False, RES).to(disp)
if PESOS_DET:
    gen.load_state_dict(torch.load(PESOS_DET, map_location=disp))

crudas_gen = {}
for n in val:
    with torch.autocast(disp, enabled=disp == 'cuda'):
        m, c = D.detectar(gen, cache[n][1], disp)
    keep = c >= 0.2
    crudas_gen[n] = ([D.a_rle(x) for x in m[keep]], c[keep])

def juez_gen(n, **post):
    rles, conf = crudas_gen[n]
    m = np.stack([mask_utils.decode(r).astype(bool) for r in rles]) if rles else np.zeros((0, RES, RES), bool)
    return D.instancias(m, conf, cache[n][2], **post)

resultados = []
for umbral in (0.6, 0.7, 0.8, 0.85, 0.9):
    for fusionar in (True, False):
        for area_min in (0, 330):
            post = dict(umbral=umbral, fusionar=fusionar, area_min=area_min)
            res = pq_examen({n: juez_gen(n, **post) for n in val})
            resultados.append((res.pq, post))
PQ_GEN, POST_GEN = max(resultados, key=lambda t: t[0])
pred_gen = {n: juez_gen(n, **POST_GEN) for n in val}
RES_GEN = pq_examen(pred_gen)
print('Juez del general:', POST_GEN, '->', resumen(RES_GEN))
""")

md("""
## Etapa 2: entrenamiento del especialista
Arranca con los pesos del general. En el examen se mide la nota del conjunto: general + especialista
(confianza 0,6), así la meseta se declara cuando el especialista deja de aportar.
""")
code("""
esp = D.construir_detector(False, E.VENTANA).to(disp)
if PESOS_DET:
    esp.load_state_dict(torch.load(PESOS_DET, map_location=disp))
params = [p for p in esp.parameters() if p.requires_grad]
opt = torch.optim.SGD(params, lr=LR_ESP, momentum=0.9, weight_decay=1e-4)
escalador = torch.amp.GradScaler(enabled=disp == 'cuda')
dl = torch.utils.data.DataLoader(E.DatosEspecialista(ejemplos_esp, aumentar=True, p_pequeno=P_PEQUENO),
                                 batch_size=LOTE_ESP, shuffle=True, num_workers=2, collate_fn=D.juntar)
r2048_val = {n: radio2048(cache[n][2]) for n in val}

def detectar_examen_esp(modelo):
    return {n: E.detectar(modelo, cache[n][0], disp) for n in val}

def juez_esp(crudas, n, **post):
    rles, conf = crudas[n]
    return E.juez(rles, conf, r2048_val[n], pred_gen[n], **post)

mejor, historia = -1, []
sin_mejora, achiques, t_esp = 0, 0, time.time()
for epoca in range(1, EPOCAS_MAX + 1):
    esp.train(); t = time.time(); suma = 0
    for imgs, objs in dl:
        imgs = [i.to(disp) for i in imgs]
        objs = [{k: v.to(disp) for k, v in o.items()} for o in objs]
        with torch.autocast(disp, enabled=disp == 'cuda'):
            l = sum(esp(imgs, objs).values())
        opt.zero_grad(); escalador.scale(l).backward(); escalador.step(opt); escalador.update()
        suma += l.item()
    linea = f'vuelta {epoca}: pérdida {suma / len(dl):.4f} ({time.time() - t:.0f} s, paso {opt.param_groups[0]["lr"]:g})'
    parar = False
    if epoca % EXAMEN_CADA == 0:
        crudas_esp = detectar_examen_esp(esp)
        res = pq_examen({n: pred_gen[n] + juez_esp(crudas_esp, n, umbral=0.6) for n in val})
        historia.append((epoca, round(res.pq, 4)))
        linea += ' | ' + resumen(res)
        if res.pq > mejor:
            if res.pq >= mejor + MEJORA_MIN:
                sin_mejora = 0
            else:
                sin_mejora += 1  # mejora demasiado pequeña: se guarda, pero cuenta para la meseta
            mejor = res.pq
            torch.save(esp.state_dict(), 'especialista_mejor.pt')
            linea += ' | guardado'
        else:
            sin_mejora += 1
        linea += f' | sin mejora {sin_mejora}/{PACIENCIA}'
        if sin_mejora >= PACIENCIA:
            if achiques < ACHIQUES:
                achiques += 1; sin_mejora = 0
                for g in opt.param_groups:
                    g['lr'] *= 0.1
                linea += ' | MESETA: achico el paso x0,1'
            else:
                linea += ' | MESETA otra vez: paro'
                parar = True
    if time.time() - t_esp > HORAS_MAX_ESP * 3600:
        linea += ' | tope de tiempo: paro'
        parar = True
    print(linea, flush=True)
    if parar:
        break
if not os.path.exists('especialista_mejor.pt'):
    torch.save(esp.state_dict(), 'especialista_mejor.pt')
print('Curva (vuelta, PQ general + especialista):', historia)
print(f'Entrenamiento del especialista: {epoca} vueltas en {(time.time() - t_esp) / 60:.0f} min')
print('General solo:', round(RES_GEN.pq, 4), '| mejor con especialista (confianza 0,6):', round(mejor, 4))
""")

md("## Etapa 3: juez del especialista\nConfianza mínima, unir o no sus detecciones que se pisan, y tamaño mínimo.")
code("""
esp.load_state_dict(torch.load('especialista_mejor.pt', map_location=disp))
crudas_esp = detectar_examen_esp(esp)
PQ_ESP, POST_ESP = RES_GEN.pq, None
for umbral in (0.5, 0.6, 0.7, 0.8, 0.9):
    for fusionar in (True, False):
        for area_min in (0, 330):
            post = dict(umbral=umbral, fusionar=fusionar, area_min=area_min)
            res = pq_examen({n: pred_gen[n] + juez_esp(crudas_esp, n, **post) for n in val})
            print(post, '->', resumen(res), flush=True)
            if res.pq > PQ_ESP:
                PQ_ESP, POST_ESP = res.pq, post
if POST_ESP is None:
    print('El especialista no sube la nota con ningún juez: se queda solo el general')
    extra = {n: [] for n in val}
else:
    extra = {n: juez_esp(crudas_esp, n, **POST_ESP) for n in val}
pred_det = {n: pred_gen[n] + extra[n] for n in val}
RES_DET = pq_examen(pred_det)
print('Juez del especialista:', POST_ESP)
print('General solo:        ', resumen(RES_GEN))
print('General + especialista:', resumen(RES_DET))
print(f'El especialista añade {sum(map(len, extra.values()))} filamentos: aciertos {RES_DET.tp - RES_GEN.tp:+d}, '
      f'inventados {RES_DET.fp - RES_GEN.fp:+d}, se escapan {RES_DET.fn - RES_GEN.fn:+d}')
""")

md("## Etapa 4: refinador de la ronda 3\nSe prueba sobre el general solo (como la ronda 3) y sobre general + especialista.")
code("""
ref = smp.Unet('resnet18', encoder_weights=None, in_channels=3, classes=1).to(disp)
if PESOS_REF:
    ref.load_state_dict(torch.load(PESOS_REF, map_location=disp))

def refinar_todo(pred, umbral):
    out = {}
    for n in val:
        with torch.autocast(disp, enabled=disp == 'cuda'):
            out[n] = D.refinar(ref, cache[n][0], pred[n], disp, umbral=umbral)
    return out

candidatos = [('general solo', None, pred_gen, RES_GEN), ('general + especialista', None, pred_det, RES_DET)]
for umbral_ref in (0.3, 0.4, 0.5):
    for nombre, pred in (('general solo', pred_gen), ('general + especialista', pred_det)):
        if nombre == 'general + especialista' and POST_ESP is None:
            continue
        p = refinar_todo(pred, umbral_ref)
        res = pq_examen(p)
        print(f'{nombre} + refinador (umbral {umbral_ref}):', resumen(res), flush=True)
        candidatos.append((nombre, umbral_ref, p, res))
NOMBRE_FINAL, UMBRAL_REF, pred_final, RES_FINAL = max(candidatos, key=lambda t: t[3].pq)
USAR_ESP = NOMBRE_FINAL == 'general + especialista'
print(f'Elegido: {NOMBRE_FINAL}' + (f' + refinador (umbral {UMBRAL_REF})' if UMBRAL_REF else '') + f' -> PQ {RES_FINAL.pq:.4f}')
print('Ronda 3 para comparar: 0,4405 (TP 820, FP 350, FN 547)')
""")

md("## Predicciones del examen para el visor")
code("""
from filamentos.datos import mascara_a_poligonos
examen = {}
for n in val:
    pred = pred_final[n]
    res = pq_examen(pred_final, [n])
    examen[n] = {'pq': round(res.pq, 3), 'tp': res.tp, 'fp': res.fp, 'fn': res.fn,
                 'f': [{'a': int(m.sum()), 'p': mascara_a_poligonos(m)} for m in pred]}
with open('predicciones_examen.json', 'w') as fh:
    json.dump({'post': POST_GEN, 'especialista': POST_ESP if USAR_ESP else None, 'refinador': UMBRAL_REF,
               'imagenes': examen}, fh, separators=(',', ':'))
peores = sorted(examen.items(), key=lambda t: t[1]['pq'])[:10]
print('Las 10 imágenes del examen donde peor lo hace:')
for n, e in peores:
    print(f"  {n}: PQ {e['pq']:.2f}  (aciertos {e['tp']}, inventados {e['fp']}, se le escapan {e['fn']})")
""")

md("""
## Informe de errores
Igual que en la ronda 3: tamaño, zona, oscuridad y telescopio; **casi** = lo vio pero lo partió, fusionó o
dibujó mal (mejor IoU 0,1-0,5); **puro** = inventado de la nada o no visto en absoluto (mejor IoU < 0,1).
""")
code("""
import pandas as pd
from filamentos.metrica import detalle_errores
TELESCOPIO = {'B': 'Big Bear', 'M': 'Mauna Loa', 'L': 'Learmonth', 'U': 'Udaipur', 'T': 'Teide', 'C': 'Cerro Tololo'}
filas = []
for n in val:
    pred = pred_final[n]
    x, r = O.preparar_imagen(f'{DATOS}/train/train_images/{n}', 1024)
    r_g = np.asarray(Image.fromarray(r).resize((2048, 2048), Image.BILINEAR))
    contraste = np.asarray(Image.fromarray(x[1]).resize((2048, 2048), Image.BILINEAR)) * 1.5
    tel = TELESCOPIO.get(os.path.splitext(n)[0][-2], '?')
    for anot, rles in zip(ANOT[n], reales_val[n]):
        reales = [mask_utils.decode(r).astype(bool) for r in rles]
        d_r, d_p = detalle_errores(reales, pred)
        for lado, mascaras, dets in (('real', reales, d_r), ('red', pred, d_p)):
            for m, (ok, iou) in zip(mascaras, dets):
                filas.append(dict(imagen=n, anotador=anot, lado=lado, acierto=ok, iou=iou, area=int(m.sum()),
                                  radio=float(r_g[m].mean()), contraste=float(contraste[m].mean()), telescopio=tel))
Err = pd.DataFrame(filas)
Err['error'] = np.where(Err.acierto, 'acierto', np.where(Err.iou >= 0.1, 'casi', 'puro'))
Err['tamaño'] = pd.cut(Err.area, [0, 330, 1000, 3000, 10000, 1e9], labels=['<330', '330-1k', '1k-3k', '3k-10k', '>10k'])
Err['zona'] = pd.cut(Err.radio, [0, 0.5, 0.8, 0.9, 2], labels=['centro <0,5', '0,5-0,8', '0,8-0,9', 'borde >0,9'])
Err['oscuridad'] = pd.cut(Err.contraste, [0, 0.8, 0.9, 0.95, 9], labels=['muy oscuro <0,8', '0,8-0,9', '0,9-0,95', 'casi fondo >0,95'])
Err.to_csv('errores_examen.csv', index=False)

def tabla(col):
    t = pd.crosstab([Err.lado, Err[col]], Err.error)
    t['% acierto'] = (100 * t.get('acierto', 0) / t.sum(axis=1)).round(0)
    return t

pd.set_option('display.width', 200)
for col in ('tamaño', 'zona', 'oscuridad', 'telescopio'):
    print(f'\\n=== Por {col} (real = filamentos de los expertos; red = lo que dibujó el modelo) ===')
    print(tabla(col))
""")

md("## Predicción del test y `submission.csv`\nEl archivo se genera pero **no se envía**.")
code("""
import csv
test_dir = f'{DATOS}/test/test_images'
filas = []
for f in (sorted(os.listdir(test_dir)) if os.path.isdir(test_dir) else []):
    grande = gris2048(f, 'test/test_images')
    img = np.asarray(Image.fromarray(grande).resize((RES, RES), Image.BILINEAR), np.uint8)
    rr = radio(img)
    with torch.autocast(disp, enabled=disp == 'cuda'):
        m, c = D.detectar(gen, img, disp)
    pred = D.instancias(m, c, rr, **POST_GEN)
    if USAR_ESP:
        rles, conf = E.detectar(esp, grande, disp)
        pred = pred + E.juez(rles, conf, radio2048(rr), pred, **POST_ESP)
    if UMBRAL_REF:
        with torch.autocast(disp, enabled=disp == 'cuda'):
            pred = D.refinar(ref, grande, pred, disp, umbral=UMBRAL_REF)
    img_id = os.path.splitext(f)[0]
    for k, mk in enumerate(pred, start=1):
        filas.append((f'{img_id}_{k}', mascara_a_rle(mk)))
    if PRUEBA_RAPIDA and len(filas) > 20:
        break
with open('submission.csv', 'w', newline='') as fh:
    w = csv.writer(fh)
    w.writerow(['filament_id', 'segmentation_rle'])
    w.writerows(filas)

ids = [a for a, _ in filas]
assert len(ids) == len(set(ids))
if filas:
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
salida = os.path.join(RAIZ, "notebooks", "entrenar_especialista.ipynb")
nbf.write(nb, salida)
print("Notebook generado:", salida)
