"""Genera data.js para el visor a partir de las anotaciones COCO de MAGFiLO.

Uso (desde la carpeta visor/):
    python3 build_visor.py ../MAGFiLO_1.0_Kaggle_2026
"""
import json
import os
import sys
from collections import defaultdict

CATEGORIAS = {1: "Left", 2: "Right", 3: "Unidentifiable", 4: "Ambiguous"}


def redondear(coords):
    return [round(c, 1) for c in coords]


def main(raiz):
    ann_path = os.path.join(raiz, "train", "MAGFiLO_1.0_Annotations_kaggle2026_train.json")
    with open(ann_path) as f:
        coco = json.load(f)

    anns_por_registro = defaultdict(list)
    for a in coco["annotations"]:
        anns_por_registro[a["image_id"]].append(a)

    # Una misma imagen puede tener varios registros: uno por anotador.
    # El id es "GGGGAA-<nombre>", donde AA identifica al anotador dentro del grupo.
    imagenes = defaultdict(list)
    for reg in coco["images"]:
        nombre = reg["file_name"]
        filamentos = []
        for a in anns_por_registro[reg["id"]]:
            filamentos.append({
                "c": a["category_id"],
                "a": round(a["area"]),
                "p": [redondear(p) for p in a["segmentation"]],
                "s": redondear(a.get("spine") or []),
                "b": redondear(a["bbox"]),
            })
        imagenes[nombre].append({"id": reg["id"].split("-")[0], "f": filamentos})

    train = []
    for nombre in sorted(imagenes):
        regs = sorted(imagenes[nombre], key=lambda r: r["id"])
        train.append({"n": nombre, "r": regs})

    test_dir = os.path.join(raiz, "test", "test_images")
    test = sorted(x for x in os.listdir(test_dir) if x.lower().endswith((".jpeg", ".jpg")))

    data = {
        "categorias": CATEGORIAS,
        "train_dir": os.path.relpath(os.path.join(raiz, "train", "train_images")),
        "test_dir": os.path.relpath(test_dir),
        "train": train,
        "test": test,
    }
    with open("data.js", "w") as f:
        f.write("window.DATA = ")
        json.dump(data, f, separators=(",", ":"))
        f.write(";\n")

    n_fil = sum(len(r["f"]) for img in train for r in img["r"])
    print(f"data.js listo: {len(train)} imágenes train, {len(test)} test, {n_fil} filamentos anotados")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "../MAGFiLO_1.0_Kaggle_2026")
