#!/usr/bin/env python3
"""
benchmark.py — Génération automatique des benchmarks et graphiques
pour le projet CUDA de filtres de convolution.

Usage :
  python3 benchmark.py --exe ./filters --img-dir ./img --out-dir ./results
"""

import subprocess
import re
import os
import argparse
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

matplotlib.rcParams['figure.dpi'] = 150
matplotlib.rcParams['font.size']  = 11

# ===========================================================================
# Configuration
# ===========================================================================

FILTERS      = ["BoxBlur", "Sobel", "Laplace", "Gaussian Blur"]
BLOCK_SIZES  = [4, 8, 16]

# Images à tester — à adapter selon ce que tu as dans img/
# Format : (label affiché, nom du fichier)
IMAGES = [
    ("Small (320x240)",   "small.jpg"),
    ("Medium (1280x720)", "medium.jpg"),
    ("Large (1920x1080)", "large.jpg"),
]

COLORS = {
    "CPU"       : "#e74c3c",
    "GPU global": "#3498db",
    "GPU opt"   : "#2ecc71",
}

# ===========================================================================
# Parsing de la sortie du programme
# ===========================================================================

def parse_output(output: str) -> dict:
    """
    Parse la sortie du programme filters et retourne un dict :
    { "BoxBlur": {"CPU": x, "GPU global": y, "GPU opt": z}, ... }
    """
    results = {}

    # Mapping entre les labels de sortie et les noms de filtres
    filter_map = {
        "BoxBlur"      : "BoxBlur",
        "Sobel"        : "Sobel",
        "Laplace"      : "Laplace",
        "Gaussian Blur": "Gaussian Blur",
    }

    current_filter = None

    for line in output.splitlines():
        line = line.strip()

        # Détection du filtre courant
        for key, name in filter_map.items():
            if line.startswith(f"--- {key}"):
                current_filter = name
                results[current_filter] = {}
                break

        if current_filter is None:
            continue

        # Extraction des temps
        m = re.search(r"CPU\s*:\s*([\d.]+)\s*ms", line)
        if m:
            results[current_filter]["CPU"] = float(m.group(1))

        m = re.search(r"GPU global\s*:\s*([\d.]+)\s*ms", line)
        if m:
            results[current_filter]["GPU global"] = float(m.group(1))

        # GPU optimisé (shared ou streams selon le filtre)
        m = re.search(r"GPU shared\s*:\s*([\d.]+)\s*ms", line)
        if m:
            results[current_filter]["GPU opt"] = float(m.group(1))

        m = re.search(r"GPU streams\s*:\s*([\d.]+)\s*ms", line)
        if m:
            results[current_filter]["GPU opt"] = float(m.group(1))

    return results


def run_benchmark(exe: str, img_path: str, block_size: int) -> dict:
    """Lance le programme et retourne les résultats parsés."""
    cmd = [exe, img_path, str(block_size)]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode != 0:
            print(f"  [Erreur] {' '.join(cmd)}\n{result.stderr}")
            return {}
        return parse_output(result.stdout)
    except subprocess.TimeoutExpired:
        print(f"  [Timeout] {' '.join(cmd)}")
        return {}


# ===========================================================================
# Benchmark 1 — Variation de la taille des blocs
# ===========================================================================

def bench1_block_size(exe: str, img_path: str, out_dir: str):
    """
    Fixe une image, fait varier la taille des blocs.
    Génère 4 graphes (un par filtre), chacun avec 3 courbes.
    """
    print("\n=== Benchmark 1 : Variation taille des blocs ===")

    # data[filtre][version] = [temps pour chaque taille de bloc]
    data = {f: {"CPU": [], "GPU global": [], "GPU opt": []} for f in FILTERS}

    for bs in BLOCK_SIZES:
        print(f"  Bloc {bs}x{bs}...")
        res = run_benchmark(exe, img_path, bs)
        for f in FILTERS:
            if f in res:
                for version in ["CPU", "GPU global", "GPU opt"]:
                    data[f][version].append(res[f].get(version, None))
            else:
                for version in ["CPU", "GPU global", "GPU opt"]:
                    data[f][version].append(None)

    # Tracé
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Benchmark 1 — Impact de la taille des blocs GPU\n"
                 f"(image : {os.path.basename(img_path)})", fontsize=14, fontweight='bold')

    for idx, (f, ax) in enumerate(zip(FILTERS, axes.flatten())):
        x = np.arange(len(BLOCK_SIZES))
        labels = [f"{b}x{b}" for b in BLOCK_SIZES]

        for version, color in COLORS.items():
            vals = data[f][version]
            valid = [(i, v) for i, v in enumerate(vals) if v is not None]
            if valid:
                xi, yi = zip(*valid)
                ax.plot([labels[i] for i in xi], yi,
                        marker='o', label=version, color=color, linewidth=2)

        ax.set_title(f, fontweight='bold')
        ax.set_xlabel("Taille des blocs")
        ax.set_ylabel("Temps (ms)")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(out_dir, "bench1_block_size.png")
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f"  Graphique sauvegardé : {path}")

    return data


# ===========================================================================
# Benchmark 2 — Variation de la taille des images
# ===========================================================================

def bench2_image_size(exe: str, img_dir: str, block_size: int, out_dir: str):
    """
    Fixe la taille de bloc (optimale), fait varier la taille des images.
    Génère 4 graphes (un par filtre), chacun avec 3 courbes.
    """
    print(f"\n=== Benchmark 2 : Variation taille des images (bloc {block_size}x{block_size}) ===")

    # Vérifier quelles images existent
    available = [(label, fname)
                 for label, fname in IMAGES
                 if os.path.exists(os.path.join(img_dir, fname))]

    if not available:
        print("  [Erreur] Aucune image trouvée dans", img_dir)
        return {}

    data = {f: {"CPU": [], "GPU global": [], "GPU opt": []} for f in FILTERS}
    img_labels = []

    for label, fname in available:
        img_path = os.path.join(img_dir, fname)
        print(f"  Image {label}...")
        img_labels.append(label)
        res = run_benchmark(exe, img_path, block_size)

        for f in FILTERS:
            if f in res:
                for version in ["CPU", "GPU global", "GPU opt"]:
                    data[f][version].append(res[f].get(version, None))
            else:
                for version in ["CPU", "GPU global", "GPU opt"]:
                    data[f][version].append(None)

    # Tracé
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Benchmark 2 — Impact de la taille des images\n"
                 f"(bloc {block_size}x{block_size})", fontsize=14, fontweight='bold')

    for idx, (f, ax) in enumerate(zip(FILTERS, axes.flatten())):
        for version, color in COLORS.items():
            vals = data[f][version]
            valid_idx = [i for i, v in enumerate(vals) if v is not None]
            valid_labels = [img_labels[i] for i in valid_idx]
            valid_vals   = [vals[i] for i in valid_idx]
            if valid_vals:
                ax.plot(valid_labels, valid_vals,
                        marker='o', label=version, color=color, linewidth=2)

        ax.set_title(f, fontweight='bold')
        ax.set_xlabel("Taille de l'image")
        ax.set_ylabel("Temps (ms)")
        ax.tick_params(axis='x', rotation=15)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(out_dir, "bench2_image_size.png")
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f"  Graphique sauvegardé : {path}")

    return data


# ===========================================================================
# Benchmark 3 — Speedup global (meilleure version GPU vs CPU)
# ===========================================================================

def bench3_global(exe: str, img_path: str, block_size: int, out_dir: str):
    """
    Calcule le speedup (CPU / GPU optimisé) pour chaque filtre.
    Affiche deux séries de barres : GPU global vs GPU optimisé.
    Ligne de référence à y=1 (pas d'accélération).
    """
    print(f"\n=== Benchmark 3 : Speedup global ===")
    print(f"  Image : {os.path.basename(img_path)}, bloc {block_size}x{block_size}")

    res = run_benchmark(exe, img_path, block_size)
    if not res:
        return

    speedup_global = []
    speedup_opt    = []

    for f in FILTERS:
        if f in res:
            cpu = res[f].get("CPU", 0)
            gpu = res[f].get("GPU global", 0)
            opt = res[f].get("GPU opt", gpu)
            speedup_global.append(round(cpu / gpu, 2) if gpu > 0 else 0)
            speedup_opt.append(round(cpu / opt, 2)    if opt > 0 else 0)
        else:
            speedup_global.append(0)
            speedup_opt.append(0)

    # Graphe barres groupées : GPU global vs GPU optimisé
    x     = np.arange(len(FILTERS))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    fig.suptitle("Benchmark 3 — Speedup GPU vs CPU (référence = 1x)\n"
                 f"(image : {os.path.basename(img_path)}, bloc {block_size}x{block_size})",
                 fontsize=14, fontweight='bold')

    bars1 = ax.bar(x - width/2, speedup_global, width,
                   label='GPU global', color=COLORS["GPU global"], alpha=0.85)
    bars2 = ax.bar(x + width/2, speedup_opt, width,
                   label='GPU optimisé', color=COLORS["GPU opt"], alpha=0.85)

    # Ligne de référence CPU (speedup = 1)
    ax.axhline(y=1, color=COLORS["CPU"], linestyle='--',
               linewidth=2, label='CPU (référence)')

    # Annotations sur les barres
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                f"x{bar.get_height():.1f}", ha='center', va='bottom',
                fontsize=10, fontweight='bold')
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                f"x{bar.get_height():.1f}", ha='center', va='bottom',
                fontsize=10, fontweight='bold', color='darkgreen')

    ax.set_xlabel("Filtre")
    ax.set_ylabel("Speedup (CPU / GPU)")
    ax.set_xticks(x)
    ax.set_xticklabels(FILTERS)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    path = os.path.join(out_dir, "bench3_speedup.png")
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f"  Graphique sauvegardé : {path}")


# ===========================================================================
# MAIN
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(description="Benchmarks CUDA filtres de convolution")
    parser.add_argument("--exe",     default="./filters",  help="Chemin vers l'exécutable")
    parser.add_argument("--img-dir", default="./img",      help="Dossier des images")
    parser.add_argument("--out-dir", default="./results",  help="Dossier de sortie des graphiques")
    parser.add_argument("--medium",  default="medium.jpg", help="Image medium pour bench 1 et 3")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    medium_img = os.path.join(args.img_dir, args.medium)
    if not os.path.exists(medium_img):
        # Fallback sur la première image disponible
        for _, fname in IMAGES:
            candidate = os.path.join(args.img_dir, fname)
            if os.path.exists(candidate):
                medium_img = candidate
                break
        else:
            print(f"[Erreur] Aucune image trouvée dans {args.img_dir}")
            return

    print(f"Image de référence : {medium_img}")

    # --- Benchmark 1 : taille des blocs ---
    data_b1 = bench1_block_size(args.exe, medium_img, args.out_dir)

    # Détermination du bloc optimal (GPU opt le plus rapide)
    best_block = 16  # défaut
    if data_b1:
        best_times = []
        for bs_idx, bs in enumerate(BLOCK_SIZES):
            total = sum(
                data_b1[f]["GPU opt"][bs_idx]
                for f in FILTERS
                if data_b1[f]["GPU opt"][bs_idx] is not None
            )
            best_times.append((total, bs))
        best_block = min(best_times, key=lambda t: t[0])[1]
        print(f"\n  => Bloc optimal détecté : {best_block}x{best_block}")

    # --- Benchmark 2 : taille des images ---
    bench2_image_size(args.exe, args.img_dir, best_block, args.out_dir)

    # --- Benchmark 3 : comparaison globale ---
    # Utilise la plus grande image disponible pour maximiser les différences
    large_img = medium_img
    for label, fname in reversed(IMAGES):
        candidate = os.path.join(args.img_dir, fname)
        if os.path.exists(candidate):
            large_img = candidate
            break

    bench3_global(args.exe, large_img, best_block, args.out_dir)

    print(f"\n=== Tous les graphiques sont dans : {args.out_dir}/ ===")


if __name__ == "__main__":
    main()