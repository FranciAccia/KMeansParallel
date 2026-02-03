import numpy as np
import time
import multiprocessing
import os
import matplotlib.pyplot as plt
from functools import partial
from sklearn.datasets import make_blobs

# IMPORTANTE: Impedisce a NumPy di usare il multithreading interno.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

# --- CONFIGURAZIONE ---
SEED = 42
np.random.seed(SEED)

# Parametri Dataset
N_SAMPLES = 8000000
N_FEATURES = 6
K_CLUSTERS = 5
MAX_ITERS = 25

# Configurazione Benchmark
NUM_RUNS = 3
WARMUP_RUNS = 1
# M2 ha 8 core totali (4 Performance + 4 Efficiency).
# Testiamo tutti gli step per vedere il "ginocchio" dopo i 4 core.
CORES_TO_TEST = [1, 2, 4, 8]


# --- 1. FUNZIONI COMUNI ---

def initialize_centroids(data, k):
    indices = np.random.choice(data.shape[0], k, replace=False)
    return data[indices]


# --- 2. SEQUENZIALE ---

def assign_clusters_sequential(data, centroids):
    distances = np.linalg.norm(data[:, np.newaxis] - centroids, axis=2)
    return np.argmin(distances, axis=1)


def update_centroids_sequential(data, labels, k):
    new_centroids = np.zeros((k, data.shape[1]))
    for i in range(k):
        points = data[labels == i]
        if len(points) > 0:
            new_centroids[i] = points.mean(axis=0)
    return new_centroids


def kmeans_sequential(data, k, max_iters):
    centroids = initialize_centroids(data, k)
    for _ in range(max_iters):
        labels = assign_clusters_sequential(data, centroids)
        centroids = update_centroids_sequential(data, labels, k)
    return centroids


# --- 3. PARALLELO ---

def worker_map_reduce(data_chunk, centroids):
    # Ricalcoliamo distanze e somme parziali
    k = centroids.shape[0]
    d = data_chunk.shape[1]

    distances = np.linalg.norm(data_chunk[:, np.newaxis] - centroids, axis=2)
    labels = np.argmin(distances, axis=1)

    partial_sums = np.zeros((k, d))
    partial_counts = np.zeros(k)

    for i in range(k):
        points_in_cluster = data_chunk[labels == i]
        if len(points_in_cluster) > 0:
            partial_sums[i] = points_in_cluster.sum(axis=0)
            partial_counts[i] = len(points_in_cluster)

    return partial_sums, partial_counts


def kmeans_parallel(data, k, max_iters, n_processes):
    centroids = initialize_centroids(data, k)
    chunk_size = int(np.ceil(data.shape[0] / n_processes))
    data_chunks = [data[i:i + chunk_size] for i in range(0, data.shape[0], chunk_size)]

    # Su macOS usiamo il context 'spawn' o di default del sistema
    ctx = multiprocessing.get_context('spawn')

    with ctx.Pool(processes=n_processes) as pool:
        for _ in range(max_iters):
            worker_func = partial(worker_map_reduce, centroids=centroids)
            results = pool.map(worker_func, data_chunks)

            global_sums = np.zeros_like(centroids)
            global_counts = np.zeros(k)

            for p_sum, p_count in results:
                global_sums += p_sum
                global_counts += p_count

            mask = global_counts > 0
            centroids[mask] = global_sums[mask] / global_counts[mask][:, np.newaxis]

    return centroids


# --- 4. BENCHMARK ---

def run_benchmark():
    print(f"--- BENCHMARK K-MEANS su MAC M2 (Seed={SEED}) ---")
    print(f"Dataset: {N_SAMPLES} samples, {N_FEATURES} features")
    print("Nota: OMP_NUM_THREADS=1 forzato per misurare puro parallelismo di processo.\n")

    print(f"Generazione dati...", end=" ", flush=True)
    X, _ = make_blobs(n_samples=N_SAMPLES, centers=K_CLUSTERS,
                      n_features=N_FEATURES, random_state=SEED)
    print("Fatto.\n")

    results_time = {}

    for p in CORES_TO_TEST:
        times = []
        desc = "Sequenziale" if p == 1 else f"Parallelo ({p} core)"
        print(f"Testing {desc}...")

        for r in range(WARMUP_RUNS + NUM_RUNS):
            start = time.perf_counter()

            if p == 1:
                kmeans_sequential(X, K_CLUSTERS, MAX_ITERS)
            else:
                kmeans_parallel(X, K_CLUSTERS, MAX_ITERS, n_processes=p)

            end = time.perf_counter()
            duration = end - start

            if r >= WARMUP_RUNS:
                times.append(duration)
                print(f"  Run {r + 1}: {duration:.4f}s")
            else:
                print(f"  Run W (Warmup): {duration:.4f}s (scartato)")

        avg_time = np.mean(times)
        results_time[p] = avg_time
        print(f"  >> Media: {avg_time:.4f}s\n")

    return results_time


def plot_results(results):
    cores = sorted(results.keys())
    times = [results[c] for c in cores]
    t_seq = results[1]
    speedups = [t_seq / t for t in times]
    efficiency = [s / c for s, c in zip(speedups, cores)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Speedup
    ax1.plot(cores, speedups, 'o-', linewidth=2, label='Tu')
    ax1.plot(cores, cores, 'k--', alpha=0.5, label='Ideale')
    ax1.set_title('Speedup')
    ax1.set_xlabel('Processi')
    ax1.set_ylabel('Speedup (T_seq / T_par)')
    ax1.grid(True)
    ax1.legend()
    ax1.set_xticks(cores)

    # Efficienza
    ax2.plot(cores, efficiency, 's-', color='orange', linewidth=2)
    ax2.axhline(y=1.0, color='k', linestyle='--')
    ax2.set_title('Efficienza Parallela')
    ax2.set_ylim(0, 1.2)
    ax2.grid(True)
    ax2.set_xticks(cores)

    plt.tight_layout()
    plt.savefig('kmeans_m2_results.png')
    print("Grafico salvato: kmeans_m2_results.png")
    plt.show()


if __name__ == "__main__":
    multiprocessing.freeze_support()
    timings = run_benchmark()
    plot_results(timings)

    print("\nRISULTATI FINALI:")
    t_seq = timings[1]
    for c in sorted(timings.keys()):
        sp = t_seq / timings[c]
        print(f"Core {c}: {timings[c]:.4f}s | Speedup: {sp:.2f}x | Eff: {sp / c:.2f}")