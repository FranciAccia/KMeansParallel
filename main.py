import numpy as np
import time
import multiprocessing
from functools import partial
from sklearn.datasets import make_blobs

# --- CONFIGURAZIONE DATASET E TEST ---
# Aumentato per rispettare la regola "tempo >= 10s"
# Se il tuo PC è molto veloce, aumenta N_SAMPLES o N_FEATURES
N_SAMPLES = 10000000
N_FEATURES = 5  # Aumentato dimensioni per carico computazionale maggiore
K_CLUSTERS = 5
MAX_ITERS = 10  # Limitato per il test, altrimenti dura troppo
NUM_RUNS = 5  # Minimo 5 run per statistiche affidabili
WARMUP_RUNS = 1  # Scarta i primi run
CORES_TO_TEST = [1, 4, 8, 16]  # Test scalabilità


# --- FUNZIONI COMUNI ---

def initialize_centroids(data, k):
    np.random.seed(42)  # Seed fisso per riproducibilità
    indices = np.random.choice(data.shape[0], k, replace=False)
    return data[indices]


def update_centroids(data, labels, k):
    new_centroids = np.zeros((k, data.shape[1]))
    for i in range(k):
        points = data[labels == i]
        if len(points) > 0:
            new_centroids[i] = points.mean(axis=0)
    return new_centroids


# --- VERSIONE SEQUENZIALE ---

def assign_clusters_sequential(data, centroids):
    # Calcolo distanze: ||data - centroids||^2
    # Broadcasting: (N, 1, D) - (K, D) -> (N, K, D)
    distances = np.linalg.norm(data[:, np.newaxis] - centroids, axis=2)
    return np.argmin(distances, axis=1)


def kmeans_sequential(data, k, max_iters):
    centroids = initialize_centroids(data, k)
    for _ in range(max_iters):
        labels = assign_clusters_sequential(data, centroids)
        centroids = update_centroids(data, labels, k)
    return centroids


# --- VERSIONE PARALLELA ---

def assign_chunk(data_chunk, centroids):
    """ Funzione worker eseguita da ogni processo su un chunk di dati """
    distances = np.linalg.norm(data_chunk[:, np.newaxis] - centroids, axis=2)
    return np.argmin(distances, axis=1)


def kmeans_parallel(data, k, max_iters, n_processes):
    centroids = initialize_centroids(data, k)

    # Prepara il pool di processi
    # Usiamo 'spawn' o 'fork' a seconda dell'OS, ma Pool gestisce i dettagli
    # Nota: L'overhead di creazione processi in Python è alto,
    # quindi per dataset piccoli la versione parallela potrebbe essere più lenta.

    # Dividiamo i dati in chunk
    chunk_size = int(np.ceil(data.shape[0] / n_processes))
    data_chunks = [data[i:i + chunk_size] for i in range(0, data.shape[0], chunk_size)]

    with multiprocessing.Pool(processes=n_processes) as pool:
        for _ in range(max_iters):
            # STEP 1: ASSEGNAZIONE PARALLELA
            # Ogni processo riceve un chunk e i centroidi correnti
            # Usiamo partial per fissare l'argomento 'centroids'
            func = partial(assign_chunk, centroids=centroids)

            # Map distribuisce il lavoro
            results = pool.map(func, data_chunks)

            # Ricostruiamo l'array delle labels dai risultati parziali
            labels = np.concatenate(results)

            # STEP 2: AGGIORNAMENTO SEQUENZIALE
            # L'aggiornamento è veloce e agisce come punto di sincronizzazione
            centroids = update_centroids(data, labels, k)

    return centroids


# --- BENCHMARKING SUITE [cite: 63] ---

def run_benchmark():
    print(f"--- INIZIO BENCHMARK ---")
    print(f"Hardware: {multiprocessing.cpu_count()} CPU cores logici rilevati [cite: 86]")
    print(f"Dataset: {N_SAMPLES} campioni, {N_FEATURES} features, {K_CLUSTERS} cluster ")
    print(f"Configurazione: {NUM_RUNS} run (media), {WARMUP_RUNS} warmup [cite: 67, 82]")
    print("-" * 60)

    # Generazione Dati (fuori dal timer)
    print("Generazione dati sintetici in corso...")
    data, _ = make_blobs(n_samples=N_SAMPLES, centers=K_CLUSTERS, n_features=N_FEATURES, random_state=42)

    results = {}

    # Test per ogni configurazione di core
    for p in CORES_TO_TEST:
        times = []
        mode = "SEQUENZIALE" if p == 1 else f"PARALLELO ({p} proc)"

        print(f"\nTestando: {mode}...")

        # Loop totale run (Warmup + Misurazioni)
        total_runs = WARMUP_RUNS + NUM_RUNS
        for i in range(total_runs):
            # Start Timer [cite: 80, 84]
            start = time.perf_counter()  # High-resolution timer

            if p == 1:
                kmeans_sequential(data, K_CLUSTERS, MAX_ITERS)
            else:
                kmeans_parallel(data, K_CLUSTERS, MAX_ITERS, n_processes=p)

            end = time.perf_counter()
            duration = end - start

            # Gestione Warmup
            if i < WARMUP_RUNS:
                print(f"  Run {i + 1} (Warmup): {duration:.4f} s (Scartato)")
            else:
                print(f"  Run {i + 1}: {duration:.4f} s")
                times.append(duration)

        # Statistiche
        mean_time = np.mean(times)
        std_time = np.std(times)
        results[p] = mean_time
        print(f"  >> MEDIA: {mean_time:.4f} s | STD: {std_time:.4f} s")

    # --- REPORT FINALE E SPEEDUP [cite: 69, 70] ---
    print("\n" + "=" * 60)
    print(f"{'Core':<10} | {'Tempo Medio (s)':<15} | {'Speedup':<10} | {'Efficienza':<10}")
    print("-" * 60)

    t_seq = results[1]

    for p in CORES_TO_TEST:
        t_par = results[p]
        speedup = t_seq / t_par
        efficiency = speedup / p  # Extra metric spesso utile

        print(f"{p:<10} | {t_par:<15.4f} | {speedup:<10.2f} | {efficiency:<10.2f}")

    print("=" * 60)
    print("Nota: Se Speedup < 1 con pochi dati, domina l'overhead di multiprocessing.")
    print("Aumentare N_SAMPLES per vedere benefici reali su Python.")


if __name__ == "__main__":
    run_benchmark()