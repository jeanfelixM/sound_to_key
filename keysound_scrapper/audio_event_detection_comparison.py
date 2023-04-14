import timeit
import librosa
from audio_video_analysis import detect_audio_events
import numpy as np

def compare(test_files,ground_truth_labels,es):
    methods = ["energy_detection", "cross_correlation_detection"]
    time_results = {method: [] for method in methods}
    accuracy_results = {method: [] for method in methods}
    recall_results = {method: [] for method in methods}
    f1_score_results = {method: [] for method in methods}

    for method in methods:
        for i, test_file in enumerate(test_files):
            
            # Charger l'audio
            audio_data, sample_rate = librosa.load(test_file)

            # Charger les étiquettes réelles
            true_events = load_ground_truth_labels(ground_truth_labels[i])

            # Mesurer le temps d'exécution
            start_time = timeit.default_timer()
            detected_events = detect_audio_events(audio_data, sample_rate, mode=method, event_sample=es)
            elapsed_time = timeit.default_timer() - start_time
            time_results[method].append(elapsed_time)

            # Calculer les métriques de fiabilité
            accuracy, recall, f1_score = evaluate_metrics(detected_events,true_events,sample_rate)
            accuracy_results[method].append(accuracy)
            recall_results[method].append(recall)
            f1_score_results[method].append(f1_score)

    # Affichez les résultats
    for method in methods:
        print(f"Méthode: {method}")
        print(f"Temps moyen d'exécution: {np.mean(time_results[method]):.4f} secondes")
        print(f"Précision moyenne: {np.mean(accuracy_results[method]):.2f}")
        print(f"Rappel moyen: {np.mean(recall_results[method]):.2f}")
        print(f"Score F1 moyen: {np.mean(f1_score_results[method]):.2f}")
        print()


def load_ground_truth_labels(file_path):
    true_events = []

    with open(file_path, "r") as file:
        for line in file.readlines():
            start_time, end_time = map(float, line.strip().split(","))
            true_events.append((start_time, end_time))

    return true_events

def calculate_tp_fp(detected_windows, true_events, overlap_threshold):
    tp = 0
    fp = 0
    used_detected_windows = set()

    for true_start, true_end in true_events:
        for i, (detected_start, detected_end, _) in enumerate(detected_windows):
            if i not in used_detected_windows:
                overlap_start = max(true_start, detected_start)
                overlap_end = min(true_end, detected_end)

                # Calculer le chevauchement en pourcentage par rapport à l'événement réel
                overlap = max(0, overlap_end - overlap_start) / (true_end - true_start)

                if overlap >= overlap_threshold:
                    tp += 1
                    used_detected_windows.add(i)
                    break
    fp = len(detected_windows) - len(used_detected_windows)

    return tp, fp

def evaluate_metrics(detected_windows, true_events, sample_rate, overlap_threshold=0.5):
    
    # Convertir les événements réels en échantillons
    true_events_samples = [(int(start_time * sample_rate), int(end_time * sample_rate)) for start_time, end_time in true_events]

    # Calculer les TP, FP et FN
    tp, fp = calculate_tp_fp(detected_windows, true_events_samples, overlap_threshold)
    fn = len(true_events) - tp

    # Calculer la précision, le rappel et le score F1
    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0
    f1_score = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0

    return precision, recall, f1_score

if __name__ == "__main__":
    
    test_files = ["test_audio1.wav", "test_audio2.wav", "test_audio3.wav"]
    ground_truth_labels = ["labels1.txt", "labels2.txt", "labels3.txt"]
    event_sample = "path/to/event_sample.wav" 

    # Charger l'échantillon de l'événement pour la détection par corrélation croisée
    es, _ = librosa.load(event_sample)

    compare(test_files, ground_truth_labels, es)

