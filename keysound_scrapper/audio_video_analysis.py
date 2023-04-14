import librosa
import numpy as np
import cv2
from keysound_scrapper.utils import load_yamnet_model, process_audio
from roi_finder import detect_text_regions
from utils import display_frames_and_spectrograms
import pytesseract
from pytesseract import Output
from queue import Queue

HOP_LENGTH = 512

def detect_audio_events(audio_data, sample_rate, factor=1.5, mode = "energy_detection", event_sample=None):
    if mode == "energy_detection":
        return energy_detection(audio_data, sample_rate, factor)
    elif mode == "cross_correlation_detection":
        return cross_correlation_detection(audio_data, event_sample,sample_rate, factor)
    elif mode == "yamnet_detection":
        return yamnet_detection(audio_data, sample_rate)
    else:
        raise Exception("Mode not supported")
  
    
  
  
def energy_detection(audio_data, sample_rate, factor=1.5):
    #Calcul de l'énergie d'une frame
    spectrogram = np.abs(librosa.stft(audio_data, hop_length=HOP_LENGTH))
    frame_energy = np.sum(spectrogram, axis=0)

    events = []

    #Calcul de l'énergie moyenne du signal
    fft_audio_data = np.fft.fft(audio_data)
    power_spectral_density = np.abs(fft_audio_data) ** 2
    average_power_spectral_density = np.mean(power_spectral_density)
    threshold = average_power_spectral_density * factor

    #Détection des événements
    start_frame = 0
    started = False
    for i in range(0, len(frame_energy)):
        tf = frame_energy[i]
        if tf > threshold:
            if not started:
                start_frame = i
                started = True
        else:
            if started:
                event_data = audio_data[start_frame * HOP_LENGTH : i * HOP_LENGTH]
                events.append((start_frame, i, event_data))
                print(f"{start_frame * HOP_LENGTH  / sample_rate} to {i * HOP_LENGTH  / sample_rate}")
                started = False
                start_frame = 0

    return events

def cross_correlation_detection(audio_data, event_sample, sample_rate, threshold_factor=1.5):
    
    # Normalisation des signaux
    audio_data = audio_data / np.max(np.abs(audio_data))
    event_sample = event_sample / np.max(np.abs(event_sample))

    # Calcul de l'intercorrélation entre le signal audio et l'échantillon de l'événement
    correlation = np.correlate(audio_data, event_sample, mode='valid')

    # Calcul du seuil
    max_possible_correlation = 1
    threshold = threshold_factor * max_possible_correlation

    # Détection des événements
    events = []
    event_length = len(event_sample)
    i = 0
    while i < len(correlation):
        if correlation[i] > threshold:
            start_sample = i
            end_sample = i + event_length
            events.append((start_sample, end_sample, audio_data[start_sample:end_sample]))
            print(f"{start_sample / sample_rate} to {end_sample / sample_rate}")
            i += event_length
        else:
            i += 1

    return events
  
def yamnet_detection(audio_data, sample_rate, class_map = None ,model_path= ""):
    
    audio_data = process_audio(audio_data, sample_rate)
    model = load_yamnet_model(model_path)
    yamnet_input = np.expand_dims(audio_data, 0)
    
    # Prédictions du modèle
    yamnet_output = model(yamnet_input)
    
    # Détection des événements
    desired_class = "Keyboard" 
    class_index = class_map[class_map["display_name"] == desired_class].index[0]

    class_scores = yamnet_output[0][:, class_index].numpy()

    threshold = 0.5
    events_detected = (class_scores >= threshold)

    return events_detected


def extract_candidate_frames(video, sample_rate, events):
    frames = []
    sons = []
    frame_number = 0
    son_id = 0

    fps = int(video.get(cv2.CAP_PROP_FPS))

    for event in events:
        start_frame, end_frame, key_sound = event
        video_start_frame = int(start_frame * HOP_LENGTH / sample_rate * fps)
        video_end_frame = int(end_frame * HOP_LENGTH / sample_rate * fps)

        sons.append(key_sound)
        while video.isOpened():
            ret, frame = video.read()
            if not ret:
                break

            if video_start_frame <= frame_number <= video_end_frame:
                frames.append({"frame": frame, "son": son_id})

            frame_number += 1
            if frame_number > video_end_frame:
                break

        son_id += 1

    video.release()
    print(f"Found !!!!!!!!!!!{len(frames)} frames")
    return frames, sons


def process_frames(frames, sons, net, queue, newW=640, newH=640,debug=True):
    ps = ""  # previous string
    pf = None  # previous frame
    key_n_sounds = []
    key_n_frames = []
    frame_count = 0
    layerNames = [
	"feature_fusion/Conv_7/Sigmoid",
	"feature_fusion/concat_3"]

    if debug:
        # Détection du texte dans les frames candidats
        for elt in frames:
            f = elt['frame']
            sid = elt['son']

            gray = detect_text_regions(f, net, layerNames, newW, newH)

            # Prétraitement sur l'image pour améliorer la précision de l'OCR (optionnel)
            treshed = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

            # Appliquer OCR sur l'image
            config = '-l eng --oem 1 --psm 1'
            text = pytesseract.image_to_string(treshed, config=config)

            text_stripped = text.rstrip()  # pour enlever les retours à la ligne, espaces, etc.
            if (text_stripped != ps and text_stripped != "" and (len(text_stripped) > 0)):
                print(text_stripped)
                avant_dernier_caractere = text_stripped[-2] if len(text_stripped) > 1 else None
                if avant_dernier_caractere == " ":
                    key_n_sound = {"sound": sons[sid - 1], "key": " "}
                    key_n_sounds.append(key_n_sound)
                    key_n_frame = {"frame": pf, "key": " "}
                    key_n_frames.append(key_n_frame)
                    queue.put((sons[sid - 1]," "))
                key = text_stripped[-1]
                queue.put((sons[sid],key))
                key_n_sound = {"sound": sons[sid], "key": key}
                key_n_frame = {"frame": f, "key": key}
                key_n_frames.append(key_n_frame)
                key_n_sounds.append(key_n_sound)
                ps = text_stripped
                pf = f

            frame_count += 1

            if frame_count % 50 == 0:
                print(f"Processed {frame_count} frames")

        print(f"Total frames: {frame_count}")
    
    else :
        # Détection du texte dans les frames candidats
        for elt in frames:
            f = elt['frame']
            sid = elt['son']

            gray = detect_text_regions(f, net, layerNames, newW, newH,debug)

            # Prétraitement sur l'image pour améliorer la précision de l'OCR (optionnel)
            treshed = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

            # Appliquer OCR sur l'image
            config = '-l eng --oem 1 --psm 1'
            text = pytesseract.image_to_string(treshed, config=config)

            text_stripped = text.rstrip()  # pour enlever les retours à la ligne, espaces, etc.
            if (text_stripped != ps and text_stripped != "" and (len(text_stripped) > 0)):
                avant_dernier_caractere = text_stripped[-2] if len(text_stripped) > 1 else None
                if avant_dernier_caractere == " ":
                    queue.put((sons[sid - 1]," "))         
                key = text_stripped[-1]
                queue.put((sons[sid],key))
                ps = text_stripped
                pf = f
    queue.put(None)
    return key_n_sounds, key_n_frames


if __name__ == "__main__":
    
    input_audio_file = "audio_output.wav"
    input_video_file = "video_output.mp4"
    
    # Détecter les événements audio
    events = detect_audio_events(input_audio_file)
    
    # Extraire les frames candidates et les sons associés
    frames, sons = extract_candidate_frames(input_video_file, events)
    
    # Traiter les frames candidates
    key_n_sounds, key_n_frames = process_frames(frames, sons)
    
    # Afficher les résultats
    print("Nombre d'événements détectés :", len(events))
    print("Nombre de frames candidates :", len(frames))
    print("Nombre de sons associés :", len(sons))
    print("Nombre de sons avec des clés détectées :", len(key_n_sounds))
    print("Nombre de frames avec des clés détectées :", len(key_n_frames))
    display_frames_and_spectrograms(key_n_frames, key_n_sounds, sample_rate=44100, columns=4)

