import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import cv2
import os
import cv2
from queue import Queue
import io
from PIL import Image

def display_frames_and_spectrograms(key_n_frames, key_n_sounds, sample_rate=44100, columns=4):
    plt.figure(figsize=(15, 15))

    for i, key_frame in enumerate(key_n_frames, start=1):
        plt.subplot(2 * (len(key_n_frames) // columns + 1), columns, 2 * i - 1)
        plt.imshow(cv2.threshold(cv2.cvtColor(key_frame["frame"], cv2.COLOR_BGR2GRAY), 127, 255, cv2.THRESH_BINARY)[1])
        plt.title(f"Key: {key_frame['key']}")
        plt.axis('off')

        # Trouver le son associé
        for key_n_sound in key_n_sounds:
            if key_n_sound["key"] == key_frame["key"]:
                audio_segment = key_n_sound["sound"]
                break

        # Générer le spectrogramme
        spectrogram = np.abs(librosa.stft(audio_segment))

        # Afficher le spectrogramme
        plt.subplot(2 * (len(key_n_frames) // columns + 1), columns, 2 * i)
        librosa.display.specshow(librosa.amplitude_to_db(spectrogram, ref=np.max), sr=sample_rate, x_axis='time', y_axis='log')
        plt.colorbar(format='%+2.0f dB')
        plt.tight_layout()

    plt.show()
    
def save_spectro(queue):
    while True:
        data = queue.get()
        # check for stop
        if data is None:
            break
        spec = spectro_gen(data[0])
        write_data(spec,data[1])     
    # all done
    print('Ecriture finie')
    
def spectro_gen(audio_segment, sample_rate=44100):
    # Générer le spectrogramme
    spectrogram = np.abs(librosa.stft(audio_segment))

    # Créer un objet figure et définir ses dimensions
    fig, ax = plt.subplots(figsize=(4, 4))

    # Afficher le spectrogramme
    img = librosa.display.specshow(librosa.amplitude_to_db(spectrogram, ref=np.max), sr=sample_rate, x_axis='time', y_axis='log', ax=ax)
    fig.colorbar(img, format='%+2.0f dB')

    # Supprimer les axes et les espaces blancs autour de l'image
    plt.axis('off')
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)

    # Sauvegarder l'image du spectrogramme dans un objet BytesIO
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    buf.seek(0)

    # Charger l'image du spectrogramme en utilisant PIL
    spectrogram_image = Image.open(buf)

    # Fermer la figure pour libérer la mémoire
    plt.close(fig)

    # Retourner l'image du spectrogramme
    return spectrogram_image


def write_data(spectrogram, key):
    # Créer un dossier avec le nom de la clé s'il n'existe pas
    output_directory = f"{key}"
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Générer un nom de fichier unique en se basant sur l'heure actuelle
    filename = f"{key}_{str(time.time()).replace('.', '_')}.png"

    # Créer le chemin du fichier de sortie
    output_path = os.path.join(output_directory, filename)

    # Enregistrer le spectrogramme dans un fichier image
    cv2.imwrite(output_path, spectrogram)
