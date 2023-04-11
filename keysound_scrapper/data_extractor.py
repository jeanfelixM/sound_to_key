from threading import Thread
from audio_video_analysis import *
from utils import *
import os

def extract_data(videofiles, net, debug=False):
    os.environ['PATH'] += r';C:\Program Files\Tesseract-OCR'
    q = Queue()
    ss = Thread(target=save_spectro,args=(q,))
    ss.start()
    for i in range(len(videofiles)):
        videofile = videofiles[i]
        video = cv2.VideoCapture(videofile)
        print("vid extracted")
        audio,sr = extract_audio(videofile)
        print("audio extracted")
        events = detect_audio_events(audio,sr)
        print("events detected")
        frames, sons = extract_candidate_frames(video, sr, events)
        #process_frames(frames, sons, net, q)
        pf = Thread(target = process_frames, args=(frames, sons, net, q,))
        pf.start()
    ss.join()

if __name__ == "__main__":
    # Charger le modèle EAST
    east_model_path = "keysound_scrapper/frozen_east_text_detection.pb"
    net = cv2.dnn.readNet(east_model_path)

    # Liste des fichiers vidéo à traiter
    videofiles = ["keysound_scrapper/vid.mp4"]

    # Extraire les données des fichiers vidéo
    extract_data(videofiles, net)