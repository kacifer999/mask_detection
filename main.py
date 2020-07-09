# 打开摄像头并灰度化显示
import cv2
import time
from mtcnn import MTCNN
from functions.functions import *
from tensorflow.keras.models import load_model
import datetime
from playsound import playsound
from multiprocessing import Process, Queue

capture = cv2.VideoCapture(0)
detector = MTCNN(min_face_size=50,steps_threshold = [0.1, 0.5, 0.9])
config = {'face_size':128}
class_model_dir = 'models/mask_model.h5'
class_model = load_model(class_model_dir)



def fachial_rec(q):
    time_lapes = 0
    class_list = [1,0]
    while(True):
        start = datetime.datetime.now()
        _, frame = capture.read()
        h,w,_ = frame.shape
        frame = cv2.resize(frame, (int(w/2),int(h/2)), interpolation=cv2.INTER_NEAREST)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = detector.detect_faces(frame)
        for face in faces:
            frame,class_id = predict_small(frame,class_model,face,config)
            class_list[class_id] += 1
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        frame = cv2.resize(frame, (int(w),int(h)), interpolation=cv2.INTER_NEAREST)
        cv2.imshow('frame', frame)
        end = datetime.datetime.now()
        time_lapes += (end - start).total_seconds()
        if time_lapes > 1:
            prob = class_list[1]/sum(class_list)
            if prob > 0.6:
                # print(datetime.datetime.now())
                q.put(True)
            time_lapes = 0
            class_list = [1,0]

        if cv2.waitKey(1) == ord('q'):
            break


def audio_player(q):
    while True:
        flag = q.get()
        if flag == True:
            playsound('Audio/Audio.mp3')
            while not q.empty():
                q.get_nowait()


if __name__ == '__main__':
    q = Queue()
    p_fachial_rec  = Process(target=fachial_rec,args=(q,))
    p_audio_player = Process(target=audio_player,args=(q,))
    p_fachial_rec.start()
    p_audio_player.start()
    p_fachial_rec.join()
    p_audio_player.terminate()