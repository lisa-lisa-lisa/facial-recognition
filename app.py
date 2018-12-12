from pathlib import Path
import cv2
import face_recognition
import json
import requests
import time
import pickle
import sys
import os


images_dir = Path("data/")
video_capture = cv2.VideoCapture(0)

known_face_encodings = []
known_face_names = []
service_url = 'http://192.168.7.53:8000/unlock'

t0 = time.time()
print('Initialisation - Beginning to encode the images')

JITTER = 5

if len(sys.argv) > 1 and sys.argv[1] == "--cached":
    print("loading from cache")
    with open("cache.pkl", "rb") as f:
        cache = pickle.load(f)
    known_face_encodings = cache["known_face_encodings"]
    known_face_names = cache["known_face_names"]
else:
    num_photos = len(list(images_dir.glob("*")))
    for n, filename in enumerate(sorted(images_dir.glob("*")), start=1):
        print(n, "of", num_photos)
        if not filename.is_dir():
            continue
        metadata_file = filename / "meta.json"
        # print(metadata_file)

        f = open(metadata_file, "r")
        metadata = json.loads(f.read())
        username = metadata["username"]
        # print(username)

        for photo in filename.glob("photos/*"):
            # print(photo)
            encoding_path = Path(f"encodings/{username}-{photo.name}.pkl")
            if encoding_path.exists():
                print('encoding found')
                with open(encoding_path, "rb") as f:
                    encoding = pickle.load(f)
            else:
                image = face_recognition.load_image_file(photo)
                try:
                    encoding = face_recognition.face_encodings(image, num_jitters=JITTER)[0]
                except IndexError :    
                    print('failed to find face in photo', photo)
                    continue
                with open(encoding_path, "wb") as f:
                    pickle.dump(encoding, f)
            known_face_encodings.append(encoding)
            known_face_names.append(username)
    # print(known_face_encodings)
    # print(known_face_names)
    print('Initialisation - Done', time.time() - t0)

    with open("cache.pkl", "wb") as f:
        f.write(pickle.dumps({
            "known_face_encodings": known_face_encodings,
            "known_face_names": known_face_names,
        }))


while True:
    
    ret, frame = video_capture.read()

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_frame = frame[:, :, ::-1]
    width = rgb_frame.shape[1]
    rgb_frame = rgb_frame[:, width // 3:2 * width // 3, :]

    try:
        input_encoding = face_recognition.face_encodings(rgb_frame, num_jitters=JITTER)[0]
    except IndexError:
        #print("No face found in  image")
        continue
    # print(input_encoding)

    distances = face_recognition.face_distance(known_face_encodings, input_encoding)
    # print(distances)

    found_match = False
    best_match = None
    best_match_distance = 1
    for i,distance in enumerate(distances):
        if distance < 0.47:
            if distance < best_match_distance:
                best_match = known_face_names[i]
                best_match_distance = distance
                found_match = True
                # print('Recognised User: ', known_face_names[i], 'with match: ', distance)
    if found_match:
        print('Best matched user: ', best_match, 'with match: ', best_match_distance)
        try:
            requests.get(service_url, timeout=2)
            # os.system("afplay access-granted.m4a")
            time.sleep(5)
        except:
            print("Problem in calling the web service for opening the door")
    else:
        print("Can't recognise the face....Please register first...")    
         