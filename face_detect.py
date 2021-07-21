import cv2
import numpy as np
from mtcnn import MTCNN

age_model = cv2.dnn.readNetFromCaffe("model/age.prototxt", "model/dex_chalearn_iccv2015.caffemodel")
gender_model = cv2.dnn.readNetFromCaffe("model/gender.prototxt", "model/gender.caffemodel")
detector = MTCNN()


def detect_face(img):
    mt_res = detector.detect_faces(img)
    return_res = []
    for face in mt_res:
        x, y, width, height = face['box']
        center = [x+(width/2), y+(height/2)]
        max_border = max(width, height)
        # center alignment
        left = max(int(center[0]-(max_border/2)), 0)
        right = max(int(center[0]+(max_border/2)), 0)
        top = max(int(center[1]-(max_border/2)), 0)
        bottom = max(int(center[1]+(max_border/2)), 0)
        # crop face
        detected_face = img[int(y):int(y+height), int(x):int(x+width)]
        detected_face = cv2.resize(detected_face, (224, 224)) #img shape is (224, 224, 3) now
        img_blob = cv2.dnn.blobFromImage(detected_face ) # img_blob shape is (1, 3, 224, 224)
        # gender detection
        gender_model.setInput(img_blob)
        gender_class = gender_model.forward()[0]
        # age detection
        age_model.setInput(img_blob)
        age_dist = age_model.forward()[0]
        output_indexes = np.array([i for i in range(0, 101)])
        age = round(np.sum(age_dist * output_indexes), 2)
        # output to the cv2
        return_res.append([top, right, bottom, left, age, gender_class])
    return return_res

# Get a reference to webcam
video_capture = cv2.VideoCapture(0)

while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()

    # Convert the image from BGR color (which OpenCV uses) to RGB color
    rgb_frame = frame[:, :, ::-1]

    # Find all the faces in the current frame of video
    face_locations = detect_face(rgb_frame)

    # Display the results
    for top, right, bottom, left, age, gender_class in face_locations:
        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        sex_preds = np.max(gender_class)
        sex_text = 'Woman ' if np.argmax(gender_class) == 0 else 'Man'
        cv2.putText(frame, 'Sex: {}({:.3f})'.format(sex_text, sex_preds), (left, top-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36,255,12), 1)
        cv2.putText(frame, 'Age: {:.3f}'.format(age), (left, top-25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36,255,12), 1)

    # Display the resulting image
    cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()