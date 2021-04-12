import cv2

image = 'smile_image.png'

# Able to run just the car detection using this video

video_file = ''

smile_video = cv2.VideoCapture(0)

smile_classifier = 'smile_detection.xml'

frontal_face_classifier = 'frontal_face_detection.xml'

detect_face = cv2.CascadeClassifier(frontal_face_classifier)

detect_smile = cv2.CascadeClassifier(smile_classifier)

print("DETECTING FACE AND SMILE ...")

while True:

    (read_face, frame) = smile_video.read()

    if not read_face:
        break

    grey_scaled_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # detect face
    faces = detect_face.detectMultiScale(grey_scaled_frame)

    # draw rectangles for smiley faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        detected_face = frame[y:y + h, x:x + w]

        # detected_face = (x, y, w, h)

        face_grey_scale = cv2.cvtColor(detected_face, cv2.COLOR_BGR2GRAY)

        # detect smiles
        smiles = detect_smile.detectMultiScale(face_grey_scale, scaleFactor=2.0, minNeighbors=20)

        # use this for drawing rectangles around smile
        for (x_, y_, w_, h_) in smiles:
           cv2.rectangle(detected_face, (x_, y_), (x_ + w_, y_ + h_), (255, 0, 0), 2)

        # label the face as smiling
        if len(smiles) > 0:
            cv2.putText(frame, " smiling ", (x+60, y+h+40), fontScale=3, fontFace= cv2.FONT_HERSHEY_PLAIN,color=(255, 255, 255))

    cv2.imshow(" SMILE DETECTION ", frame)

    key = cv2.waitKey(1)

    if key == 27 or key == 113:
        print("SMILE DETECTION FINISHED.")
        break

# video.release()
#cv2.destroyAllWindows()
