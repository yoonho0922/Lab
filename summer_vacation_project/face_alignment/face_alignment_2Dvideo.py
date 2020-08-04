import face_alignment
import cv2

capture = cv2.VideoCapture(0)
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, device='cpu')

while True:
    ret, frame = capture.read()

    preds = fa.get_landmarks(frame)

    for (x, y) in preds[0]:
        print("{}, {}".format(x, y))
        cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)

    cv2.imshow("VideoFrame", frame)
    if cv2.waitKey(1) > 0: break

capture.release()
cv2.destroyAllWindows()