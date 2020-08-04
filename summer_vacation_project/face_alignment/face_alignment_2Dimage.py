import face_alignment
from skimage import io
import cv2


fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, device='cpu')

image = cv2.imread('./assets/aflw-test.jpg')
input = io.imread('./assets/aflw-test.jpg')
preds = fa.get_landmarks(input)

for (x, y) in preds[0]:
   print("{}, {}".format(x,y))
   cv2.circle(image, (x, y), 1, (0, 0, 255), -1)

cv2.imshow("result", image)
cv2.waitKey(0)