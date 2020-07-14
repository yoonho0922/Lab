import cv2

img = cv2.imread('public/cc-2.png', 0)

cv2.imshow('image', img)
k = cv2.waitKey(0) & 0xff

if k == 27:
    cv2.destroyAllWindows()

elif k == ord('s'):
    cv2.imwrite('public/messigray.png', img)
    cv2.destroyAllWindows()