import k_means.Img_Segmentation.K_means_ImgSeg as IS
import k_means.K_means as KM
import cv2

def f(x):
    pass

cap = cv2.VideoCapture(0)

cv2.namedWindow('image')

cv2.createTrackbar('k points', 'image', 2, 255, f)

cv2.createTrackbar('max iteration', 'image', -1, 400, f)

cv2.createTrackbar('scale', 'image', 1, 100, f)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    scale = cv2.getTrackbarPos('scale', 'image') + 1

    img_array = cv2.resize(frame, dsize=(int(frame.shape[1] * (1 / scale)), int(frame.shape[0] * (1 / scale))),interpolation=3)

    # Our operations on the frame come here
    k = cv2.getTrackbarPos('k points', 'image')
    kp = KM.make_k_points(k, 3, 0, 255)
    gray = IS.img_seg(img_array, kp, cv2.getTrackbarPos('max iteration', 'image'))

    gray = cv2.resize(gray, dsize=(int(gray.shape[1] * scale), int(gray.shape[0] * scale)), interpolation=3)

    # Display the resulting frame
    cv2.imshow('image', gray)
    if cv2.waitKey(50) & 0xFF == ord('q'):
        break


# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()