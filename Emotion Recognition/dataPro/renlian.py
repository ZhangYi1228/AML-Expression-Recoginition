# Face
import cv2
import os
import glob

# final cropped image size
size_m = 120
size_n = 120


def detect(img, cascade):
    rects = cascade.detectMultiScale(img, scaleFactor=1.3, minNeighbors=4, minSize=(30, 30),
                                     flags=cv2.CASCADE_SCALE_IMAGE)
    if len(rects) == 0:
        return []
    rects[:, 2:] += rects[:, :2]
    return rects


cascade = cv2.CascadeClassifier("D:/CK/haarcascade_frontalface_default.xml")
imglist = glob.glob("D:/renlian/*")
for list in imglist:

    # print(list)
    # cv2read image
    img = cv2.imread(list)
    dst = img
    rects = detect(dst, cascade)
    for x1, y1, x2, y2 in rects:
        # Adjust the size of the face capture. Horizontal is x, vertical is y
        roi = dst[y1 + 10:y2 + 20, x1 + 10:x2]
        img_roi = roi
        re_roi = cv2.resize(img_roi, (size_m, size_n))

        # Save the new image to data/image/jaffe_1
        f = "{}/{}".format("D:/renlian", "jaffe_1")
        # print(f)
        if not os.path.exists(f):
            os.mkdir(f)
        # Cut image path
        path = list.split(".")

        # The new image is saved to data/image/jaffe_1 with the suffix jpg
        cv2.imwrite("D:/renlian/t1.png", re_roi)