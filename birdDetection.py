from cgitb import small
from skimage import io
import numpy as np
import cv2


confidence_thr = 0.5


CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
    "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
    "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
    "sofa", "train", "tvmonitor"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

# load our serialized model from disk
print("[INFO] loading model...")

shared_dir = '../../MobileNet-SSD/'
net = cv2.dnn.readNetFromCaffe('./deploy.prototxt' , './mobilenet_iter_73000.caffemodel')



# load the input image and construct an input blob for the image
# by resizing to a fixed 300x300 pixels and then normalizing it
# (note: normalization is done via the authors of the MobileNet SSD
# implementation)
blob=None
def applySSD(image):
    global blob
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 0.007843, (300, 300), 127.5)
    # pass the blob through the network and obtain the detections and
    # predictions
#     print("[INFO] computing object detections...")
    net.setInput(blob)
    detections = net.forward()
    found = False
    # loop over the detections
    for i in np.arange(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with the
        # prediction
        confidence = detections[0, 0, i, 2]
       

        # filter out weak detections by ensuring the `confidence` is
        # greater than the minimum confidence
        if confidence > confidence_thr:
            # extract the index of the class label from the `detections`,
            # then compute the (x, y)-coordinates of the bounding box for
            # the object
            idx = int(detections[0, 0, i, 1])
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            
            # display the prediction
            label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
            if(CLASSES[idx] == 'bird'):

                found = True
                smallImg = cleanImg[startY:endY,startX:endX]
#             print("[INFO] {}".format(label))
            cv2.rectangle(image, (startX, startY), (endX, endY),
                COLORS[idx], 2)
            y = startY - 15 if startY - 15 > 15 else startY + 15
            cv2.putText(image, label, (startX, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)
    if smallImg is not None:
        return image,smallImg
    return image



# create input blob 
cv2.namedWindow("preview")
rval = True
labelx = 0


if rval: # try to get the first frame
    image = io.imread("./test4.jpg")
    global cleanImg 
    cleanImg = io.imread("./test4.jpg")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    cleanImg = cv2.cvtColor(cleanImg, cv2.COLOR_BGR2RGB)
    (h, w) = image.shape[0] , image.shape[1]
else:
    rval = False





while rval:
     cv2.imshow("preview", image)
     key = cv2.waitKey(20)
     if key == 27: # exit on ESC
        break
     if labelx == 0:
        labelx +=1
        image,croppedImg = applySSD(image)
#    cv2.imshow("preview", frame)
#    key = cv2.waitKey(20)
#    if key == 27: # exit on ESC
#        break
#    frame = applySSD(frame)

while rval:
    cv2.imshow('preview',croppedImg)
    key = cv2.waitKey(20)
    if key == 27: # exit on ESC
        break
croppedImg = cv2.cvtColor(croppedImg, cv2.COLOR_RGB2BGR)
io.imsave('bird-image.jpeg',croppedImg)    
  

cv2.destroyWindow("preview")