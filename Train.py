
from sklearn.model_selection import train_test_split

from imutils import paths
import numpy as np
import random
import cv2
import os
from sklearn.preprocessing import LabelEncoder
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import mahotas
import cv2
import os
# Converting each image to RGB from BGR format
bins= 8
def Convert_to_bgr(image):
    rgb_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return rgb_img


# Conversion to HSV image format from RGB

def Convert_to_hsv(rgb_img):
    hsv_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2HSV)
    return hsv_img



# image segmentation

# for extraction of green and brown color


def img_segmentation(rgb_img,hsv_img):
    lower_green = np.array([25,0,20])
    upper_green = np.array([100,255,255])
    healthy_mask = cv2.inRange(hsv_img, lower_green, upper_green)
    result = cv2.bitwise_and(rgb_img,rgb_img, mask=healthy_mask)
    lower_brown = np.array([10,0,10])
    upper_brown = np.array([30,255,255])
    disease_mask = cv2.inRange(hsv_img, lower_brown, upper_brown)
    disease_result = cv2.bitwise_and(rgb_img, rgb_img, mask=disease_mask)
    final_mask = healthy_mask + disease_mask
    final_result = cv2.bitwise_and(rgb_img, rgb_img, mask=final_mask)
    return final_result




# feature-descriptor-1: Hu Moments
def get_shape_feats(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    feature = cv2.HuMoments(cv2.moments(image)).flatten()
    return feature
# feature-descriptor-2: Haralick Texture
def get_texture_feats(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    haralick = mahotas.features.haralick(gray).mean(axis=0)
    return haralick
# feature-descriptor-3: Color Histogram
def get_color_feats(image, mask=None):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist  = cv2.calcHist([image], [0, 1, 2], None, [bins, bins, bins], [0, 256, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()


data = []
labels = []
print("[INFO] loading images...")
img_dir=sorted(list(paths.list_images("Medicinal Leaf Dataset/Segmented Medicinal Leaf Images")))
random.shuffle(img_dir)
tot=len(img_dir)
print("total-->",tot)
print("[INFO]  Preprocessing...")
cnt=1
for i in img_dir:
        image = cv2.imread(i)
        image = cv2.resize(image, (500, 500))

        bgrim       = Convert_to_bgr(image)
      
        hsvim       = Convert_to_hsv(bgrim)
  
        seg_image   = img_segmentation(bgrim,hsvim)
        


        f_shape = get_shape_feats(seg_image)
        f_text   = get_texture_feats(seg_image)
        f_color  = get_color_feats(seg_image)

        # Concatenate 

        f_combined = np.hstack([f_color, f_text, f_shape])


        lab=i.split(os.path.sep)[-2]
        labels.append(lab)
        data.append(f_combined)
        print("image processed-->",str(cnt),"/",str(tot))
        cnt+=1
        
print(len(data))
print(len(labels))
pickle.dump(data,open("data_all.pkl",'wb'))
pickle.dump(labels,open("labels_all.pkl",'wb'))



data=pickle.load(open("data_all.pkl",'rb'))
data=np.array(data)

scaler = MinMaxScaler()
data=scaler.fit_transform(data)
pickle.dump(scaler,open("scaler_all.pkl",'wb'))
print(data.shape)
labels=pickle.load(open("labels_all.pkl",'rb'))
le=LabelEncoder()
labels=le.fit_transform(labels)
pickle.dump(le,open("leenc_all.pkl",'wb'))
print(set(labels))
print(len(set(labels)))


# print("[INFO] Splitting Datas...")
(trainX, testX, trainY, testY) = train_test_split(data,labels, test_size=0.25, random_state=42)

from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.svm import LinearSVC
import pickle
# Building a Support Vector Machine on train data


print("Training started")
svc_model = LinearSVC()
svc_model.fit(trainX, trainY)
print("Training compleded")
pickle.dump(svc_model,open('svm_model_all.pkl','wb'))
print("model saved")
prediction = svc_model .predict(testX)
# check the accuracy on the training set
acscore=accuracy_score(testY,prediction)
print(" LinearSVC acscore-->",acscore)

