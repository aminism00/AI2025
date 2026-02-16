import cv2
import numpy as np
import joblib
from skimage.feature import hog
import matplotlib.pyplot as plt


svm_model = joblib.load("svm_model.pkl")
scaler = joblib.load("svm_scaler.pkl")

image_path = "img3.jpg"   

img = cv2.imread(image_path)

if img is None:
    raise FileNotFoundError("Image not found!")

img_resized = cv2.resize(img, (32, 32))

gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)

hog_features = hog(
    gray,
    orientations=9,
    pixels_per_cell=(8,8),
    cells_per_block=(2,2),
    block_norm='L2-Hys',
    visualize=False
)

hist = cv2.calcHist(
    [img_resized], [0,1,2], None,
    [8,8,8],
    [0,256,0,256,0,256]
)

hist = cv2.normalize(hist, hist).flatten()

feature_vector = np.hstack([hog_features, hist]).reshape(1, -1)

feature_vector_scaled = scaler.transform(feature_vector)


prediction = svm_model.predict(feature_vector_scaled)




classes = { 0:'Speed limit (20km/h)', 1:'Speed limit (30km/h)', 2:'Speed limit (50km/h)',
            3:'Speed limit (60km/h)', 4:'Speed limit (70km/h)', 5:'Speed limit (80km/h)',
            6:'End of speed limit (80km/h)', 7:'Speed limit (100km/h)', 8:'Speed limit (120km/h)',
            9:'No passing', 10:'No passing veh over 3.5 tons', 11:'Right-of-way at intersection',
            12:'Priority road', 13:'Yield', 14:'Stop', 15:'No vehicles',
            16:'Veh > 3.5 tons prohibited', 17:'No entry', 18:'General caution',
            19:'Dangerous curve left', 20:'Dangerous curve right', 21:'Double curve',
            22:'Bumpy road', 23:'Slippery road', 24:'Road narrows on the right',
            25:'Road work', 26:'Traffic signals', 27:'Pedestrians', 28:'Children crossing',
            29:'Bicycles crossing', 30:'Beware of ice/snow', 31:'Wild animals crossing',
            32:'End speed + passing limits', 33:'Turn right ahead', 34:'Turn left ahead',
            35:'Ahead only', 36:'Go straight or right', 37:'Go straight or left',
            38:'Keep right', 39:'Keep left', 40:'Roundabout mandatory',
            41:'End of no passing', 42:'End no passing veh > 3.5 tons' }






print("Predicted Class:", prediction[0])

plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title(f"Predicted Class: {classes[prediction[0]]}")
plt.axis("off")
plt.show()
