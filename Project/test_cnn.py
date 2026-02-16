from tensorflow.keras.models import load_model
import numpy as np
import cv2
import matplotlib.pyplot as plt

model = load_model("MyCNNModel.h5")



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





def predict_image(image_path):
    img = cv2.imread(image_path)

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (30, 30))
    img_normalized = img_resized.astype("float32") / 255.0
    img_input = np.expand_dims(img_normalized, axis=0)
 
    prediction = model.predict(img_input)
    class_id = np.argmax(prediction)
    confidence = np.max(prediction)

    plt.figure(figsize=(6,6))
    plt.imshow(img_rgb)
    plt.title(f"Prediction: {classes[class_id]} \n Confidence: {confidence*100:.2f}%")
    plt.axis("off")
    plt.show()

predict_image("img3.jpg")

