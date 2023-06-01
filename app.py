import numpy as np
# import mnist
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from keras.utils import to_categorical
from keras.optimizers import SGD
from gtts import gTTS
from IPython.display import Audio
from num2words import num2words
import keras.models as models
import keras.utils as image
from googletrans import Translator
# import keras_cv
import matplotlib.pyplot as plt
import cv2
import pygame
import time
import os
# from playsound import playsound

model = models.load_model('mymodel')
translator = Translator()
'''
Epoch 1
46s 760us/step - loss: 0.2433 - acc: 0.9276 - val_loss: 0.1176 - val_acc: 0.9634
Epoch 2
46s 771us/step - loss: 0.1184 - acc: 0.9648 - val_loss: 0.0936 - val_acc: 0.9721
Epoch 3
48s 797us/step - loss: 0.0930 - acc: 0.9721 - val_loss: 0.0778 - val_acc: 0.9744
'''
# Test my image
print('\n--- Testing the CNN with my image ---')
"""
# Read the image using opencv to make my life easy
img = cv2.imread('./numbers.jpg', 0)
img = cv2.resize(img, (28,28)) # Resize - important! 
img = cv2.bitwise_not(img)
img = (img / 255) - 0.5  # The other version does this in the convolution forward() function

# Try to use the image.load_img() function from keras utils as well
#img = image.load_img('images_mine/eight.bmp', color_mode = "grayscale", target_size=(28, 28))
#to_grayscale = keras_cv.layers.preprocessing.Grayscale()
#img = to_grayscale(img)
#...

img_tensor = np.expand_dims(img, axis=0)

prediction = model.predict(img_tensor)
print(prediction)
classes=np.argmax(prediction,axis=1)
print(classes)
num = classes[0]
nums = num2words(num)
print(num2words(num))
tts = gTTS(nums)
# Save the speech as an audio file
tts.save("output.mp3")
# Play the audio in the notebook
Audio("./output.mp3")

# plt.imshow(img_tensor[0], cmap=plt.get_cmap('gray_r'))
# plt.show()
"""
import tkinter as tk
import tkinter.filedialog

def upload_image():
    # Get the image path from the user
    pygame.mixer.init()
    image_path = tk.filedialog.askopenfilename()

    # Read the image
    img = cv2.imread(image_path, 0)

    # Resize the image
    img = cv2.resize(img, (28,28)) # Resize - important! 
    img = cv2.bitwise_not(img)
    img = (img / 255) - 0.5  # The other version does this in the convolution forward() function

    # Convert the image to a tensor
    img_tensor = np.expand_dims(img, axis=0)

    # Make the prediction
    prediction = model.predict(img_tensor)

    # Get the class label
    classes=np.argmax(prediction,axis=1)
    print(classes)
    num = classes[0]
    numss = num2words(num)
    nums = translator.translate(numss, dest="fr").text
    # print(num2words(num))
    tts = gTTS(nums, lang='fr')
    # Save the speech as an audio file
    tts.save("./output.mp3")
    # Play the audio in the notebook
    # Audio("./output.mp3")
    mp3_file_path = "./output.mp3"
    pygame.mixer.music.load(mp3_file_path)
    pygame.mixer.music.play()
    time.sleep(5)
    pygame.quit()
    if os.path.exists(mp3_file_path):
        # Delete the file
        os.remove(mp3_file_path)
        print(f"File '{mp3_file_path}' has been deleted.")
    else:
        print(f"File '{mp3_file_path}' does not exist.")


root = tk.Tk()
root.geometry("1600x900")
root.title('GeeksforGeeks sound player')
# Create a button to upload image
upload_button = tk.Button(root, text="Upload Image", command=upload_image())
upload_button.pack()

# Create a label to display the output audio
output_label = tk.Label(root)
output_label.pack()

# Create a function to upload images


