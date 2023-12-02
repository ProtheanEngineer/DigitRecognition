import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import mnist

# Load the trained model
model = load_model('digit_recognition_model.h5')

# Load the MNIST test dataset
(_, _), (test_images, test_labels) = mnist.load_data()

# Normalize pixel values to the range [0, 1]
test_images = test_images / 255.0

# Prompt the user for the number of images to display
num_images_to_show = int(input("Enter the number of images to display and predict: "))

# Assuming 'test_images' and 'test_labels' are your MNIST test data
# If you don't have these, replace them with your actual test data
test_image_indices = np.random.choice(len(test_images), size=num_images_to_show, replace=False)

for test_image_index in test_image_indices:
    # Display the test image
    plt.imshow(test_images[test_image_index], cmap='gray')
    plt.title(f"Actual Label: {test_labels[test_image_index]}")
    plt.show()

    # Make predictions
    predictions = model.predict(np.expand_dims(test_images[test_image_index], axis=0))

    # Interpret predictions
    predicted_digit = np.argmax(predictions)
    print(f'The predicted digit is: {predicted_digit}')
    print("-" * 30)
