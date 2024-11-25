import os
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

# Function to load and preprocess a single image
def load_and_preprocess_image(image_path, target_size=(224, 224)):
    img = image.load_img(image_path, target_size=target_size)
    img_array = image.img_to_array(img) 
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array /= 255.0  # Rescale pixel values
    return img_array

# Function to dynamically extract class labels from directory names
def get_class_labels_from_directory(directory_path):
    class_labels = sorted(os.listdir(directory_path))
    return class_labels

# Function to predict pest name and similar pests
def predict_pest(image_path, model, class_labels, top_n=5):
    img = load_and_preprocess_image(image_path)
    prediction = model.predict(img)
    
    # Get the most likely class
    predicted_class_index = np.argmax(prediction, axis=1)[0]
    confidence = np.max(prediction)
    predicted_pest = class_labels[predicted_class_index]
    
    # Exclude the main prediction from the top N similar pests
    sorted_indices = np.argsort(prediction[0])[::-1]  # Sort predictions in descending order
    top_n_indices = [i for i in sorted_indices if i != predicted_class_index][:top_n]
    top_n_pests = [(class_labels[i], prediction[0][i]) for i in top_n_indices]
    
    # Main prediction based on confidence threshold
    if confidence > 0.5:  # You can adjust the threshold as needed
        print(f"Pest is found to be: {predicted_pest} with confidence {confidence:.2f}")
    else:
        print("Sorry, I don't recognize the species.")
    
    # Display the top N similar predictions with their probabilities (excluding the predicted pest)
    print("\nTop similar pests and their probabilities:")
    for i, (pest_name, prob) in enumerate(top_n_pests, 1):
        print(f"{i}. {pest_name}: {prob * 100:.2f}% confidence")

# Main pipeline function
def run_pest_detection():
    model_path = r"C:\Users\Abhishek\Desktop\github upload\pest\efficientnet_final_retrained_model.h5"
    test_directory =r'D:\pest\test'  # Change this to your test images directory

    # Load the model
    model = load_model(model_path)
    
    # Get class labels from test directory
    class_labels = get_class_labels_from_directory(test_directory)

    # Input loop for user to enter image path
    while True:
        image_path = input("Enter the path to the pest image (or type 'exit' to quit): ")
        if image_path.lower() == 'exit':
            print("Exiting the pest detection pipeline.")
            break
        if not os.path.exists(image_path):
            print("The provided image path does not exist. Please try again.")
            continue

        # Run prediction
        predict_pest(image_path, model, class_labels, top_n=5)

# Run the pest detection pipeline
if __name__ == "__main__":
    run_pest_detection()
