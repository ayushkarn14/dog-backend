import numpy as np
from flask import Flask, render_template, request, jsonify, render_template_string
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.inception_v3 import preprocess_input
from io import BytesIO
from PIL import Image
from flask_cors import CORS

# Initialize the Flask app
app = Flask(__name__)
CORS(app)
# Define the target size for the model input (for InceptionV3)
TARGET_SIZE = (224, 224)

# Load the trained InceptionV3 model from the .h5 file
model = load_model("model/dog_breed_model.h5")  # Replace with your .h5 file path


# Route to serve the index page
@app.route("/")
def index():
    with open("index.html", "r") as file:
        content = file.read()
    return render_template_string(content)


# Define the prediction route
@app.route("/predict", methods=["POST"])
def predict():
    # Ensure the file exists in the request
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files["file"]

    # Check if the file has a valid filename
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    # Convert the FileStorage object to a PIL Image
    try:
        img = Image.open(BytesIO(file.read()))
    except Exception as e:
        return jsonify({"error": f"Error loading image: {str(e)}"}), 400

    # Resize the image to the target size expected by InceptionV3
    img = img.resize(TARGET_SIZE)

    # Convert the PIL Image to a numpy array
    img_array = image.img_to_array(img)

    # Expand dimensions to match the input shape of the model (batch size of 1)
    img_array = np.expand_dims(img_array, axis=0)

    # Preprocess the image for InceptionV3 (scaling)
    img_array = preprocess_input(img_array)

    # Make the prediction
    predictions = model.predict(img_array)

    # Decode predictions to human-readable format
    # InceptionV3 predictions are class indices, but for a dog breed classifier, you'll need to map to actual dog breeds
    decoded_predictions = predictions.flatten()  # Adjust to your custom breed labels

    nums = decoded_predictions.tolist()
    # Example: if you had a list of dog breeds, you would map the index to a breed name
    # dog_breeds = ['breed1', 'breed2', 'breed3', ...]  # Replace with actual breed names
    # top_prediction = dog_breeds[np.argmax(decoded_predictions)]

    # For simplicity, just return the raw prediction values
    max1_idx = max2_idx = max3_idx = -1

    for i, num in enumerate(nums):
        if max1_idx == -1 or num > nums[max1_idx]:
            max3_idx = max2_idx
            max2_idx = max1_idx
            max1_idx = i
        elif max2_idx == -1 or (num > nums[max2_idx] and num != nums[max1_idx]):
            max3_idx = max2_idx
            max2_idx = i
        elif max3_idx == -1 or (
            num > nums[max3_idx] and num != nums[max1_idx] and num != nums[max2_idx]
        ):
            max3_idx = i

    dog_breeds = [
        "Chihuahua",
        "Japanese_spaniel",
        "Maltese_dog",
        "Pekinese",
        "Shih-Tzu",
        "Blenheim_spaniel",
        "papillon",
        "toy_terrier",
        "Rhodesian_ridgeback",
        "Afghan_hound",
        "basset",
        "beagle",
        "bloodhound",
        "bluetick",
        "black-and-tan_coonhound",
        "Walker_hound",
        "English_foxhound",
        "redbone",
        "borzoi",
        "Irish_wolfhound",
        "Italian_greyhound",
        "whippet",
        "Ibizan_hound",
        "Norwegian_elkhound",
        "otterhound",
        "Saluki",
        "Scottish_deerhound",
        "Weimaraner",
        "Staffordshire_bullterrier",
        "American_Staffordshire_terrier",
        "Bedlington_terrier",
        "Border_terrier",
        "Kerry_blue_terrier",
        "Irish_terrier",
        "Norfolk_terrier",
        "Norwich_terrier",
        "Yorkshire_terrier",
        "wire-haired_fox_terrier",
        "Lakeland_terrier",
        "Sealyham_terrier",
        "Airedale",
        "cairn",
        "Australian_terrier",
        "Dandie_Dinmont",
        "Boston_bull",
        "miniature_schnauzer",
        "giant_schnauzer",
        "standard_schnauzer",
        "Scotch_terrier",
        "Tibetan_terrier",
        "silky_terrier",
        "soft-coated_wheaten_terrier",
        "West_Highland_white_terrier",
        "Lhasa",
        "flat-coated_retriever",
        "curly-coated_retriever",
        "golden_retriever",
        "Labrador_retriever",
        "Chesapeake_Bay_retriever",
        "German_short-haired_pointer",
        "vizsla",
        "English_setter",
        "Irish_setter",
        "Gordon_setter",
        "Brittany_spaniel",
        "clumber",
        "English_springer",
        "Welsh_springer_spaniel",
        "cocker_spaniel",
        "Sussex_spaniel",
        "Irish_water_spaniel",
        "kuvasz",
        "schipperke",
        "groenendael",
        "malinois",
        "briard",
        "kelpie",
        "komondor",
        "Old_English_sheepdog",
        "Shetland_sheepdog",
        "collie",
        "Border_collie",
        "Bouvier_des_Flandres",
        "Rottweiler",
        "German_shepherd",
        "Doberman",
        "miniature_pinscher",
        "Greater_Swiss_Mountain_dog",
        "Bernese_mountain_dog",
        "Appenzeller",
        "EntleBucher",
        "boxer",
        "bull_mastiff",
        "Tibetan_mastiff",
        "French_bulldog",
        "Great_Dane",
        "Saint_Bernard",
        "Eskimo_dog",
        "malamute",
        "Siberian_husky",
        "affenpinscher",
        "basenji",
        "pug",
        "Leonberg",
        "Newfoundland",
        "Great_Pyrenees",
        "Samoyed",
        "Pomeranian",
        "chow",
        "keeshond",
        "Brabancon_griffon",
        "Pembroke",
        "Cardigan",
        "toy_poodle",
        "miniature_poodle",
        "standard_poodle",
        "Mexican_hairless",
        "dingo",
        "dhole",
        "African_hunting_dog",
    ]

    # Example usage

    return jsonify(
        {
            "predictions": [
                dog_breeds[max1_idx],
                dog_breeds[max2_idx],
                dog_breeds[max3_idx],
            ]
        }
    )


# Run the app
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
