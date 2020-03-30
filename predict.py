from werkzeug.wrappers import Request, Response
import io
import numpy
from PIL import Image
import tensorflow as tf
import json

# Location of our model
model_path = 'model.h5'

# Store our model
mnistModel = None

def read_input(request):
    # Ensure that we've received a file named 'image' through POST
    # If we have a valid request proceed, otherwise return None
    if request.method != 'POST' and 'image' not in request.files:
        return None

    # Load the image that was sent
    imageFile = request.files.get('image')
    img = Image.open(imageFile.stream)
    img.load()

    # Resize image to 28x28 and convert to grayscale
    img = img.resize((28, 28)).convert('L')
    img_array = numpy.array(img)

    # We're reshaping the model as our model is expecting 3 dimensions
    # with the first one describing the number of images
    image_data = numpy.reshape(img_array, (1, 28, 28))

    return image_data

def mypredictor(environ, start_response):
    # Get the request object from the environment
    request = Request(environ)

    global mnistModel
    if not mnistModel:
        mnistModel = tf.keras.models.load_model(model_path)

    # Get the image file from our request
    image = read_input(request)

    # If read_input didn't find a valid file
    if (image is None):
        response = Response("\nNo image", content_type='text/html')
        return response(environ, start_response)


    # Use our model to predict the class of the file sent over a form.
    prediction = mnistModel.predict_classes(image)

    # Generate a JSON output with the prediction
    json_response = json.dumps("{Predicted_Digit: %s}" % prediction[0])

    # Send a response back with the prediction
    response = Response(json_response, content_type='application/json')
    return response(environ, start_response)
# When running locally
if __name__ == "__main__":
    from werkzeug.serving import run_simple

    # Update model path to point to our downloaded model when testing locally
    model_path = '.models/model.h5'

    # Run a local server on port 5000.
    run_simple("localhost", 8000, mypredictor)