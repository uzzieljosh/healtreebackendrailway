from flask import Flask, request, send_file, jsonify
from PIL import Image
from werkzeug.utils import secure_filename
import os
from ultralytics import YOLO
import base64

app = Flask(__name__)

# Load the YOLOv8 model
model = YOLO('Tilliabest.pt')

other_model = ['Cherrybest.pt', 'fagusbest.pt', 'Quercusbest.pt', 'Thujabest.pt']

otherclass =[{0:'Cherry_healthytrunk',1:'Cherry_gummosis',2:'Cherry_healthyleaf'},
             {0:'Fagussylvatica_healthyleaf',1:'Fagussylvatica_leafscorch',2:'Fagussylvatica_leafspot'},
             {0:'Quercuspetraea_gall',1:'Quercuspetraea_healthyleaf',2:'Quercuspetraea_healthytrunk',3:'Quercuspetraea_leafhole',4:'Quercuspetraea_leafscorch',5:'Quercuspetraea_leafspot',6:'Quercuspetraea_resinosis',7:'Quercuspetraea_shothole',8:'Quercuspetraea_wooddecay'},
             {0:'Thujaoccidentalis_healthy',1:'Thujaoccidentalis_needleblight'}]

othername = ['Cherry','Fagus sylvatica','Quercus petraea','Thuja occidentalis']

@app.route("/")
def home():
    return "Hello, Flask!"

@app.route("/detect", methods=["POST"])
def detect():
    try:
        conf = 0
        output = "None"
        # Check if the post request has the file part
        if 'image' not in request.files:
            return 'No image file provided', 400
        
        if not os.path.exists('images'):
            os.makedirs('images')
        
        if not os.path.exists('saved_images'):
            os.makedirs('saved_images')

        file = request.files['image']
        filename = secure_filename(file.filename)
        file_path = os.path.join('images', filename)
        file.save(file_path)

        # Save the received image to a specific location
        save_path = os.path.join('saved_images', filename)
        file.save(save_path)
        
        results = model(file_path)

        # Get the first result
        result = results[0]

        class_names = {
            0: 'Tilliacordata_healthy',
            1: 'Tilliacordata_leafscorch',
            2: 'Tilliacordata_leafspot',
            3: 'Tilliacordata_shothole',
        }
        classes = [class_names[int(x)] for x in result.boxes.cls.tolist()]
        confidences = result.boxes.conf.tolist()

        if confidences:
            conf = confidences[0]
            output = "Tillia cordata"

        for i in range(len(other_model)):
            model2 = YOLO(other_model[i])
            results = model2(file_path)
            result2 = results[0]
            confidences2 = result2.boxes.conf.tolist()

            if(confidences2 and confidences2[0] > conf):
                result = result2
                classes = [otherclass[i][int(x)] for x in result.boxes.cls.tolist()]
                confidences = result.boxes.conf.tolist()
                output = othername[i]
                conf = confidences[0]

        # Plot the result and save it to a temporary file
        im_array = result.plot()
        im = Image.fromarray(im_array[..., ::-1])  # Convert BGR to RGB
        temp_result = "temp_result.jpg"
        im.save(temp_result)

        # Convert the image to a base64 string
        with open(temp_result, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')

        # Return the processed image and the console output as a response
        return jsonify({
            'image': encoded_string,
            'output': output,
            'classes': classes,
            'confidences': confidences
        })

    except Exception as e:
        print("error")
        print(str(e))
        return str(e)

if __name__ == "__main__":
    app.run(debug=True)