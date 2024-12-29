from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.preprocessing.image import img_to_array
import tensorflow as tf
from PIL import Image
import numpy as np
from fastapi.templating import Jinja2Templates
import io
import base64

# FastAPI application
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load TensorFlow Lite model
MODEL_PATH = "model/model.tflite"
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

# Get input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Class labels
class_labels = [
    "type_dog", "type_cat", "type_human", "gender_female",
    "gender_male", "hair_short", "hair_long", "hair_light", "hair_dark"
]

# Set up templates
templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Render the home page with a file upload form."""
    return templates.TemplateResponse("index.html", {"request": request, "image_url": None, "predictions": None})


@app.post("/")
async def predict_home(request: Request, file: UploadFile = File(...)):
    """Handle file upload and process the image in memory."""
    try:
        # Read the uploaded file in memory
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data)).convert("RGB")

        # Preprocess the image
        image = image.resize((224, 224))
        image_array = img_to_array(image) / 255.0
        image_array = np.expand_dims(image_array, axis=0).astype(np.float32)

        # Run the model
        interpreter.set_tensor(input_details[0]['index'], image_array)
        interpreter.invoke()
        predictions = interpreter.get_tensor(output_details[0]['index'])[0]

        # Map predictions to class labels
        results = {class_labels[i]: float(predictions[i]) for i in range(len(class_labels))}

        # Extract predictions for type, gender, and hair attributes
        type_predictions = {k: v for k, v in results.items() if k.startswith("type_")}
        gender_predictions = {k: v for k, v in results.items() if k.startswith("gender_")}
        hair_length_predictions = {k: v for k, v in results.items() if k.startswith("hair_") and ("short" in k or "long" in k)}
        hair_color_predictions = {k: v for k, v in results.items() if k.startswith("hair_") and ("light" in k or "dark" in k)}

        # Determine the strongest predictions
        type_result = max(type_predictions, key=type_predictions.get).split("_")[1]
        if type_result == "human":
            gender_result = max(gender_predictions, key=gender_predictions.get).split("_")[1]
            hair_length_result = max(hair_length_predictions, key=hair_length_predictions.get).split("_")[1]
            hair_color_result = max(hair_color_predictions, key=hair_color_predictions.get).split("_")[1]
            predictions = {
                "type": type_result,
                "gender": gender_result,
                "hair_length": hair_length_result,
                "hair_color": hair_color_result
            }
        else:
            predictions = {"type": type_result}

        # Convert the in-memory image to a Base64 string for rendering
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        image_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

        # Render the page with the uploaded image and predictions
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "image_url": f"data:image/jpeg;base64,{image_base64}",
                "predictions": predictions
            }
        )
    except Exception as e:
        return JSONResponse(content={"error": "Could not process the image."}, status_code=500)
