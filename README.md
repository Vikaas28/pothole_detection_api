
# Pothole Detection System

AI-powered pothole detection system that uses computer vision to identify potholes in road images. The system provides a hosted REST API for image-based detection, GPS geotagging metadata, and integration with road monitoring applications.

## 🚀 Live API

**Hosted API:** https://vikaas18-vmc.hf.space

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Check API health |
| `/detect/frame` | POST | Detect potholes in an uploaded image |

## ✨ Features

- **AI-Based Detection:** Uses a trained YOLO model to identify potholes in road images.
- **REST API:** Exposes detection functionality through FastAPI.
- **Image Upload:** Accepts images through multipart form data.
- **Confidence Scores:** Returns confidence values for detected objects.
- **Bounding Boxes:** Provides coordinates for detected objects.
- **GPS Metadata:** Accepts latitude and longitude for geotagging.
- **Timestamp Support:** Accepts timestamps for recording detection events.
- **Trip Tracking:** Accepts a trip ID to associate detections with a journey.
- **Cloud Deployment:** Hosted on Hugging Face Spaces.

## 🛠️ Tech Stack

- **Language:** Python
- **API Framework:** FastAPI
- **Computer Vision:** OpenCV
- **AI Model:** YOLO (Ultralytics)
- **Numerical Processing:** NumPy
- **Server:** Uvicorn
- **Deployment:** Hugging Face Spaces
- **API Testing:** Postman

## 📂 Project Structure

```text
pothole_detection_api/
├── app/
│   ├── detector.py
│   └── ...
├── pothole.pt
├── garbage.pt
├── requirements.txt
├── README.md
└── ...
```

*The exact structure may vary depending on the current repository contents.*

## 🔗 API Documentation

Once the API is running, interactive API documentation is available at:

- **Swagger UI:** https://vikaas18-vmc.hf.space/docs
- **OpenAPI Schema:** https://vikaas18-vmc.hf.space/openapi.json

## 💚 Check API Health

### Request

```http
GET https://vikaas18-vmc.hf.space/health
```

You can open this URL in your browser or send a GET request using Postman.

A successful response indicates that the health endpoint is reachable.

## 🔍 Pothole Detection API

The `/detect/frame` endpoint accepts an image and associated metadata.

### Endpoint

```http
POST https://vikaas18-vmc.hf.space/detect/frame
```

### Request Parameters

Send the request as `multipart/form-data`.

| Key | Type | Description |
|---|---|---|
| `image` | File | Road image to analyze |
| `lat` | Number | GPS latitude |
| `lon` | Number | GPS longitude |
| `timestamp` | String | Date and time of the capture |
| `trip_id` | String | Identifier for the trip |

### Example Request Using Python

Install the required package:

```bash
pip install requests
```

Run the following Python script:

```python
import requests

url = "https://vikaas18-vmc.hf.space/detect/frame"

image_path = "road.jpg"

data = {
    "lat": 22.3072,
    "lon": 73.1812,
    "timestamp": "2026-11-25T10:30:00",
    "trip_id": "trip_001"
}

with open(image_path, "rb") as image:
    files = {
        "image": (
            "road.jpg",
            image,
            "image/jpeg"
        )
    }

    response = requests.post(
        url,
        files=files,
        data=data,
        timeout=120
    )

print("Status Code:", response.status_code)
print("Response:")

try:
    print(response.json())
except ValueError:
    print(response.text)
```

Replace `road.jpg` with the path to your test image.

## 🧪 Test Using Postman

1. Open Postman.
2. Create a new request.
3. Select the `POST` method.
4. Enter the endpoint:

   `https://vikaas18-vmc.hf.space/detect/frame`

5. Open the **Body** tab.
6. Select **form-data**.
7. Add the following keys:

| Key | Type | Example |
|---|---|---|
| `image` | File | Select a road image |
| `lat` | Text | `22.3072` |
| `lon` | Text | `73.1812` |
| `timestamp` | Text | `2026-11-25T10:30:00` |
| `trip_id` | Text | `trip_001` |

8. Set the `image` field type to **File** and select your image.
9. Click **Send**.
10. Inspect the HTTP status and JSON response.

**Tip:** Let Postman set the multipart `Content-Type` header automatically so the boundary is generated correctly.

## 📦 Installation and Local Setup

### 1. Clone the Repository

```bash
git clone https://github.com/Vikaas28/pothole_detection_api.git
cd pothole_detection_api
```

### 2. Create a Virtual Environment

Windows:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

Linux/macOS:

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Verify Model Files

Ensure the required model files are available in the locations expected by the application:

- `pothole.pt`
- `garbage.pt`

The model filenames and paths in the code must match the files included in the repository.

### 5. Run the API

If the FastAPI application instance is named `app` in `app/main.py`, run:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

If your entry point differs, adjust the Uvicorn module path accordingly.

The local API will be available at:

- API: http://127.0.0.1:8000
- Swagger UI: http://127.0.0.1:8000/docs

## 🧠 How It Works

1. A client submits a road image to the API.
2. FastAPI receives the uploaded image and request metadata.
3. OpenCV decodes the image into an array.
4. The trained YOLO model performs object detection.
5. The application extracts detected labels, confidence scores, and bounding-box coordinates.
6. The API returns the detection results to the client.

GPS coordinates, timestamps, and trip IDs are accepted as metadata alongside the image.

## 📊 Detection Output

Each detection contains information such as:

| Field | Description |
|---|---|
| `label` | Detected object class |
| `confidence` | Model confidence score |
| `bbox` | Bounding-box coordinates `[x1, y1, x2, y2]` |

Example detection object:

```json
{
  "label": "pothole",
  "confidence": 0.87,
  "bbox": [120, 80, 350, 240]
}
```

*This is an illustrative example, not a live API response. Actual output depends on the model and input image.*

## ⚙️ Configuration

The detection implementation uses a confidence threshold to filter predictions. The current detector configuration uses `conf=0.6`.

You can adjust this threshold in the detector code to balance detection sensitivity and false positives. Validate any change against representative road images before deployment.

## ⚠️ Notes

- Detection accuracy depends on image quality, lighting, camera angle, and the training data.
- A successful HTTP response does not guarantee that potholes will be detected in every image.
- GPS coordinates and trip identifiers are supplied by the client; their accuracy is not independently verified by the model.
- Keep Python dependencies consistent with `requirements.txt`.
- Large model files may require Git LFS for repository storage.

## 🔮 Future Improvements

- Real-time video-stream detection.
- Mobile application integration.
- Road-condition monitoring dashboards.
- Detection history and geospatial visualization.
- Model performance evaluation on diverse road conditions.
- Alerts for frequently reported pothole locations.

## 👨‍💻 Author

**Vikas Mangorora**

- GitHub: [@Vikaas28](https://github.com/Vikaas28)
- Repository: [pothole_detection_api](https://github.com/Vikaas28/pothole_detection_api)

## 📄 License

Add a license to the repository if you intend to distribute or reuse the project publicly.
