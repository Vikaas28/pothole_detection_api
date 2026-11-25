Pothole Detection System
AI-powered pothole detection web application with GPS geotagging, user
feedback, interactive maps, and analytics dashboard.
Features
Core Features
- ✅ Real-time Pothole Detection: Automatically identifies
  potholes using YOLO AI models
- ✅ GPS-based Geotagging: Associates detected potholes with
  accurate location data
- ✅ Interactive Map Visualization: Displays potholes on a map
  with filtering options
- ✅ Public Reporting: Manual report submission with photos and
  location
- ✅ Severity Assessment: Automatic size and severity
  classification
- ✅ Status Tracking: Track repair status (reported, in progress,
  repaired)
- ✅ User Feedback System: Verify/dispute, upvote/downvote,
  comments, and photo updates
- ✅ Automated Alerts: Notifications for high-priority potholes
- ✅ Data Export & Analytics: CSV export and comprehensive
  analytics dashboard
- ✅ User Account Management: Role-based access (citizen, admin,
  contractor)
- ✅ Mobile & Web Compatible: Responsive design for all devices
Project Structure
pothole-detection-app/
├── backend/
│   ├── api/
│   │   ├── auth.py              # Authentication endpoints
│   │   ├── detection.py         # YOLO inference endpoints
│   │   ├── report.py            # Report CRUD endpoints
│   │   ├── feedback.py          # Feedback endpoints
│   │   └── analytics.py         # Analytics endpoints
│   ├── db/
│   │   ├── models.py            # SQLAlchemy models
│   │   └── database.py          # Database configuration
│   ├── utils/
│   │   ├── auth.py              # Authentication utilities
│   │   ├── geolocation.py       # GPS utilities
│   │   └── notifications.py     # Alert utilities
│   └── main.py                  # FastAPI application
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── DetectionUpload.jsx
│   │   │   ├── MapDashboard.jsx
│   │   │   └── FeedbackPanel.jsx
│   │   ├── pages/
│   │   │   ├── Home.jsx
│   │   │   ├── Login.jsx
│   │   │   ├── UserDashboard.jsx
│   │   │   └── AdminDashboard.jsx
│   │   ├── utils/
│   │   │   ├── api.js
│   │   │   ├── AuthContext.jsx
│   │   │   └── geolocation.js
│   │   └── App.jsx
│   ├── package.json
│   └── vite.config.js
├── database/
│   └── pothole.db
├── app.py
├── best.pt
├── requirements.txt
└── README.md
Hosted AI Detection API
The image-inference service is hosted on Hugging Face Spaces and can be
called independently of the main web application.
- Base URL: https://vikaas18-vmc.hf.space
- Health check: GET https://vikaas18-vmc.hf.space/health
- Frame detection:
  POST https://vikaas18-vmc.hf.space/detect/frame
- Model used by the hosted service: pothole.pt
Frame Detection Request
Send a multipart/form-data POST request to /detect/frame with these
fields:

  Field         Type            Description
  image       File            Road/frame image (JPEG or PNG)
lat         Number/string   Latitude of the frame
lon         Number/string   Longitude of the frame
timestamp   String          Capture timestamp
trip_id     String          Identifier for the trip

Example using Python:
import requests

API_URL = "https://vikaas18-vmc.hf.space/detect/frame"

with open("road_frame.jpg", "rb") as image_file:
    response = requests.post(
        API_URL,
        files={"image": ("road_frame.jpg", image_file, "image/jpeg")},
        data={
            "lat": "22.3072",
            "lon": "73.1812",
            "timestamp": "2026-11-25T12:00:00",
            "trip_id": "demo-trip-001",
        },
        timeout=60,
    )

response.raise_for_status()
print(response.json())
The coordinates and trip details above are examples; replace them with
the actual capture metadata. Inspect the returned JSON for the exact
response fields produced by the current implementation.
Testing the Hosted API
1. Open the health-check URL in a browser to verify that the service
   responds.
2. In Postman, create a POST request to /detect/frame.
3. Select Body → form-data.
4. Add image as a File field and add lat, lon, timestamp,
   and trip_id as text fields.
5. Send a road image and inspect the JSON response.
Note: The Hugging Face Space is a separate hosted inference
service. A model attached to this repository is not automatically
uploaded to the Space; the Space must contain the model file and its
code must load the correct filename.

Setup Instructions
Backend Setup
1. Install Python dependencies:
   pip install -r requirements.txt
2. Initialize database: The database is automatically created on
   first run. To initialize manually:
   from backend.db.database import init_db
   init_db()
3. Run backend server:
   cd backend
   python main.py
   Or:
   uvicorn backend.main:app --reload --port 8000
   Backend: http://localhost:8000
   API documentation: http://localhost:8000/docs
Frontend Setup
1. Install Node.js dependencies:
   cd frontend
   npm install
2. Run the development server:
   npm run dev
   Frontend: http://localhost:3000
3. Build for production:
   npm run build
Usage
For Citizens
1. Register/Login: Create an account or log in.
2. Upload Image: Upload a road image for AI detection.
3. Add Location: Provide GPS coordinates or use browser location.
4. Submit Report: Create a pothole report.
5. View Map: See reported potholes on the interactive map.
6. Provide Feedback: Verify/dispute reports, add comments, and
   vote.
For Administrators
1. Login: Use admin credentials.
2. View Analytics: Access the dashboard and statistics.
3. Update Status: Change pothole repair status.
4. Export Data: Download reports as CSV.
5. Monitor Trends: View charts and report trends.
API Endpoints
Authentication
- POST /api/auth/register - Register new user
- POST /api/auth/login - Login and get token
- GET /api/auth/me - Get current user info
Detection
- POST /api/detection/detect - Detect potholes in an image (requires
  authentication)
Reports
- GET /api/reports - Get all reports (with filters)
- GET /api/reports/{id} - Get a specific report
- PUT /api/reports/{id}/status - Update report status
  (admin/contractor)
- DELETE /api/reports/{id} - Delete report (admin/contractor)
Feedback
- POST /api/feedback/{report_id} - Submit feedback
- POST /api/feedback/{report_id}/photo - Upload updated photo
- GET /api/feedback/{report_id} - Get all feedback for a report
- DELETE /api/feedback/{feedback_id} - Delete own feedback
Analytics (Admin only)
- GET /api/analytics - Get analytics data
- GET /api/analytics/export/csv - Export reports as CSV
Database Schema
Users
- id, username, email, hashed_password, full_name, role, is_active,
  created_at
PotholeReports
- id, reporter_id, image_path, output_image_path, detections_count,
  latitude, longitude, address, description, severity, status,
  detection_data, statistics, material_estimate, verification_count,
  dispute_count, upvote_count, downvote_count, created_at, updated_at
Feedbacks
- id, report_id, user_id, feedback_type, comment, upvote, downvote,
  updated_photo_path, created_at
StatusUpdates
- id, report_id, updated_by_id, old_status, new_status, notes,
  created_at
Technology Stack
Backend
- FastAPI: Modern Python web framework
- SQLAlchemy: ORM for database operations
- SQLite: Database (can be migrated to PostgreSQL)
- YOLO (Ultralytics): AI model for pothole detection
- JWT: Authentication tokens
- bcrypt: Password hashing
Frontend
- React 18: UI framework
- Material-UI (MUI): Component library
- React Router: Routing
- Leaflet: Interactive maps
- Recharts: Data visualization
- Axios: HTTP client
- Vite: Build tool
Environment Variables
Create a .env file for production:
SECRET_KEY=your-secret-key-here
DATABASE_URL=sqlite:///./database/pothole.db
MODEL_PATH=./best.pt
Security Notes
- Change SECRET_KEY in backend/utils/auth.py for production.
- Use environment variables for sensitive data.
- Implement rate limiting for production.
- Use HTTPS in production.
- Consider migrating to PostgreSQL for production.
Contributing
1. Fork the repository.
2. Create a feature branch.
3. Make your changes.
4. Submit a pull request.
License
MIT License
Support
For issues or questions, please open an issue on GitHub.
