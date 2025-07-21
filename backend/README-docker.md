# Backend Docker Usage

## 1. Build the Docker Image

From the backend directory, run:

```
docker build -t mock-interview-backend .
```

## 2. Run the Docker Container

To run the container and access both FastAPI (audio) and Flask (video) services:

```
docker run -it -p 8000:8000 -p 8001:8001 mock-interview-backend
```

## 3. Start the Services

Inside the container, you can start the services manually:

For FastAPI (audio):
```
python audio.py
```

For Flask (video):
```
python video.py
```

You can also run them in separate containers if you want them to run in parallel.

---

**Note:**
- Make sure your model/data files (e.g., `model.h5`, `shape_predictor_68_face_landmarks.dat`) are present in the backend directory before building the image.
- If you want to automate service startup, you can modify the Dockerfile CMD to launch one or both services automatically. 