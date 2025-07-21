import cv2
import numpy as np
import mediapipe as mp
import dlib
from scipy.spatial import distance as dist
import time
import os
import tempfile
from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
# matplotlib.use('Agg')
import pandas as pd
import seaborn as sns
import io
import base64

app = Flask(__name__)
CORS(app)

# Initialize MediaPipe Face Mesh for facial landmarks
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)

# Initialize dlib's face detector and facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat") 

# Load emotion detection model
emotion_classifier = load_model('model.h5')  
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
emotion_colors = ['#FF5252', '#4CAF50', '#FF9800', '#2196F3', '#9C27B0', '#607D8B', '#795548']

# Constants
EAR_THRESHOLD = 0.25 
NO_FACE_THRESHOLD = 90 
LONG_EYE_CLOSURE_THRESHOLD = 90
POSE_HOLD_THRESHOLD = 30 
NORMAL_BLINK_RATE = 15  
BLINK_RATE_PENALTY_THRESHOLD = NORMAL_BLINK_RATE * 2 
BLINK_RATE_PENALTY = 0.5 

def reset_analysis_data():
    """Reset all tracking metrics and data storage for a new analysis session"""
    return {
        "metrics": {
            "blink_count": 0,
            "long_eye_closure_count": 0,
            "no_face_count": 0,
            "emotion_count": {
                "Angry": 0, "Disgust": 0, "Fear": 0, 
                "Happy": 0, "Neutral": 0, "Sad": 0, "Surprise": 0
            },
            "head_movement": {
                "left_look_count": 0,
                "right_look_count": 0,
                "up_look_count": 0,
                "down_look_count": 0,
                "left_lean_count": 0,
                "right_lean_count": 0
            }
        },
        "data_collection": {
            "ear_history": [],
            "emotion_timeline": [],
            "yaw_history": [],
            "pitch_history": [],
            "frame_timestamps": []
        }
    }

def eye_aspect_ratio(eye):
    """Calculate eye aspect ratio for blink detection"""
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

def calculate_face_angle(mesh):
    """Calculate face angles (roll, yaw, pitch) from facial landmarks"""
    if not mesh:
        return {}

    # Extract required landmarks
    nose_tip = mesh[4]       # Nose tip
    forehead = mesh[10]      # Forehead
    left_eye_inner = mesh[133]  # Left eye inner corner
    right_eye_inner = mesh[362] # Right eye inner corner
    chin = mesh[152]         # Chin

    # Calculate vectors for orientation
    h_vec = np.array([right_eye_inner.x - left_eye_inner.x,
                     right_eye_inner.y - left_eye_inner.y,
                     right_eye_inner.z - left_eye_inner.z])
    
    v_vec = np.array([forehead.x - chin.x,
                     forehead.y - chin.y,
                     forehead.z - chin.z])
    
    # Calculate angles
    roll = np.arctan2(h_vec[1], h_vec[0])
    yaw = np.arctan2(h_vec[2], h_vec[0])
    pitch = np.arctan2(v_vec[2], v_vec[1])
    
    return {
        "roll": roll,
        "yaw": yaw,
        "pitch": pitch
    }

def classify_head_pose(angles):
    """Classify head pose based on calculated angles"""
    roll = angles["roll"]
    yaw = angles["yaw"]
    pitch = angles["pitch"]

    # Classify roll (leaning left/right)
    if roll < -0.1:
        roll_status = "Leaning Right"
    elif roll > 0.1:
        roll_status = "Leaning Left"
    else:
        roll_status = "Straight (No Lean)"

    # Classify yaw (turning left/right)
    if yaw < -0.1:
        yaw_status = "Looking Right"
    elif yaw > 0.1:
        yaw_status = "Looking Left"
    else:
        yaw_status = "Straight (No Turn)"

    # Classify pitch (looking up/down)
    if pitch < -0.1:
        pitch_status = "Looking Down"
    elif pitch > 0.1:
        pitch_status = "Looking Up"
    else:
        pitch_status = "Straight (No Tilt)"

    return roll_status, yaw_status, pitch_status

def update_pose_counters(roll_status, yaw_status, pitch_status, analysis_data, current_pose, pose_frames):
    """Update pose counters to track sustained head positions"""
    metrics = analysis_data["metrics"]
    head_movement = metrics["head_movement"]
    
    # Determine current pose
    new_pose = (roll_status, yaw_status, pitch_status)
    
    if new_pose == current_pose:
        pose_frames += 1
        if pose_frames == POSE_HOLD_THRESHOLD:
            # Count the sustained poses
            if current_pose[1] == "Looking Left":
                head_movement["left_look_count"] += 1
            elif current_pose[1] == "Looking Right":
                head_movement["right_look_count"] += 1
                
            if current_pose[2] == "Looking Up":
                head_movement["up_look_count"] += 1
            elif current_pose[2] == "Looking Down":
                head_movement["down_look_count"] += 1
                
            if current_pose[0] == "Leaning Left":
                head_movement["left_lean_count"] += 1
            elif current_pose[0] == "Leaning Right":
                head_movement["right_lean_count"] += 1
    else:
        current_pose = new_pose
        pose_frames = 1
        
    return current_pose, pose_frames

def classify_facial_expression(frame, face_rect, analysis_data):
    """Classify facial expression and update emotion counts"""
    metrics = analysis_data["metrics"]
    
    # Convert dlib rectangle to coordinates
    x, y, w, h = face_rect.left(), face_rect.top(), face_rect.width(), face_rect.height()
    
    # Extract ROI and preprocess
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    roi_gray = gray[y:y+h, x:x+w]
    
    try:
        # Resize to 48x48 as expected by most emotion recognition models
        roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
        
        if np.sum([roi_gray]) == 0:
            return "No Face", None
            
        # Normalize and prepare for prediction
        roi = roi_gray.astype('float') / 255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)
        
        # Make prediction
        prediction = emotion_classifier.predict(roi)[0]
        label = emotion_labels[prediction.argmax()]
        
        # Update emotion count
        metrics["emotion_count"][label] += 1
        
        # Store emotional probabilities if needed
        if len(analysis_data["data_collection"]["emotion_timeline"]) < 1000:
            analysis_data["data_collection"]["emotion_timeline"].append(prediction)
        
        return label, roi
        
    except Exception as e:
        print(f"Expression detection error: {e}")
        return "Neutral", None

def calculate_final_metrics(analysis_data, duration_seconds):
    """Calculate final metrics from the collected data"""
    metrics = analysis_data["metrics"]
    
    # Calculate blink rate
    blink_rate = (metrics["blink_count"] / duration_seconds) * 60 if duration_seconds > 0 else 0
    if blink_rate>20:
        blink_rate=20
    # Calculate positive emotions percentage
    emotion_count = metrics["emotion_count"]
    positive_emotions = emotion_count["Happy"] + emotion_count["Surprise"] + emotion_count["Neutral"]
    total_emotions = sum(emotion_count.values())
    positive_percentage = (positive_emotions / total_emotions) * 100 if total_emotions > 0 else 0
    emotion_percentages = {}
    for emotion, count in emotion_count.items():
        emotion_percentages[emotion] = (count / total_emotions) * 100 if total_emotions > 0 else 0
    # Calculate looking/leaning away counts
    head_movement = metrics["head_movement"]
    total_looking_away = (
        head_movement["left_look_count"] + 
        head_movement["right_look_count"] + 
        head_movement["up_look_count"] + 
        head_movement["down_look_count"]
    )
    total_leaning_away = (
        head_movement["left_lean_count"] + 
        head_movement["right_lean_count"]
    )

    # Calculate interview score (base 10/10)
    interview_score = 10.0
    
    # Emotion percentage deductions
    if positive_percentage >= 80:
        # No deduction for good emotional engagement
        pass
    elif 70 <= positive_percentage < 80:
        interview_score -= 0.3
    elif 50 <= positive_percentage < 70:
        # Scale deduction between 0.5-0.7 based on how close to 50%
        deduction = 0.5 + (0.2 * ((70 - positive_percentage) / 20))
        interview_score -= deduction
    else:
        interview_score -= 1.0
    
    # Other deductions
    interview_score -= metrics["long_eye_closure_count"] * 0.5
    interview_score -= metrics["no_face_count"] * 0.5    
    interview_score -= total_looking_away * 0.5    
    interview_score -= total_leaning_away * 0.3
    
    # Add blink rate deduction if blinking too fast
    if blink_rate > BLINK_RATE_PENALTY_THRESHOLD:
        interview_score -= BLINK_RATE_PENALTY
    
    # Ensure score doesn't go below 0
    interview_score = max(0, interview_score)

    # Generate feedback
    feedback = []

    if total_looking_away > 3:
        feedback.append(f"Reduce looking away from camera (looked away {total_looking_away} times)")
    elif total_looking_away > 0:
        feedback.append(f"Try to maintain better eye contact (looked away {total_looking_away} times)")
    else:
        feedback.append("Excellent eye contact maintained")

    if blink_rate > BLINK_RATE_PENALTY_THRESHOLD:
        feedback.append(f"Reduce blinking rate (current: {blink_rate:.1f}/min, normal: {NORMAL_BLINK_RATE}/min)")
    else:
        feedback.append(f"Good blinking rate ({blink_rate:.1f}/min)")

    if metrics["long_eye_closure_count"] > 0:
        feedback.append(f"Avoid long eye closures (had {metrics['long_eye_closure_count']})")

    if positive_percentage >= 70:
        feedback.append(f"Good emotional engagement ({positive_percentage:.1f}% positive)")
    else:
        feedback.append(f"Try to maintain more positive expressions ({positive_percentage:.1f}% positive)")

    return {
        "blink_rate": round(blink_rate, 1),
        "positive_emotions_percentage": round(positive_percentage, 1),
        "emotion_percentages": {emotion: round(percentage, 1) for emotion, percentage in emotion_percentages.items()},
        "total_looking_away": total_looking_away,
        "total_leaning_away": total_leaning_away,
        "interview_score": round(interview_score, 1),
        "feedback": feedback,
        "raw_metrics": metrics
    }

def generate_charts(analysis_data):
    """Generate charts for visualization and return as base64 encoded images"""
    metrics = analysis_data["metrics"]
    data_collection = analysis_data["data_collection"]
    
    charts = {}
    
    # Overall score and feedback chart
    plt.figure(figsize=(10, 6))
    plt.subplots_adjust(top=0.85)
    plt.title('Interview Performance Score', fontsize=16)
    
    final_metrics = calculate_final_metrics(analysis_data, 1)  # Duration not needed for this chart
    interview_score = final_metrics["interview_score"]
    
    # Create a gauge-like visualization for the score
    plt.axis('equal')
    plt.pie([interview_score, 10-interview_score], 
            colors=['#2196F3', '#E0E0E0'], 
            startangle=90, 
            counterclock=False)
    plt.text(0, 0, f"{interview_score}/10", 
             ha='center', va='center', fontsize=24, fontweight='bold')
    
    # Create circle in the middle to make it look like a gauge
    circle = plt.Circle((0, 0), 0.7, fc='white')
    plt.gca().add_patch(circle)
    
    # Save to buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    charts["score_chart"] = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close()
    
    # Emotion distribution pie chart
    plt.figure(figsize=(10, 6))
    plt.title('Emotion Distribution', fontsize=16)
    
    emotions = []
    counts = []
    colors = []
    
    for emotion, count in metrics["emotion_count"].items():
        if count > 0:
            emotions.append(emotion)
            counts.append(count)
            colors.append(emotion_colors[emotion_labels.index(emotion)])
    
    if counts:
        plt.pie(counts, labels=emotions, autopct='%1.1f%%', colors=colors, startangle=90)
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    charts["emotion_chart"] = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close()
    
    # Head movement bar chart
    plt.figure(figsize=(10, 6))
    plt.title('Head Movement Counts', fontsize=16)
    
    head_movement = metrics["head_movement"]
    movements = ['Left', 'Right', 'Up', 'Down', 'Lean Left', 'Lean Right']
    movement_counts = [
        head_movement["left_look_count"], 
        head_movement["right_look_count"], 
        head_movement["up_look_count"], 
        head_movement["down_look_count"],
        head_movement["left_lean_count"], 
        head_movement["right_lean_count"]
    ]
    
    bars = plt.bar(movements, movement_counts, color='mediumslateblue')
    plt.ylabel("Count")
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.2, int(yval), 
                 ha='center', va='bottom', fontsize=9)
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    charts["head_movement_chart"] = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close()
    
    # Eye aspect ratio chart (if we have data)
    if data_collection["ear_history"]:
        plt.figure(figsize=(10, 4))
        plt.title('Blink Detection (Eye Aspect Ratio)', fontsize=16)
        
        plt.plot(data_collection["ear_history"], label='EAR')
        plt.axhline(EAR_THRESHOLD, color='r', linestyle='--', label='Blink Threshold')
        plt.ylabel('Eye Aspect Ratio')
        plt.xlabel('Frame Number')
        plt.legend()
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        charts["ear_chart"] = base64.b64encode(buf.getvalue()).decode('utf-8')
        plt.close()
    
    # Head orientation charts (if we have data)
    if data_collection["yaw_history"] and data_collection["pitch_history"]:
        plt.figure(figsize=(10, 6))
        plt.suptitle('Head Orientation Over Time', fontsize=16)
        
        # Yaw
        plt.subplot(2, 1, 1)
        plt.plot(data_collection["yaw_history"], label='Yaw (Left/Right)', color='green')
        plt.axhline(0.1, color='gray', linestyle=':')
        plt.axhline(-0.1, color='gray', linestyle=':')
        plt.ylabel('Yaw Angle')
        plt.legend()
        
        # Pitch
        plt.subplot(2, 1, 2)
        plt.plot(data_collection["pitch_history"], label='Pitch (Up/Down)', color='orange')
        plt.axhline(0.1, color='gray', linestyle=':')
        plt.axhline(-0.1, color='gray', linestyle=':')
        plt.ylabel('Pitch Angle')
        plt.xlabel('Frame Number')
        plt.legend()
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        charts["head_orientation_chart"] = base64.b64encode(buf.getvalue()).decode('utf-8')
        plt.close()
    
    return charts

def analyze_video(video_path):
    """Process video and extract metrics for interview performance"""
    # Initialize analysis data storage
    analysis_data = reset_analysis_data()
    metrics = analysis_data["metrics"]
    data_collection = analysis_data["data_collection"]
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Could not open video file")
    
    # Get video FPS for proper timing
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30  # Default assumption if we can't get FPS
    
    start_time = time.time()
    total_frames = 0
    
    # Tracking variables
    cooldown_counter = 0
    eye_closed_frames = 0
    no_face_frames = 0
    current_pose = None
    pose_frames = 0
    eye_closed = False
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        total_frames += 1
        
        # Convert frame to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)
        
        # Detect facial landmarks using dlib
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)
        
        if len(faces) == 0:
            no_face_frames += 1
            if no_face_frames == NO_FACE_THRESHOLD:
                metrics["no_face_count"] += 1
                no_face_frames = 0  # Reset to avoid multiple counts
        else:
            no_face_frames = 0  # Reset counter if face is detected
            
            for face in faces:
                landmarks = predictor(gray, face)
                
                # Extract eye landmarks for EAR calculation
                left_eye = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(36, 42)]
                right_eye = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(42, 48)]
                left_ear = eye_aspect_ratio(left_eye)
                right_ear = eye_aspect_ratio(right_eye)
                avg_ear = (left_ear + right_ear) / 2.0
                
                # Store EAR value (sampled to reduce data size)
                if total_frames % 2 == 0 and len(data_collection["ear_history"]) < 1000:
                    data_collection["ear_history"].append(avg_ear)
                
                # Detect eye blink
                if avg_ear < EAR_THRESHOLD:
                    if not eye_closed and cooldown_counter == 0:
                        metrics["blink_count"] += 1
                        eye_closed = True
                        cooldown_counter = 5  # Start cooldown
                    eye_closed_frames += 1
                    if eye_closed_frames == LONG_EYE_CLOSURE_THRESHOLD:
                        metrics["long_eye_closure_count"] += 1
                        eye_closed_frames = 0  # Reset to avoid multiple counts
                else:
                    eye_closed = False
                    eye_closed_frames = 0
                
                # Decrement cooldown counter
                if cooldown_counter > 0:
                    cooldown_counter -= 1
                
                # Classify facial expression
                expression, _ = classify_facial_expression(frame, face, analysis_data)
                
                # Calculate face angles using MediaPipe Face Mesh
                if results.multi_face_landmarks:
                    for face_landmarks in results.multi_face_landmarks:
                        mesh = face_landmarks.landmark
                        angles = calculate_face_angle(mesh)
                        
                        # Store orientation data (sampled to reduce size)
                        if total_frames % 3 == 0:
                            if len(data_collection["yaw_history"]) < 1000:
                                data_collection["yaw_history"].append(angles.get("yaw", 0))
                            if len(data_collection["pitch_history"]) < 1000:
                                data_collection["pitch_history"].append(angles.get("pitch", 0))
                        
                        roll_status, yaw_status, pitch_status = classify_head_pose(angles)
                        current_pose, pose_frames = update_pose_counters(
                            roll_status, yaw_status, pitch_status, 
                            analysis_data, current_pose, pose_frames
                        )
    
    # Calculate duration and release resources
    duration_seconds = (total_frames / fps) if fps > 0 else (time.time() - start_time)
    cap.release()
    
    # Generate final metrics and charts
    final_metrics = calculate_final_metrics(analysis_data, duration_seconds)
    charts = generate_charts(analysis_data)
    
    return {
        "metrics": final_metrics,
        "charts": charts,
        "duration": round(duration_seconds, 1),
        "frame_count": total_frames
    }

@app.route('/upload-video/<email>/<int:question_index>', methods=['POST'])
def upload_video(email, question_index):
    if 'video' not in request.files:
        return jsonify({"error": "No video file uploaded"}), 400
    
    video_file = request.files['video']
    
    # Optional parameters
    # email = request.form.get('email', 'anonymous')
    # question_index = request.form.get('question_index', 0)
    
    # Create a temporary file to save the uploaded video
    with tempfile.NamedTemporaryFile(delete=False, suffix='.webm') as temp_file:
        video_file.save(temp_file.name)
        temp_file_path = temp_file.name
    
    try:
        # Analyze the video
        print(f"Processing video for {email}, question {question_index}...")
        analysis_result = analyze_video(temp_file_path)
        
        # Clean up temporary file
        os.unlink(temp_file_path)
        
        # Add user info to result
        analysis_result["user_info"] = {
            "email": email,
            "question_index": question_index
        }
        # print("analysis result :-" , jsonify(analysis_result))
        return jsonify({
            "analysis": analysis_result
        })
    
    except Exception as e:
        # Ensure temporary file is deleted even if an error occurs
        if os.path.exists(temp_file_path):
            os.unlink(temp_file_path)
        
        return jsonify({
            "error": str(e)
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Simple health check endpoint"""
    return jsonify({"status": "ok", "message": "Video analysis service is running"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8001, debug=True)