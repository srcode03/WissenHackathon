from fastapi import FastAPI, UploadFile, HTTPException, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import logging
import numpy as np
import io
import librosa
from pydub import AudioSegment
import json
from typing import Optional
import re
from gensim.models import Word2Vec
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
import xgboost as xgb

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('tokenizers/punkt_tab')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('punkt_tab')
    nltk.download('stopwords')

app = FastAPI()

# Configuration
FRONTEND_URL = "http://localhost:3000"

# Logger setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=[FRONTEND_URL],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ConfidenceAnalyzer:
    """Handles confidence analysis using the trained XGBoost model."""
    
    def __init__(self, model_path='confidence_model.json'):
        self.model = xgb.XGBClassifier()
        self.model.load_model(model_path)
    
    def extract_features(self, audio_data: np.ndarray, sr: int) -> np.ndarray:
        """Extract features for confidence prediction"""
        try:
            mfcc = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=13)
            rms = librosa.feature.rms(y=audio_data)
            zcr = librosa.feature.zero_crossing_rate(y=audio_data)
            cent = librosa.feature.spectral_centroid(y=audio_data, sr=sr)

            mfcc_mean = np.mean(mfcc, axis=1)
            mfcc_std = np.std(mfcc, axis=1)
            rms_mean = np.mean(rms)
            zcr_mean = np.mean(zcr)
            cent_mean = np.mean(cent)

            # Combine all features
            features = np.hstack([mfcc_mean, mfcc_std, rms_mean, zcr_mean, cent_mean])
            return features

        except Exception as e:
            logger.error(f"Feature extraction failed: {e}")
            return None
    
    def predict_confidence(self, audio_data: np.ndarray, sr: int) -> dict:
        """Predict confidence using the trained model"""
        try:
            features = self.extract_features(audio_data, sr)
            if features is None:
                return {'error': 'Failed to extract features'}

            features = features.reshape(1, -1)
            pred_prob = self.model.predict_proba(features)[0][1]
            pred_class = 1 if pred_prob > 0.5 else 0

            return {
                'class': 'Confident' if pred_class == 1 else 'Not Confident',
                'confidence_score': pred_prob * 100
            }
        except Exception as e:
            logger.error(f"Error in predict_confidence: {str(e)}", exc_info=True)
            return {'error': f'Confidence prediction failed: {str(e)}'}

class TextAnalyzer:
    """Handles text-based analysis for semantic similarity and key point extraction using Word2Vec."""
    
    def __init__(self, vector_size=100, window=5, min_count=1):
        self.stop_words = set(stopwords.words('english'))
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.model = None
    
    def preprocess_text(self, text):
        """Preprocess text by tokenizing and removing stopwords"""
        words = word_tokenize(text.lower())
        return [w for w in words if w.isalnum() and w not in self.stop_words]
    
    def train_word2vec(self, texts):
        """Train Word2Vec model on the provided texts"""
        processed_texts = [self.preprocess_text(text) for text in texts]
        model = Word2Vec(sentences=processed_texts, vector_size=self.vector_size, 
                        window=self.window, min_count=self.min_count, workers=4)
        return model
    
    def get_document_vector(self, text, model):
        """Get document vector by averaging word vectors"""
        words = self.preprocess_text(text)
        word_vectors = [model.wv[word] for word in words if word in model.wv]
        
        if not word_vectors:
            return np.zeros(self.vector_size)
        
        return np.mean(word_vectors, axis=0)
    
    def calculate_semantic_similarity(self, reference_answer: str, candidate_answer: str) -> float:
        """
        Calculate semantic similarity using Word2Vec vectors and cosine similarity.
        """
        try:
            # If both texts are identical, return 1.0
            if reference_answer.strip() == candidate_answer.strip():
                return 1.0
            
            # Train Word2Vec model on the two texts
            self.model = self.train_word2vec([reference_answer, candidate_answer])
            
            # Get document vectors
            ref_vector = self.get_document_vector(reference_answer, self.model)
            cand_vector = self.get_document_vector(candidate_answer, self.model)
            
            # Calculate cosine similarity
            if np.all(ref_vector == 0) or np.all(cand_vector == 0):
                return 0.0
                
            similarity = np.dot(ref_vector, cand_vector) / (np.linalg.norm(ref_vector) * np.linalg.norm(cand_vector))
            return float(similarity)
        except Exception as e:
            logger.error(f"Error calculating semantic similarity: {str(e)}", exc_info=True)
            return 0.0
    
    def extract_key_points(self, text: str, max_points: int = 5) -> list:
        """
        Extract key points from a text using Word2Vec embeddings and sentence importance.
        Returns a list of sentences ranked by importance.
        """
        try:
            # Split into sentences
            sentences = sent_tokenize(text)
            if not sentences or len(sentences) <= 1:
                return [] if not sentences else [{"point": sentences[0], "weight": 10}]
            
            # Train Word2Vec on the text
            if self.model is None:
                self.model = self.train_word2vec([text])
            
            # Get document vector (represents the whole text)
            doc_vector = self.get_document_vector(text, self.model)
            
            # Calculate sentence vectors and their similarity to the document vector
            sentence_scores = []
            for i, sent in enumerate(sentences):
                sent_vector = self.get_document_vector(sent, self.model)
                
                # Skip sentences with zero vectors
                if np.all(sent_vector == 0) or np.all(doc_vector == 0):
                    continue
                
                # Calculate similarity to document vector
                similarity = np.dot(sent_vector, doc_vector) / (np.linalg.norm(sent_vector) * np.linalg.norm(doc_vector))
                
                # Add a length factor to prioritize longer sentences (typically more informative)
                length_factor = min(1.0, len(self.preprocess_text(sent)) / 10)
                adjusted_score = similarity * (0.8 + 0.2 * length_factor)
                
                sentence_scores.append((i, adjusted_score))
            
            # Sort by score and get the top sentences
            top_sentences = sorted(sentence_scores, key=lambda x: x[1], reverse=True)
            
            # Return the top sentences as key points
            key_points = [{"point": sentences[idx], "weight": min(int(score * 10) + 1, 10)} 
                         for idx, score in top_sentences[:max_points]]
            
            return key_points
        except Exception as e:
            logger.error(f"Error extracting key points: {str(e)}", exc_info=True)
            return []
    
    def check_key_points_coverage(self, reference_key_points: list, candidate_answer: str) -> dict:
        """
        Check how many key points from the reference answer are covered in the candidate answer.
        Returns a score and feedback on missing points.
        """
        try:
            score = 0
            max_possible_score = sum(point["weight"] for point in reference_key_points)
            feedback = []
            
            if max_possible_score == 0:
                return {"score": 0, "percentage": 0, "feedback": ["No key points to evaluate"]}
            
            # Ensure we have a model
            if self.model is None:
                points_text = " ".join([point["point"] for point in reference_key_points])
                self.model = self.train_word2vec([points_text, candidate_answer])
            
            for point in reference_key_points:
                # Calculate semantic similarity between point and candidate answer
                point_similarity = self.calculate_semantic_similarity(point["point"], candidate_answer)
                
                # Check if the candidate's answer contains this key point
                if point_similarity > 0.7 or point["point"].lower() in candidate_answer.lower():
                    score += point["weight"]
                    feedback.append(f"✓ Included key point: {point['point']}")
                else:
                    feedback.append(f"✗ Missing key point: {point['point']}")
            
            percentage_score = (score / max_possible_score) * 100
            
            return {
                "score": score,
                "percentage": round(percentage_score, 2),
                "feedback": feedback
            }
        except Exception as e:
            logger.error(f"Error checking key points coverage: {str(e)}", exc_info=True)
            return {"score": 0, "percentage": 0, "feedback": [f"Error: {str(e)}"]}
    
    def analyze_text(self, reference_answer: str, candidate_answer: str) -> dict:
        """
        Perform complete text analysis, combining semantic similarity and key point extraction.
        """
        try:
            # Calculate semantic similarity
            similarity = self.calculate_semantic_similarity(reference_answer, candidate_answer)
            
            # Extract key points from reference answer
            key_points = self.extract_key_points(reference_answer)
            
            # Check key points coverage
            key_point_result = self.check_key_points_coverage(key_points, candidate_answer)
            
            # Generate semantic rating
            semantic_rating = self.get_rating(similarity)
            
            # Detect filler words
            filler_words_count = self.count_filler_words(candidate_answer)
            
            # Calculate overall score (weighted average)
            similarity_weight = 0.4
            key_point_weight = 0.6
            overall_score = (similarity * 100 * similarity_weight) + (key_point_result["percentage"] * key_point_weight)
            
            return {
                "semantic_similarity": round(similarity, 3),
                "semantic_rating": semantic_rating,
                "key_points": key_points,
                "key_point_coverage": key_point_result,
                "filler_words_count": filler_words_count,
                "overall_score": round(overall_score, 2)
            }
        except Exception as e:
            logger.error(f"Error in analyze_text: {str(e)}", exc_info=True)
            return {"error": f"Text analysis failed: {str(e)}"}
    
    def get_rating(self, similarity_score: float) -> str:
        """Convert similarity score to rating."""
        if similarity_score > 0.85:
            return "Excellent"
        elif similarity_score > 0.70:
            return "Good"
        elif similarity_score > 0.50:
            return "Satisfactory"
        else:
            return "Needs Improvement"
    
    def count_filler_words(self, text: str) -> dict:
        """Count filler words in the text and calculate percentage."""
        filler_words = ["um", "uh", "er", "ah", "like", "you know", "basically", "actually", 
                       "literally", "so", "well", "i mean", "kind of", "sort of"]
        
        text_lower = text.lower()
        counts = {}
        
        for word in filler_words:
            count = len(re.findall(r'\b' + re.escape(word) + r'\b', text_lower))
            if count > 0:
                counts[word] = count
        
        total_filler = sum(counts.values())
        total_words = len(re.findall(r'\b\w+\b', text_lower))  # count total words in text

        percentage = (total_filler / total_words * 100) if total_words > 0 else 0

        return {
            "total": round(percentage, 2),
            "details": counts
        }

class SimpleVoiceAnalyzer:
    """A simple voice analyzer using basic signal processing."""
    
    def __init__(self, sample_rate: int = 22050):
        self.sample_rate = sample_rate

    def load_audio(self, audio_bytes: bytes) -> (np.ndarray, int):
        """
        Convert the input WebM audio (as bytes) to a WAV waveform using PyDub
        and load it with librosa.
        """
        try:
            audio_stream = io.BytesIO(audio_bytes)
            # Load the WebM file using PyDub (ensure the file is indeed in WebM format)
            audio_segment = AudioSegment.from_file(audio_stream, format="webm")
            # Export to WAV format in-memory
            wav_io = io.BytesIO()
            audio_segment.export(wav_io, format="wav")
            wav_io.seek(0)
            # Load the WAV data using librosa
            audio, sr = librosa.load(wav_io, sr=self.sample_rate)
            return audio, sr
        except Exception as e:
            logger.error(f"Error in load_audio: {str(e)}", exc_info=True)
            raise

    def analyze_audio(self, audio_data: bytes) -> dict:
        """
        Perform basic analysis (volume, clarity, rhythm) on the raw bytes.
        (This method assumes a raw PCM interpretation; adjust dtype if needed.)
        """
        try:
            # Convert bytes to numpy array of 8-bit integers.
            audio_array = np.frombuffer(audio_data, dtype=np.int8)
            if len(audio_array) == 0:
                return {"error": "Empty audio data"}
            
            # Normalize the audio data to [-1, 1]
            audio_float = audio_array.astype(np.float32) / np.iinfo(np.int8).max
            
            # Calculate basic metrics
            metrics = self._calculate_metrics(audio_float)
            
            return {
                "status": "success",
                "metrics": metrics,
                "feedback": self._generate_feedback(metrics["overall_score"])
            }
            
        except Exception as e:
            logger.error(f"Error in analyze_audio: {str(e)}", exc_info=True)
            return {"error": f"Analysis failed: {str(e)}"}

    def _calculate_metrics(self, audio_data: np.ndarray) -> dict:
        """Calculate various basic audio metrics (volume, clarity, rhythm)."""
        try:
            # Volume (RMS energy)
            rms = np.sqrt(np.mean(np.square(audio_data)))
            volume_score = min(100, rms * 100)

            # Clarity (peak-to-RMS ratio)
            peak = np.max(np.abs(audio_data))
            clarity_score = min(100, (peak / (rms + 1e-6)) * 50)

            # Rhythm (zero crossings rate)
            zero_crossings = np.sum(np.diff(np.signbit(audio_data).astype(int)))
            rhythm_score = min(100, zero_crossings / len(audio_data) * 1000)

            overall_score = np.mean([volume_score, clarity_score, rhythm_score])

            return {
                "overall_score": round(float(overall_score), 2),
                "volume_score": round(float(volume_score), 2),
                "clarity_score": round(float(clarity_score), 2),
                "rhythm_score": round(float(rhythm_score), 2)
            }
        except Exception as e:
            logger.error(f"Error in _calculate_metrics: {str(e)}", exc_info=True)
            raise

    def analyze_pitch(self, audio_bytes: bytes) -> float:
        """
        Analyze pitch characteristics of the audio using librosa.yin.
        Computes the fundamental frequency (F0) over time and calculates
        a stability score based on the coefficient of variation.
        """
        try:
            audio, sr = self.load_audio(audio_bytes)
            # Set minimum and maximum frequencies for pitch detection.
            fmin = 50   # Adjust as needed for your voice recordings.
            fmax = 300
            # Use librosa.yin to estimate the fundamental frequency over time.
            pitch_values = librosa.yin(audio, fmin=fmin, fmax=fmax, sr=sr)
            valid_pitches = pitch_values[pitch_values > 0]
            if valid_pitches.size == 0:
                return 0.0
            pitch_mean = np.mean(valid_pitches)
            pitch_std = np.std(valid_pitches)
            # Compute stability as a percentage: lower variation yields higher score.
            stability_score = max(0, 100 - (pitch_std / pitch_mean * 100))
            return stability_score
        except Exception as e:
            logger.error(f"Error in analyze_pitch: {str(e)}", exc_info=True)
            return 0.0

    def analyze_tone(self, audio_wave: np.ndarray) -> float:
        """Analyze tone quality based on spectral characteristics."""
        try:
            stft = librosa.stft(audio_wave)
            db = librosa.amplitude_to_db(np.abs(stft))
            tone_score = min(100, max(0, np.mean(db) + 100))
            return tone_score
        except Exception as e:
            logger.error(f"Error in analyze_tone: {str(e)}", exc_info=True)
            return 0.0

    def analyze_frequency_variation(self, audio_wave: np.ndarray) -> float:
        """Analyze frequency variation."""
        try:
            spectral_centroids = librosa.feature.spectral_centroid(y=audio_wave, sr=self.sample_rate)[0]
            variation_score = min(100, max(0, 100 - (np.std(spectral_centroids) * 0.1)))
            return variation_score
        except Exception as e:
            logger.error(f"Error in analyze_frequency_variation: {str(e)}", exc_info=True)
            return 0.0

    def analyze_pauses(self, audio_wave: np.ndarray) -> float:
        """Analyze speech pauses based on RMS energy."""
        try:
            rms = librosa.feature.rms(y=audio_wave)[0]
            silence_threshold = np.mean(rms) * 0.5
            pauses = np.sum(rms < silence_threshold) / len(rms)
            pause_score = min(100, max(0, 100 - abs(pauses - 0.2) * 200))
            return pause_score
        except Exception as e:
            logger.error(f"Error in analyze_pauses: {str(e)}", exc_info=True)
            return 0.0

    def analyze_additional_features(self, audio_bytes: bytes) -> dict:
        """
        Analyze additional features (tone, frequency variation, and pauses)
        from the audio waveform.
        """
        try:
            audio_wave, sr = self.load_audio(audio_bytes)
            tone_score = self.analyze_tone(audio_wave)
            freq_variation = self.analyze_frequency_variation(audio_wave)
            pause_score = self.analyze_pauses(audio_wave)
            return {
                "tone_score": round(float(tone_score), 2),
                "frequency_variation": round(float(freq_variation), 2),
                "pause_score": round(float(pause_score), 2)
            }
        except Exception as e:
            logger.error(f"Error in analyze_additional_features: {str(e)}", exc_info=True)
            return {
                "tone_score": 0.0,
                "frequency_variation": 0.0,
                "pause_score": 0.0
            }

    def _generate_feedback(self, score: float) -> str:
        """Generate feedback based on the overall score."""
        if score > 85:
            return "Excellent voice quality! Your speech is clear and well-modulated."
        elif score > 70:
            return "Good voice quality. Minor improvements could be made in clarity and modulation."
        elif score > 50:
            return "Average voice quality. Try speaking more clearly and varying your tone."
        else:
            return "Voice quality needs improvement. Focus on speaking clearly and maintaining consistent volume."

@app.get("/health-check")
async def health_check():
    return {"status": "healthy"}

@app.post("/upload-audio/{userEmail}/{questionId}")
async def upload_audio(
    userEmail: str, 
    questionId: str, 
    audio: UploadFile = File(...),
    capturedAnswer: str = Form(...),  # Now required
    correctAnswer: str = Form(...)   # Now required
):
    try:
        # Convert questionId to int if needed for internal processing
        question_id_int = int(questionId)
        
        # Read the audio data from the uploaded file (in-memory, no file saving)
        contents = await audio.read()
        if not contents:
            raise HTTPException(status_code=400, detail="Empty audio file")
        
        # Create analyzers
        voice_analyzer = SimpleVoiceAnalyzer()
        confidence_analyzer = ConfidenceAnalyzer()
        
        # Get audio data and sample rate
        audio_data, sr = voice_analyzer.load_audio(contents)
        
        # Voice analysis (volume, clarity, rhythm)
        voice_result = voice_analyzer.analyze_audio(contents)
        
        # Compute pitch score
        pitch_score = voice_analyzer.analyze_pitch(contents)
        
        # Compute additional voice features
        additional = voice_analyzer.analyze_additional_features(contents)
        
        # Confidence analysis using the trained model
        confidence_result = confidence_analyzer.predict_confidence(audio_data, sr)
        
        # Merge all voice scores into the metrics dictionary
        voice_metrics = voice_result["metrics"]
        voice_metrics["pitch_score"] = round(float(pitch_score), 2)
        voice_metrics["tone_score"] = additional["tone_score"]
        voice_metrics["frequency_variation"] = additional["frequency_variation"]
        voice_metrics["pause_score"] = additional["pause_score"]
            
        if "error" in voice_result:
            logger.error(f"Voice analysis error: {voice_result['error']}")
            raise HTTPException(status_code=500, detail=voice_result["error"])
        
        # Create text analyzer and process text
        text_analyzer = TextAnalyzer()
        text_analysis = text_analyzer.analyze_text(correctAnswer, capturedAnswer)
        
        # Extract confidence scores
        confidence_score = confidence_result.get('confidence_score', 0) if 'error' not in confidence_result else 0
        confidence_class = confidence_result.get('class', 'Unknown') if 'error' not in confidence_result else 'Unknown'
        
        # Calculate combined metrics
        bleu_score = 0.0  # Placeholder for BLEU score
        
        # Create a filler words dictionary in the required format for your frontend
        filler_words_count = text_analysis.get("filler_words_count", {"total": 0, "details": {}})
        
        # Calculate an overall score combining voice, text, and confidence analysis
        voice_weight = 0.3
        text_weight = 0.4
        confidence_weight = 0.3
        
        overall_score = (
            voice_metrics["overall_score"] * voice_weight + 
            text_analysis.get("overall_score", 0) * text_weight +
            confidence_score * confidence_weight
        )
        
        # Generate comprehensive feedback
        feedback_points = []
        
        # Voice feedback
        feedback_points.append(voice_result["feedback"])
        
        # Confidence feedback
        feedback_points.append(f"Your confidence level is detected as: {confidence_class} ({confidence_score:.1f}%)")
        
        # Semantic similarity feedback
        if "semantic_rating" in text_analysis:
            feedback_points.append(f"Your answer is {text_analysis['semantic_rating'].lower()} in terms of semantic similarity to the expected answer.")
        
        # Key point feedback
        if "key_point_coverage" in text_analysis and "feedback" in text_analysis["key_point_coverage"]:
            feedback_points.extend(text_analysis["key_point_coverage"]["feedback"])
        
        # Filler words feedback
        if filler_words_count["total"] > 0:
            filler_details = ", ".join([f"'{word}' ({count} times)" for word, count in filler_words_count["details"].items()])
            feedback_points.append(f"You used {filler_words_count['total']} filler words: {filler_details}. Try to reduce these in your answers.")
        
        # Prepare response in the format expected by your frontend
        response = {
            "status": "success",
            "metrics": {
                # Voice metrics
                "volume_score": voice_metrics["volume_score"],
                "clarity_score": voice_metrics["clarity_score"],
                "rhythm_score": voice_metrics["rhythm_score"],
                "pitch_score": voice_metrics["pitch_score"],
                "tone_score": voice_metrics["tone_score"],
                "frequency_variation": voice_metrics["frequency_variation"],
                "pause_score": voice_metrics["pause_score"],
                "voice_overall": voice_metrics["overall_score"],
                
                # Confidence metrics
                "confidence_score": confidence_score,
                "confidence_class": confidence_class,
                
                # Text metrics
                "semantic_similarity": text_analysis.get("semantic_similarity", 0) * 100,  # Convert to percentage
                "semantic_rating": text_analysis.get("semantic_rating", "Not Available"),
                "key_point_score": text_analysis.get("key_point_coverage", {}).get("percentage", 0),
                "fillerWordsCount": filler_words_count,
                "bleuScore": bleu_score,  # Placeholder - implement BLEU if needed
                "text_overall": text_analysis.get("overall_score", 0),
                
                # Combined score
                "overall_score": round(overall_score, 2)
            },
            "feedback": feedback_points,
            "voiceMetrics": voice_metrics,  # For backward compatibility
            "textAnalysis": {
                "semanticSimilarity": {
                    "score": text_analysis.get("semantic_similarity", 0) * 100,
                    "rating": text_analysis.get("semantic_rating", "Not Available")
                },
                "keyPointsCoverage": text_analysis.get("key_point_coverage", {})
            }
        }
        
        return JSONResponse(content=response)

    except Exception as e:
        logger.error(f"Error processing audio: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process audio: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("audio:app", host="0.0.0.0", port=8000, reload=True)