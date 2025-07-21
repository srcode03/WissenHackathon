import React, { useState, useRef, useEffect } from "react";
import Webcam from "react-webcam";
import { Button } from "../../../../../../components/ui/button";
import { useUser } from "@clerk/nextjs";
import { db } from "../../../../../../utils/db";
import { UserAnswer } from "../../../../../../utils/schema";
import { useParams } from "next/navigation";
import useSpeechToText from "react-hook-speech-to-text";
import { Mic, StopCircle, Video, VideoOff } from "lucide-react";
import { toast } from "sonner";
import { chatSession } from "../../../../../../utils/GeminiAIModal";
import moment from "moment";
import { eq, and } from "drizzle-orm";

function RecordQuestionSection({
  mockInterviewQuestion = [],
  activeQuestionIndex = 0,
  interviewData,
}) {
  const [recording, setRecording] = useState(false);
  const [videoRecording, setVideoRecording] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [analysisResult, setAnalysisResult] = useState(null);
  const [userAnswer, setUserAnswer] = useState("");
  const [isSubmitted, setIsSubmitted] = useState(false);
  const mediaRecorderRef = useRef(null);
  const videoRecorderRef = useRef(null);
  const webcamRef = useRef(null);
  const chunksRef = useRef([]);
  const videoChunksRef = useRef([]);
  const finalAnswerRef = useRef("");
  const { user } = useUser();
  const params = useParams();

  const {
    error: speechError,
    interimResult,
    isRecording,
    results,
    startSpeechToText,
    stopSpeechToText,
    setResults,
  } = useSpeechToText({
    continuous: true,
    useLegacyResults: false,
    speechRecognitionProperties: {
      lang: "en-IN",
    },
  });

  // Listen for tab changes using the Page Visibility API.
  useEffect(() => {
    const handleVisibilityChange = () => {
      if (document.hidden) {
        toast("Tab change detected. Please remain on this page during the interview.");
        // Optionally, you can automatically stop recording if desired.
        if (recording) {
          stopRecording();
        }
        if (videoRecording) {
          stopVideoRecording();
        }
      }
    };

    document.addEventListener("visibilitychange", handleVisibilityChange);
    return () => {
      document.removeEventListener("visibilitychange", handleVisibilityChange);
    };
  }, [recording, videoRecording]);

  useEffect(() => {
    if (!isSubmitted) {
      const newTranscript = results.map(result => result?.transcript || "").join(" ");
      if (newTranscript) {
        setUserAnswer(prevAnswer => {
          const updatedAnswer = prevAnswer + " " + newTranscript;
          finalAnswerRef.current = updatedAnswer.trim();
          return updatedAnswer;
        });
      }
    }
  }, [results, isSubmitted]);

  const startRecording = async () => {
    try {
      setIsSubmitted(false);
      setError(null);
      setAnalysisResult(null);
      setUserAnswer("");
      finalAnswerRef.current = "";
      
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });

      const mediaRecorder = new MediaRecorder(stream, {
        mimeType: "audio/webm;codecs=opus",
      });

      mediaRecorderRef.current = mediaRecorder;
      chunksRef.current = [];

      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          chunksRef.current.push(event.data);
        }
      };

      mediaRecorder.onstop = async () => {
        try {
          const audioBlob = new Blob(chunksRef.current, {
            type: "audio/webm;codecs=opus",
          });
          await sendAudio(audioBlob);
        } finally {
          stream.getTracks().forEach((track) => track.stop());
        }
      };

      mediaRecorder.start();
      setRecording(true);
      startSpeechToText();
      startVideoRecording(); // Start video recording alongside audio
    } catch (error) {
      setError(`Error accessing microphone: ${error.message}`);
      console.error("Error accessing microphone", error);
    }
  };

  const startVideoRecording = async () => {
    try {
      if (!webcamRef.current || !webcamRef.current.video) {
        throw new Error("Webcam not initialized");
      }

      const stream = webcamRef.current.video.srcObject;
      if (!stream) {
        throw new Error("No webcam stream available");
      }

      const videoRecorder = new MediaRecorder(stream, {
        mimeType: "video/webm;codecs=vp9",
      });

      videoRecorderRef.current = videoRecorder;
      videoChunksRef.current = [];

      videoRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          videoChunksRef.current.push(event.data);
        }
      };

      videoRecorder.onstop = async () => {
        try {
          const videoBlob = new Blob(videoChunksRef.current, {
            type: "video/webm;codecs=vp9",
          });
          await sendVideo(videoBlob);
        } catch (error) {
          console.error("Error processing video:", error);
        }
      };

      videoRecorder.start(1000); // Collect data every second
      setVideoRecording(true);
      toast.success("Video recording started");
    } catch (error) {
      setError(`Error recording video: ${error.message}`);
      console.error("Error recording video", error);
    }
  };

  const stopVideoRecording = () => {
    if (videoRecorderRef.current && videoRecording) {
      videoRecorderRef.current.stop();
      setVideoRecording(false);
      toast.success("Video recording stopped");
    }
  };

  const stopRecording = async () => {
    if (mediaRecorderRef.current && recording) {
      setIsSubmitted(true);
      const capturedAnswer = finalAnswerRef.current || userAnswer;
      console.log("Final captured answer:", capturedAnswer);
      
      mediaRecorderRef.current.stop();
      setRecording(false);
      await stopSpeechToText();
      
      if (capturedAnswer) {
        finalAnswerRef.current = capturedAnswer;
      }

      // Stop video recording
      stopVideoRecording();
    }
  };

  const sendVideo = async (blob) => {
    try {
      if (!blob || blob.size === 0) {
        throw new Error("No video data recorded");
      }
  
      const capturedAnswer = finalAnswerRef.current || userAnswer;
      console.log("Sending video to server with transcription:", capturedAnswer);
  
      const formData = new FormData();
      formData.append("video", blob, "recording.webm");
      formData.append("transcription", capturedAnswer);
  
      const userEmail = user?.primaryEmailAddress?.emailAddress;
      if (!userEmail) {
        throw new Error("User email not found");
      }
  
      toast.info("Analyzing eye movements...");
  
      const response = await fetch(
        `http://localhost:8001/upload-video/${encodeURIComponent(
          userEmail
        )}/${activeQuestionIndex}`,
        {
          method: "POST",
          body: formData,
        }
      );
  
      if (!response.ok) {
        const errorData = await response.json().catch(() => null);
        throw new Error(errorData?.detail || `Video server error: ${response.status}`);
      }
      
      const videoAnalysisData = await response.json();
      // print("video response", videoAnalysisData)
      const videoFeedback = `Take this video analysis data :- ${JSON.stringify(videoAnalysisData)} and i want you to give feedback to the student's video who just gave the interview.Give an encouraging feeback with keys to improve.Remove greetings.`;
      const result = await chatSession .sendMessage(videoFeedback,{
        generation_config:{temperature:0.0},
      });
      const mockJsonResp = result.response
        .text()
        .replace("```json", "")
        .replace("```", "");
        console.log(mockJsonResp)
      // console.log("Eye movement analysis results:", videoAnalysisData.analysis);
      videoAnalysisData.analysis.feedback=mockJsonResp;
      console.log("Eye movement analysis results:", videoAnalysisData.analysis);
      
      
      
      // Update the existing database record with video analysis metrics
      await db.update(UserAnswer).set({
        videoMetrics: videoAnalysisData.analysis, // Directly store the JSON object
      })
      .where(
        eq(UserAnswer.mockIdRef, params.interviewId) // Use the interview ID from URL params
      );
      
      toast.success("Eye movement analysis complete");
      
      // Update analysis result state
      setAnalysisResult(prevResults => ({
        ...prevResults,
        videoMetrics: videoAnalysisData.analysis
      }));
      
    } catch (error) {
      console.error("Error sending video:", error);
      toast.error(`Video analysis failed: ${error.message}`);
    }
  };

  const sendAudio = async (blob) => {
    setLoading(true);
    setError(null);
  
    try {
      if (!blob || blob.size === 0) {
        throw new Error("No audio data recorded");
      }
  
      const capturedAnswer = finalAnswerRef.current || userAnswer;
      console.log("Answer being sent to server:", capturedAnswer);
  
      if (!capturedAnswer) {
        throw new Error("No answer was recorded");
      }
  
      // Get the correct answer for the current question
      const correctAnswer = 
        mockInterviewQuestion[activeQuestionIndex].answerExample ||
        mockInterviewQuestion[activeQuestionIndex].answer;
  
      const formData = new FormData();
      formData.append("audio", blob, "recording.webm");
      formData.append("capturedAnswer", capturedAnswer);
      formData.append("correctAnswer", correctAnswer);
  
      const userEmail = user?.primaryEmailAddress?.emailAddress;
      if (!userEmail) {
        throw new Error("User email not found");
      }
  
      // No changes needed here - you're already sending capturedAnswer and correctAnswer 
      // in the FormData which is correct
      console.log(`http://localhost:8000/upload-audio/${encodeURIComponent(
          userEmail
        )}/${activeQuestionIndex}`)
      const response = await fetch(
        `http://localhost:8000/upload-audio/${encodeURIComponent(
          userEmail
        )}/${activeQuestionIndex}`,
        {
          method: "POST",
          body: formData,
        }
      );
  
      if (!response.ok) {
        const errorData = await response.json().catch(() => null);
        throw new Error(errorData?.detail || `Server error: ${response.status}`);
      }
      const data = await response.json();
      console.log("dataaa",data)
      const audioFeedback = `Take this audio analysis data :- ${JSON.stringify(data)} and i want you to give feedback to the student's audio who just gave the interview.Give an encouraging feeback with keys to improve.Remove greetings.`;
      const result = await chatSession .sendMessage(audioFeedback,{
        generation_config:{temperature:0.0},
      });
      const mockJsonResp = result.response
        .text()
        .replace("```json", "")
        .replace("```", "");
      data.metrics.feedback=mockJsonResp
      console.log("data",data);
      setAnalysisResult(data);
  
      await updateUserAnswer(data.metrics, capturedAnswer);
    } catch (error) {
      setError(`Failed to process recording: ${error.message}`);
      console.error("Error sending audio:", error);
    } finally {
      setLoading(false);
    }
  };
  
  const updateUserAnswer = async (voiceMetrics = null, capturedAnswer) => {
    if (!capturedAnswer) {
      setError("No answer to submit");
      return;
    }

    setLoading(true);
    try {
      if (!mockInterviewQuestion[activeQuestionIndex]) {
        throw new Error("Invalid question index");
      }

      console.log("Saving answer to DB:", capturedAnswer);

      const feedbackPrompt = `Question: ${mockInterviewQuestion[activeQuestionIndex]?.question}, 
        User Answer: ${capturedAnswer}. Please provide a rating out of 100 and feedback in JSON format with fields: "rating" and "feedback".Please dont look for punctuation mistakes!Go easy on ratings.give good rating`;

      const result = await chatSession .sendMessage(feedbackPrompt,{
        generation_config:{temperature:0.0},
      });
      const mockJsonResp = result.response
        .text()
        .replace("```json", "")
        .replace("```", "");
      const JsonFeedbackResp = JSON.parse(mockJsonResp);
      console.log("answer hai ", mockInterviewQuestion);
      const correctAnswer =
        mockInterviewQuestion[activeQuestionIndex].answerExample ||
        mockInterviewQuestion[activeQuestionIndex].answer;

      const nlpResponse = await fetch('/api/process-text', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          reference: correctAnswer,
          candidate: capturedAnswer
        }),
      });
  
      const nlpData = await nlpResponse.json();
      if (!nlpData.success) {
        throw new Error(nlpData.error);
      }
      console.log("nlpdata",nlpData)
      await db
        .update(UserAnswer)
        .set({
          question: mockInterviewQuestion[activeQuestionIndex]?.question,
          userAns: capturedAnswer,
          correctAns: correctAnswer,
          feedback: JsonFeedbackResp?.feedback,
          rating: JsonFeedbackResp?.rating,
          createdAt: moment().format("DD-MM-yyyy"),
          bleuScore: nlpData.metrics.bleuScore,
          fillerWordsCount:nlpData.metrics.fillerWordsCount,
          userEmail: user?.primaryEmailAddress.emailAddress,
          voiceMetrics: voiceMetrics || null,
        })
        .where(
          and(
            eq(UserAnswer.mockIdRef, interviewData?.mockId)
          )
        );

      toast("User Answer Recorded successfully.");

      setTimeout(() => {
        setResults([]);
        setUserAnswer("");
        finalAnswerRef.current = "";
        setIsSubmitted(false);
      }, 500);

    } catch (error) {
      setError(`Failed to update answer: ${error.message}`);
      console.error("Error updating answer:", error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex items-center justify-center flex-col">
      <div className="flex flex-col justify-center items-center rounded-lg p-5">
        <img src="/webcam3.png" alt="WebCAM" width={140} height={140} />
      </div>
      <div className="flex flex-col justify-center items-center rounded-lg p-5 mt-5 bg-black">
        <Webcam 
          ref={webcamRef}
          mirrored={true} 
          style={{ height: 300, width: "100%", zIndex: 100 }} 
          audio={false}
          videoConstraints={{
            width: 640,
            height: 480,
            facingMode: "user"
          }}
        />
      </div>

      {recording && (
        <div className="mt-4 p-4 bg-white rounded shadow w-full max-w-2xl">
          <h3 className="font-bold mb-2">Current Answer:</h3>
          <p>{userAnswer || "Recording..."}</p>
          {videoRecording && (
            <div className="mt-2">
              <span className="text-sm text-red-500 animate-pulse flex items-center gap-1">
                <Video size={16} /> Video recording active
              </span>
            </div>
          )}
        </div>
      )}

      <Button 
        disabled={loading} 
        variant="outline" 
        className="my-10" 
        onClick={recording ? stopRecording : startRecording}
      >
        {loading ? (
          <h2 className="text-gray-500">Processing...</h2>
        ) : recording ? (
          <h2 className="text-red-500 flex animate-pulse items-center gap-2">
            <StopCircle /> Stop Recording...
          </h2>
        ) : (
          <h2 className="flex gap-2 items-center">
            <Mic /> Record Answer
          </h2>
        )}
      </Button>

      {error && (
        <div className="mt-4 p-4 bg-red-100 text-red-700 rounded">
          {error}
        </div>
      )}
    </div>
  );
}

export default RecordQuestionSection;