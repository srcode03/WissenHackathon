"use client";
import React, { useEffect, useState } from "react";
import { db } from "../../../../../utils/db";
import { UserAnswer } from "../../../../../utils/schema";
import { eq } from "drizzle-orm";
import { Bar } from 'react-chartjs-2';
import { Chart as ChartJS, BarElement, CategoryScale, LinearScale } from 'chart.js';
ChartJS.register(BarElement, CategoryScale, LinearScale);
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "../../../../../components/ui/collapsible";
import ReactMarkdown from "react-markdown";
import { Prism as SyntaxHighlighter } from "react-syntax-highlighter";
import { dracula } from "react-syntax-highlighter/dist/cjs/styles/prism";
import { 
  ChevronDown, 
  Award, 
  AlertCircle, 
  CheckCircle, 
  Volume2, 
  Video, 
  Smile, 
  MoveHorizontal,
  Download,
  BarChart,
  PieChart,
  LineChart,
  TrendingUp
} from "lucide-react";
import { Button } from "../../../../../components/ui/button";
import { useRouter } from "next/navigation";
import { MdOutlineDashboardCustomize } from "react-icons/md";
import { useParams } from "next/navigation";

// Create a fallback Progress component in case the import fails
const FallbackProgress = ({ value, className, indicatorClassName }) => (
  <div className={`relative h-2 w-full overflow-hidden rounded-full bg-gray-200 ${className || ""}`}>
    <div 
      className={`h-full bg-blue-600 transition-all ${indicatorClassName || ""}`} 
      style={{ width: `${value || 0}%` }}
    />
  </div>
);

// Attempt to import Progress, fall back to our custom component if needed
let Progress;
try {
  // Dynamic import to avoid build errors
  Progress = FallbackProgress;
} catch (error) {
  Progress = FallbackProgress;
}

// Emotion color mapping for better visualization
const EMOTION_COLORS = {
  Happy: "bg-green-100 text-green-800 border-green-200",
  Neutral: "bg-slate-100 text-slate-800 border-slate-200",
  Sad: "bg-blue-100 text-blue-800 border-blue-200",
  Angry: "bg-red-100 text-red-800 border-red-200",
  Disgust: "bg-purple-100 text-purple-800 border-purple-200",
  Fear: "bg-yellow-100 text-yellow-800 border-yellow-200",
  Surprise: "bg-orange-100 text-orange-800 border-orange-200"
};

// Movement direction color mapping
const MOVEMENT_COLORS = {
  left: "bg-blue-100 text-blue-800 border-blue-200",
  right: "bg-green-100 text-green-800 border-green-200", 
  up: "bg-yellow-100 text-yellow-800 border-yellow-200",
  down: "bg-red-100 text-red-800 border-red-200"
};

function Feedback() {
  const [feedbackList, setFeedbackList] = useState([]);
  const [expandedQuestion, setExpandedQuestion] = useState(null);
  const [isGeneratingPDF, setIsGeneratingPDF] = useState(false);
  const router = useRouter();
  const params = useParams();
  const interviewId = params.interviewId;

  useEffect(() => {
    getFeedback();
  }, []);

  const getFeedback = async () => {
    try {
      const result = await db
        .select()
        .from(UserAnswer)
        .where(eq(UserAnswer.mockIdRef, interviewId))
        .orderBy(UserAnswer.id);

      setFeedbackList(result);
    } catch (error) {
      console.error("Error fetching feedback:", error);
    }
  };

  const handleQuestionToggle = (index) => {
    setExpandedQuestion(expandedQuestion === index ? null : index);
  };

  const downloadPDF = () => {
    try {
      setIsGeneratingPDF(true);
      
      // Create a hidden iframe to hold the printable content
      const printFrame = document.createElement('iframe');
      printFrame.style.position = 'fixed';
      printFrame.style.right = '0';
      printFrame.style.bottom = '0';
      printFrame.style.width = '0';
      printFrame.style.height = '0';
      printFrame.style.border = '0';
      
      document.body.appendChild(printFrame);
      
      // Generate the PDF content as HTML
      const content = `
        <!DOCTYPE html>
        <html>
        <head>
          <title>Interview Feedback Report</title>
          <style>
            body {
              font-family: Arial, sans-serif;
              color: #333;
              line-height: 1.5;
              padding: 20px;
            }
            .header {
              background: linear-gradient(to right, #22c55e, #059669);
              color: white;
              padding: 20px;
              border-radius: 10px;
              margin-bottom: 20px;
            }
            .question {
              background: white;
              border: 1px solid #ddd;
              border-radius: 10px;
              padding: 15px;
              margin-bottom: 15px;
            }
            .rating-high {
              color: #16a34a;
              font-weight: bold;
            }
            .rating-medium {
              color: #ca8a04;
              font-weight: bold;
            }
            .rating-low {
              color: #dc2626;
              font-weight: bold;
            }
            .section {
              padding: 15px;
              border-radius: 8px;
              margin-bottom: 15px;
            }
            .user-answer {
              background-color: #fee2e2;
              border: 1px solid #fecaca;
            }
            .ideal-answer {
              background-color: #dcfce7;
              border: 1px solid #bbf7d0;
            }
            .feedback {
              background-color: #fef3c7;
              border: 1px solid #fde68a;
            }
            .metrics {
              display: flex;
              flex-wrap: wrap;
              gap: 10px;
              margin-bottom: 15px;
            }
            .metric-card {
              background-color: #f0f9ff;
              border: 1px solid #bae6fd;
              padding: 10px;
              border-radius: 8px;
              width: calc(25% - 10px);
            }
            h1, h2, h3, h4 {
              margin-top: 0;
            }
            .progress-bar {
              height: 10px;
              background-color: #e5e7eb;
              border-radius: 5px;
              margin-top: 5px;
              overflow: hidden;
            }
            .progress-fill {
              height: 100%;
              border-radius: 5px;
            }
            .analysis-section {
              background-color: #f8f9fa;
              border: 1px solid #e9ecef;
              border-radius: 8px;
              padding: 15px;
              margin-bottom: 15px;
            }
            .chart-container {
              margin: 15px 0;
              text-align: center;
            }
            .chart-section {
              background-color: #ffffff;
              border: 1px solid #e0e0e0;
              border-radius: 8px;
              padding: 10px;
              margin-bottom: 10px;
            }
            .emotion-grid {
              display: grid;
              grid-template-columns: repeat(3, 1fr);
              gap: 10px;
              margin-top: 10px;
            }
            .emotion-item {
              background-color: #f0f4f8;
              border: 1px solid #d0d7de;
              padding: 8px;
              border-radius: 6px;
            }
          </style>
        </head>
        <body>
          <div class="header">
            <h1>Interview Feedback Report</h1>
            <p>Interview ID: ${interviewId}</p>
            <p>Date: ${new Date().toLocaleDateString()}</p>
          </div>
          
          <div>
            <h2>Performance Summary</h2>
            <p>Overall Score: <span class="${
              overallScore >= 80 ? 'rating-high' : 
              overallScore >= 60 ? 'rating-medium' : 
              'rating-low'
            }">${overallScore}/100</span></p>
            <div class="progress-bar">
              <div class="progress-fill" style="width: ${overallScore}%; background-color: ${
                overallScore >= 80 ? '#16a34a' : 
                overallScore >= 65 ? '#ca8a04' : 
                '#dc2626'
              };"></div>
            </div>
            
            <div class="metrics">
              <div class="metric-card">
                <h4>Total Questions</h4>
                <p>${feedbackList.length}</p>
              </div>
              <div class="metric-card">
                <h4>Strong Answers</h4>
                <p>${feedbackList.filter(item => parseInt(item.rating) >= 75).length}</p>
              </div>
              <div class="metric-card">
                <h4>Need Improvement</h4>
                <p>${feedbackList.filter(item => parseInt(item.rating) < 75).length}</p>
              </div>
            </div>
          </div>
          
          <h2>Detailed Feedback</h2>
          ${feedbackList.map((item, index) => `
            <div class="question">
              <h3>Question ${index + 1}: ${item.question}</h3>
              <p>Score: <span class="${
                parseInt(item.rating) >= 80 ? 'rating-high' : 
                parseInt(item.rating) >= 60 ? 'rating-medium' : 
                'rating-low'
              }">${item.rating}/100</span></p>
              
              ${item.bleuScore !== undefined ? `<p>BLEU Score: ${item.bleuScore}</p>` : ''}
              ${item.fillerWordsCount !== undefined ? `<p>Filler Words: ${item.fillerWordsCount}</p>` : ''}
              
              <div class="section user-answer">
                <h4>Your Answer</h4>
                <p>${item.userAns}</p>
              </div>
              
              <div class="section ideal-answer">
                <h4>Ideal Answer</h4>
                <p>${item.correctAns}</p>
              </div>
              
              <div class="section feedback">
                <h4>Feedback for Improvement</h4>
                <p>${item.feedback}</p>
              </div>
              
              ${item.voiceMetrics ? `
                <div class="analysis-section">
                  <h4>Voice Analysis</h4>
                  <div class="metrics">
                    <div class="metric-card">
                      <h5>Overall</h5>
                      <p>${item.voiceMetrics.confidence_score}</p>
                    </div>
                    <div class="metric-card">
                      <h5>Clarity</h5>
                      <p>${item.voiceMetrics.clarity_score}</p>
                    </div>
                    <div class="metric-card">
                      <h5>Rhythm</h5>
                      <p>${item.voiceMetrics.rhythm_score}</p>
                    </div>
                    <div class="metric-card">
                      <h5>Tone</h5>
                      <p>${item.voiceMetrics.tone_score}</p>
                    </div>
                  </div>
                  <p><strong>Voice Feedback:</strong> ${item.voiceMetrics.feedback}</p>
                </div>
              ` : ''}
              
              ${item.videoMetrics ? `
                <div class="analysis-section">
                  <h4>Video Analysis</h4>
                  
                  <div class="metrics">
                    <div class="metric-card">
                      <h5>Interview Score</h5>
                      <p>${item.videoMetrics.metrics.interview_score}/10</p>
                    </div>
                    <div class="metric-card">
                      <h5>Positive Emotions</h5>
                      <p>${item.videoMetrics.metrics.positive_emotions_percentage}%</p>
                    </div>
                    <div class="metric-card">
                      <h5>Blink Rate</h5>
                      <p>${item.videoMetrics.metrics.blink_rate.toFixed(1)}/min</p>
                    </div>
                    <div class="metric-card">
                      <h5>Looking Away</h5>
                      <p>${item.videoMetrics.metrics.total_looking_away} times</p>
                    </div>
                  </div>

                  <h5>Emotion Distribution</h5>
                  <div class="emotion-grid">
                    ${Object.entries(item.videoMetrics.metrics.raw_metrics.emotion_count || {}).map(([emotion, count]) => `
                      <div class="emotion-item">
                        <strong>${emotion}:</strong> ${count}
                      </div>
                    `).join('')}
                  </div>
                  
                  <h5>Head Movement</h5>
                  <div class="emotion-grid">
                    ${Object.entries(item.videoMetrics.metrics.raw_metrics.head_movement || {}).map(([direction, count]) => `
                      <div class="emotion-item">
                        <strong>${direction.replace(/_/g, ' ')}:</strong> ${count}
                      </div>
                    `).join('')}
                  </div>
                  
                  <p><strong>Video Feedback:</strong></p>
                  <ul>
                    ${item.videoMetrics.feedback}
                  </ul>
                </div>
              ` : ''}
            </div>
          `).join('')}
        </body>
        </html>
      `;
      
      // Write the content to the iframe
      printFrame.contentDocument.open();
      printFrame.contentDocument.write(content);
      printFrame.contentDocument.close();
      
      // Wait for content to load
      printFrame.onload = () => {
        // Print the iframe content (which will open the print dialog)
        printFrame.contentWindow.print();
        
        // Remove the iframe after printing
        setTimeout(() => {
          document.body.removeChild(printFrame);
          setIsGeneratingPDF(false);
        }, 500);
      };
      
    } catch (error) {
      console.error("Error generating PDF:", error);
      alert("Failed to generate PDF. Please try again.");
      setIsGeneratingPDF(false);
    }
  };

  const renderScoreCard = (score, label, color = "bg-blue-500") => {
    return (
      <div className="flex flex-col items-center">
        <div className={`text-xl font-bold ${score >= 80 ? 'text-green-600' : score >= 60 ? 'text-yellow-600' : 'text-red-600'}`}>
          {score}
        </div>
        <div className="text-xs font-medium text-gray-500 mt-1">{label}</div>
        <div className="h-1.5 w-16 mt-1 bg-gray-200 rounded-full overflow-hidden">
          <div 
            className={score >= 80 ? 'bg-green-500' : score >= 60 ? 'bg-yellow-500' : 'bg-red-500'} 
            style={{ width: `${score}%`, height: '100%' }}
          ></div>
        </div>
      </div>
    );
  };

  const renderChartImage = (base64String, title, icon) => {
    if (!base64String) return null;
    
    const Icon = icon || BarChart;
    
    return (
      <div className="mb-4">
        <div className="flex items-center gap-2 mb-2">
          <Icon className="h-4 w-4 text-gray-600" />
          <h4 className="text-sm font-medium text-gray-700">{title}</h4>
        </div>
        <div className="bg-white border rounded-lg overflow-hidden shadow-sm">
          <img 
            src={`data:image/png;base64,${base64String}`} 
            alt={title} 
            className="w-full h-auto object-contain" 
          />
        </div>
      </div>
    );
  };

  const renderAnalysisMetrics = (metrics) => {
    if (!metrics) return null;
    
    return (
      <div className="space-y-6 mt-6">
        {/* Voice Analysis Metrics */}
        <div className="rounded-xl border border-blue-200 overflow-hidden">
          <div className="bg-blue-50 px-4 py-3 flex items-center gap-2 border-b border-blue-200">
            <Volume2 className="text-blue-600 h-5 w-5" />
            <h3 className="font-semibold text-blue-800">Voice Analysis</h3>
          </div>
          
          <div className="p-4 bg-white">
            <div className="grid grid-cols-4 gap-6 mb-6">
              {renderScoreCard(metrics.voiceMetrics.confidence_score, "Overall")}
              {renderScoreCard(metrics.voiceMetrics.clarity_score, "Clarity")}
              {renderScoreCard(metrics.voiceMetrics.rhythm_score, "Rhythm")}
              {renderScoreCard(metrics.voiceMetrics.tone_score, "Tone")}
            </div>
            
            <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
              <div className="p-3 bg-blue-50 rounded-lg">
                <div className="text-sm text-blue-700 font-medium">Volume</div>
                <div className="text-lg font-bold">{metrics.voiceMetrics.volume_score}</div>
              </div>
              <div className="p-3 bg-blue-50 rounded-lg">
                <div className="text-sm text-blue-700 font-medium">Pitch</div>
                <div className="text-lg font-bold">{metrics.voiceMetrics.pitch_score}</div>
              </div>
              <div className="p-3 bg-blue-50 rounded-lg">
                <div className="text-sm text-blue-700 font-medium">Pauses</div>
                <div className="text-lg font-bold">{metrics.voiceMetrics.pause_score}</div>
              </div>
              <div className="p-3 bg-blue-50 rounded-lg">
                <div className="text-sm text-blue-700 font-medium">Frequency Variation</div>
                <div className="text-lg font-bold">{metrics.voiceMetrics.frequency_variation}</div>
              </div>
            </div>
            
            <div className="bg-blue-50 p-4 rounded-lg">
              <h4 className="font-medium text-blue-800 mb-2">Voice Feedback:</h4>
              <ReactMarkdown
                components={{
                  code({ node, inline, className, children, ...props }) {
                    const match = /language-(\w+)/.exec(className || "");
                    return !inline && match ? (
                      <SyntaxHighlighter
                        style={dracula}
                        language={match[1]}
                        PreTag="div"
                        {...props}
                      >
                        {String(children).replace(/\n$/, "")}
                      </SyntaxHighlighter>
                    ) : (
                      <code className="bg-green-100 px-1 py-0.5 rounded" {...props}>
                        {children}
                      </code>
                    );
                  },
                }}
              >
                {metrics.voiceMetrics.feedback}
              </ReactMarkdown>
            </div>
          </div>
        </div>

        {/* Video Analysis Metrics */}
        {metrics.videoMetrics && (
          <div className="rounded-xl border border-emerald-200 overflow-hidden">
            <div className="bg-emerald-50 px-4 py-3 flex items-center gap-2 border-b border-emerald-200">
              <Video className="text-emerald-600 h-5 w-5" />
              <h3 className="font-semibold text-emerald-800">Video Analysis</h3>
            </div>
            
            <div className="p-4 bg-white">
              <div className="flex flex-col md:flex-row gap-6 mb-6">
                {/* Left column - metrics */}
                <div className="w-full md:w-1/3 space-y-4">
                  <div className="bg-white border border-emerald-200 rounded-lg p-4 shadow-sm">
                    <div className="flex justify-between items-center mb-2">
                      <h4 className="font-medium text-emerald-800 flex items-center gap-2">
                        <Award className="h-4 w-4" />
                        Interview Score
                      </h4>
                      <span className="text-lg font-bold text-emerald-700">
                        {metrics.videoMetrics.metrics.interview_score}/10
                      </span>
                    </div>
                    <div className="h-2 bg-emerald-100 rounded-full overflow-hidden">
                      <div 
                        className="h-2 bg-emerald-500 rounded-full" 
                        style={{width: `${metrics.videoMetrics.metrics.interview_score * 10}%`}}
                      ></div>
                    </div>
                  </div>
                
                  <div className="bg-white border border-emerald-200 rounded-lg p-4 shadow-sm">
                    <div className="flex justify-between items-center mb-2">
                      <h4 className="font-medium text-emerald-800 flex items-center gap-2">
                        <Smile className="h-4 w-4" />
                        Positive Emotions
                      </h4>
                      <span className="text-lg font-bold text-emerald-700">
                        {metrics.videoMetrics.metrics.positive_emotions_percentage}%
                      </span>
                    </div>
                    <div className="h-2 bg-emerald-100 rounded-full overflow-hidden">
                      <div 
                        className="h-2 bg-emerald-500 rounded-full" 
                        style={{width: `${metrics.videoMetrics.metrics.positive_emotions_percentage}%`}}
                      ></div>
                    </div>
                  </div>
                  
                  <div className="bg-white border border-emerald-200 rounded-lg p-4 shadow-sm">
                    <div className="flex justify-between items-center mb-2">
                      <h4 className="font-medium text-emerald-800">Blink Rate</h4>
                      <span className="text-lg font-bold text-emerald-700">
                        {metrics.videoMetrics.metrics.blink_rate.toFixed(1)}
                        <span className="text-xs font-normal ml-1">blinks/min</span>
                      </span>
                    </div>
                    <div className="h-2 bg-emerald-100 rounded-full overflow-hidden">
                      <div 
                        className="h-2 bg-emerald-500 rounded-full" 
                        style={{width: `${Math.min(100, metrics.videoMetrics.metrics.blink_rate * 5)}%`}}
                      ></div>
                    </div>
                  </div>
                  
                  <div className="bg-white border border-emerald-200 rounded-lg p-4 shadow-sm">
                    <div className="flex justify-between items-center mb-2">
                      <h4 className="font-medium text-emerald-800">Looking Away</h4>
                      <span className="text-lg font-bold text-emerald-700">
                        {metrics.videoMetrics.metrics.total_looking_away}
                        <span className="text-xs font-normal ml-1">times</span>
                      </span>
                    </div>
                    <div className="h-2 bg-emerald-100 rounded-full overflow-hidden">
                      <div 
                        className="h-2 bg-emerald-500 rounded-full" 
                        style={{width: `${Math.min(100, metrics.videoMetrics.metrics.total_looking_away * 10)}%`}}
                      ></div>
                    </div>
                  </div>
                </div>
                
                {/* Right column - charts */}
                <div className="w-full md:w-2/3">
                  <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
                    {metrics.videoMetrics.charts.score_chart && 
                      renderChartImage(metrics.videoMetrics.charts.score_chart, "Interview Performance Score", Award)}
                      
                    {metrics.videoMetrics.charts.emotion_chart && 
                      renderChartImage(metrics.videoMetrics.charts.emotion_chart, "Emotion Distribution", PieChart)}
                      
                    {metrics.videoMetrics.charts.ear_chart && 
                      renderChartImage(metrics.videoMetrics.charts.ear_chart, "Blink Detection", LineChart)}
                      
                    {metrics.videoMetrics.charts.head_movement_chart && 
                      renderChartImage(metrics.videoMetrics.charts.head_movement_chart, "Head Movement Count", BarChart)}
                      
                    {metrics.videoMetrics.charts.head_orientation_chart && 
                      renderChartImage(metrics.videoMetrics.charts.head_orientation_chart, "Head Orientation Over Time", TrendingUp)}
                  </div>
                </div>
              </div>
              
              {/* Emotion Analysis */}
              <div className="mb-6">
                <h4 className="font-medium text-emerald-800 mb-3 flex items-center gap-2">
                  <Smile className="h-4 w-4" />
                  Emotion Analysis
                </h4>
                <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-3">
                  {metrics.videoMetrics.metrics.raw_metrics.emotion_count && 
                   Object.entries(metrics.videoMetrics.metrics.raw_metrics.emotion_count).map(([emotion, count]) => (
                    <div 
                      key={emotion} 
                      className={`${EMOTION_COLORS[emotion] || 'bg-gray-100'} p-3 rounded-lg border`}
                    >
                      <div className="flex justify-between items-center mb-2">
                        <h5 className="font-medium">{emotion}</h5>
                        <span className="text-sm font-bold">
                          {count}
                        </span>
                      </div>
                      <div className="h-1.5 bg-white/50 rounded-full overflow-hidden">
                        <div 
                          className={
                            emotion === "Happy" ? "bg-green-500" :
                            emotion === "Neutral" ? "bg-gray-500" :
                            emotion === "Sad" ? "bg-blue-500" :
                            emotion === "Angry" ? "bg-red-500" :
                            emotion === "Disgust" ? "bg-purple-500" :
                            emotion === "Fear" ? "bg-yellow-500" :
                            "bg-orange-500"
                          }
                          style={{width: `${Math.min(100, count * 5)}%`, height: '100%'}}
                        ></div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              {/* Head Movement Analysis */}
              <div className="mb-6">
                <h4 className="font-medium text-emerald-800 mb-3 flex items-center gap-2">
                  <MoveHorizontal className="h-4 w-4" />
                  Head Movement Analysis
                </h4>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                  {metrics.videoMetrics.metrics.raw_metrics.head_movement && 
                   Object.entries(metrics.videoMetrics.metrics.raw_metrics.head_movement).map(([direction, count]) => (
                    <div 
                      key={direction} 
                      className={`bg-gray-100 p-3 rounded-lg border`}
                    >
                      <div className="flex justify-between items-center mb-1">
                        <h5 className="font-medium capitalize">{direction.replace(/_/g, ' ')}</h5>
                        <span className="text-sm font-bold">
                          {count}
                        </span>
                      </div>
                      <div className="h-1.5 bg-white/50 rounded-full overflow-hidden">
                        <div 
                          className="bg-blue-500"
                          style={{width: `${Math.min(100, count * 10)}%`, height: '100%'}}
                        ></div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
              
              <div className="bg-emerald-50 p-4 rounded-lg">
                <h4 className="font-medium text-emerald-800 mb-2">Video Feedback:</h4>
                <ReactMarkdown
                components={{
                  code({ node, inline, className, children, ...props }) {
                    const match = /language-(\w+)/.exec(className || "");
                    return !inline && match ? (
                      <SyntaxHighlighter
                        style={dracula}
                        language={match[1]}
                        PreTag="div"
                        {...props}
                      >
                        {String(children).replace(/\n$/, "")}
                      </SyntaxHighlighter>
                    ) : (
                      <code className="bg-green-100 px-1 py-0.5 rounded" {...props}>
                        {children}
                      </code>
                    );
                  },
                }}
              >
                {metrics.videoMetrics.feedback}
              </ReactMarkdown>
              </div>
            </div>
          </div>
        )}
      </div>
    );
  };

  const overallScore = feedbackList.length > 0 
    ? Math.round(feedbackList.reduce((sum, item) => sum + (parseInt(item.rating) || 0), 0) / feedbackList.length) 
    : 0;

  return (
    <div className="max-w-6xl mx-auto p-6">
      {feedbackList.length === 0 ? (
        <div className="bg-white rounded-xl shadow-md p-10 text-center">
          <AlertCircle className="h-16 w-16 text-gray-400 mx-auto mb-4" />
          <h2 className="font-bold text-xl text-gray-500 mb-2">
            No Interview Feedback Record Found
          </h2>
          <p className="text-gray-400 mb-6">
            It seems there's no feedback available for this interview yet.
          </p>
          <Button
            onClick={() => router.replace("/dashboard")}
            className="bg-blue-600 hover:bg-blue-700"
          >
            Return to Dashboard
          </Button>
        </div>
      ) : (
        <div className="space-y-6">
          {/* Header with Summary */}
          <div className="bg-white rounded-xl shadow-md overflow-hidden">
            <div className="bg-gradient-to-r from-green-500 to-emerald-600 px-6 py-8 text-white">
              <div className="flex items-center gap-3 mb-2">
                <Award className="h-8 w-8" />
                <h2 className="text-3xl font-bold">Interview Feedback</h2>
              </div>
              <p className="opacity-90">
                Review your performance and get insights to improve your next interview.
              </p>
            </div>
            
            <div className="p-6">
              <div className="flex flex-col md:flex-row items-center justify-between gap-6 mb-6">
                <div className="text-center md:text-left">
                  <h3 className="text-lg font-medium text-gray-500 mb-1">Overall Performance</h3>
                  <div className="flex items-center gap-2">
                    <span className="text-4xl font-bold text-gray-800">{overallScore}</span>
                    <span className="text-sm text-gray-500">out of 100</span>
                  </div>
                </div>
                
                <div className="flex items-center gap-4">
                  <div className="text-center px-4 py-2 bg-blue-50 rounded-lg border border-blue-100">
                    <div className="text-sm font-medium text-blue-700 mb-1">Questions</div>
                    <div className="text-2xl font-bold text-blue-900">{feedbackList.length}</div>
                  </div>
                  
                  <div className="text-center px-4 py-2 bg-green-50 rounded-lg border border-green-100">
                    <div className="text-sm font-medium text-green-700 mb-1">Strong Answers</div>
                    <div className="text-2xl font-bold text-green-900">
                      {feedbackList.filter(item => parseInt(item.rating) >= 75).length}
                    </div>
                  </div>
                  
                  <div className="text-center px-4 py-2 bg-amber-50 rounded-lg border border-amber-100">
                    <div className="text-sm font-medium text-amber-700 mb-1">Need Improvement</div>
                    <div className="text-2xl font-bold text-amber-900">
                      {feedbackList.filter(item => parseInt(item.rating) < 75).length}
                    </div>
                  </div>
                </div>
              </div>
              
              <div className="h-2.5 bg-gray-200 rounded-full overflow-hidden mb-2">
                <div 
                  className={
                    overallScore >= 80 ? "bg-green-500" :
                    overallScore >= 65 ? "bg-yellow-500" :
                    "bg-red-500"
                  } 
                  style={{width: `${overallScore}%`, height: '100%'}}
                ></div>
              </div>
              
              <p className="text-sm text-gray-500 italic">
                Click on each question below to view detailed feedback and analysis.
              </p>
            </div>
          </div>
          
          {/* Questions */}
          <div className="space-y-4">
            {feedbackList.map((item, index) => (
              <div key={index} className="bg-white rounded-xl shadow-sm border overflow-hidden">
                <div 
                  onClick={() => handleQuestionToggle(index)}
                  className={`p-4 flex items-center justify-between cursor-pointer transition-colors
                    ${expandedQuestion === index ? 'bg-gray-50 border-b' : ''}
                    ${parseInt(item.rating) >= 80 ? 'hover:bg-green-50' : 
                      parseInt(item.rating) >= 60 ? 'hover:bg-yellow-50' : 'hover:bg-red-50'}`}
                >
                  <div className="flex items-center gap-4 flex-1">
                    <div className={`flex-shrink-0 h-10 w-10 rounded-full flex items-center justify-center
                      ${parseInt(item.rating) >= 80 ? 'bg-green-100 text-green-600' : 
                        parseInt(item.rating) >= 60 ? 'bg-yellow-100 text-yellow-600' : 'bg-red-100 text-red-600'}`}>
                      {index + 1}
                    </div>
                    <div className="flex-1">
                      <h3 className="font-medium text-gray-900">{item.question}</h3>
                      <div className="flex items-center gap-3 mt-1">
                        <div className="text-sm font-medium flex items-center gap-1">
                          <span className="text-gray-500">Score:</span>
                          <span className={`
                            ${parseInt(item.rating) >= 80 ? 'text-green-600' : 
                              parseInt(item.rating) >= 60 ? 'text-yellow-600' : 'text-red-600'}`}>
                            {item.rating}
                          </span>
                        </div>
                        
                        {item.bleuScore !== undefined && (
                          <div className="text-sm font-medium flex items-center gap-1">
                            <span className="text-gray-500">Text Matching Score:</span>
                            <span className="text-blue-600">{item.voiceMetrics.semantic_similarity}</span>
                          </div>
                        )}
                        
                        {item.fillerWordsCount !== undefined && (
                          <div className="text-sm font-medium flex items-center gap-1">
                            <span className="text-gray-500">Filler Words:</span>
                            <span className="text-purple-600">{item.fillerWordsCount}%</span>
                          </div>
                        )}
                      </div>
                    </div>
                  </div>
                  <ChevronDown className={`h-5 w-5 text-gray-400 transition-transform
                    ${expandedQuestion === index ? 'rotate-180' : ''}`} />
                </div>
                
                {expandedQuestion === index && (
                  <div className="p-4 bg-white">
                    <div className="space-y-4">
                      <div className="rounded-lg border border-red-100 bg-red-50 p-4">
                        <h4 className="font-medium text-red-800 mb-2 flex items-center gap-2">
                          <AlertCircle className="h-4 w-4" />
                          Your Answer
                        </h4>
                        <p className="text-sm text-red-900">{item.userAns}</p>
                      </div>
                      
                      <div className="rounded-lg border border-green-100 bg-green-50 p-4">
                        <h4 className="font-medium text-green-800 mb-2 flex items-center gap-2">
                          <CheckCircle className="h-4 w-4" />
                          Ideal Answer
                        </h4>
                        <ReactMarkdown
      components={{
        code({ node, inline, className, children, ...props }) {
          const match = /language-(\w+)/.exec(className || "");
          return !inline && match ? (
            <SyntaxHighlighter
              style={dracula}
              language={match[1]}
              PreTag="div"
              {...props}
            >
              {String(children).replace(/\n$/, "")}
            </SyntaxHighlighter>
          ) : (
            <code className="bg-green-100 px-1 py-0.5 rounded" {...props}>
              {children}
            </code>
          );
        },
      }}
    >
                        {item.correctAns}
                          </ReactMarkdown>
                        {/* <p className="text-sm text-green-900">{item.correctAns}</p> */}
                      </div>
                      
                      {renderAnalysisMetrics(item)}
                      
                      <div className="rounded-lg border border-amber-100 bg-amber-50 p-4">
                        <h4 className="font-medium text-amber-800 mb-2">Feedback for Improvement</h4>
                        {/* <p className="text-sm text-amber-900">{item.feedback}</p> */}
                        <ReactMarkdown
      components={{
        code({ node, inline, className, children, ...props }) {
          const match = /language-(\w+)/.exec(className || "");
          return !inline && match ? (
            <SyntaxHighlighter
              style={dracula}
              language={match[1]}
              PreTag="div"
              {...props}
            >
              {String(children).replace(/\n$/, "")}
            </SyntaxHighlighter>
          ) : (
            <code className="bg-green-100 px-1 py-0.5 rounded" {...props}>
              {children}
            </code>
          );
        },
      }}
    >
                        {item.feedback}
                          </ReactMarkdown>
                      </div>
                    </div>
                  </div>
                )}
              </div>
            ))}
          </div>
          
          {/* Action Buttons */}
          <div className="flex flex-wrap gap-4">
            <Button
              onClick={() => router.replace("/dashboard")}
              className="flex items-center gap-2 bg-blue-600 hover:bg-blue-700"
            >
              <MdOutlineDashboardCustomize className="h-4 w-4" />
              Return to Dashboard
            </Button>
            
            <Button
              onClick={downloadPDF}
              disabled={isGeneratingPDF || feedbackList.length === 0}
              className="flex items-center gap-2 bg-purple-600 hover:bg-purple-700"
            >
              <Download className="h-4 w-4" />
              {isGeneratingPDF ? "Generating PDF..." : "Download Feedback as PDF"}
            </Button>
          </div>
        </div>
      )}
    </div>
  );
}

export default Feedback;