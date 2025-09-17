"""
Voice-Enabled CSV Assistant Web App
Flask web application with voice interface for CSV analysis.

Dependencies:
- flask
- pandas
- openai
- python-dotenv
- langchain_experimental
- langchain_openai
- werkzeug
"""

import os
import io
import base64
import pandas as pd
from flask import Flask, render_template_string, request, jsonify, send_file
from dotenv import load_dotenv
from typing import Dict, Any
from openai import OpenAI, OpenAIError
import tempfile
import logging

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CSVAnalyzer:
    def __init__(self, openai_api_key: str):
        self.api_key = openai_api_key
        self.df = None
        self.agent = None

        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not provided")

    def load_data_from_file(self, file_content: bytes, filename: str) -> bool:
        try:
            # Create a temporary file to store the uploaded content
            with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.csv') as temp_file:
                temp_file.write(file_content)
                temp_file_path = temp_file.name
            
            self.df = pd.read_csv(temp_file_path)
            os.unlink(temp_file_path)  # Clean up temp file
            
            logger.info(f"✅ Loaded CSV: {len(self.df)} rows, {len(self.df.columns)} columns")
            return True
        except pd.errors.ParserError as e:
            logger.error(f"❌ Error parsing CSV: {e}")
            return False
        except Exception as e:
            logger.error(f"❌ Unexpected error loading CSV: {e}")
            return False

    def create_agent(self) -> bool:
        try:
            from langchain_experimental.agents import create_pandas_dataframe_agent
            from langchain_openai import ChatOpenAI

            if self.df is None:
                logger.error("❌ No data loaded.")
                return False

            llm = ChatOpenAI(
                openai_api_key=self.api_key,
                model="gpt-3.5-turbo",
                temperature=0
            )
            self.agent = create_pandas_dataframe_agent(
                llm=llm,
                df=self.df,
                verbose=False,
                allow_dangerous_code=True,
                agent_type="tool-calling"
            )
            logger.info("✅ Agent created successfully")
            return True
        except ImportError as e:
            logger.error(f"❌ Error importing dependencies: {e}")
            return False
        except Exception as e:
            logger.error(f"❌ Error creating agent: {e}")
            return False

    def query(self, question: str) -> Dict[str, Any]:
        if not self.agent:
            return {"success": False, "error": "Agent not initialized", "answer": None, "question": question}
        try:
            response = self.agent.invoke({"input": question})
            answer = response.get("output", str(response)) if isinstance(response, dict) else str(response)
            return {"success": True, "error": None, "answer": answer, "question": question}
        except Exception as e:
            return {"success": False, "error": str(e), "answer": None, "question": question}

class VoiceCSVWebApp:
    def __init__(self):
        self.app = Flask(__name__)
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.client = OpenAI(api_key=self.api_key) if self.api_key else None
        self.analyzer = None
        
        if not self.api_key:
            logger.warning("⚠️  OPENAI_API_KEY not found. Some features will be disabled.")
        
        self.setup_routes()

    def speech_to_text(self, audio_data: bytes) -> str:
        if not self.client:
            return "OpenAI API key not configured"
        
        try:
            # Save audio data to temporary file
            with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.wav') as temp_file:
                temp_file.write(audio_data)
                temp_file_path = temp_file.name
            
            with open(temp_file_path, "rb") as f:
                transcript = self.client.audio.transcriptions.create(
                    model="whisper-1",
                    file=f
                )
            
            os.unlink(temp_file_path)  # Clean up temp file
            return transcript.text
        except OpenAIError as e:
            logger.error(f"❌ OpenAI API error: {e}")
            return f"Speech-to-text error: {str(e)}"
        except Exception as e:
            logger.error(f"❌ Error in speech-to-text: {e}")
            return f"Error: {str(e)}"

    def text_to_speech(self, text: str) -> bytes:
        if not self.client:
            return b""
        
        try:
            response = self.client.audio.speech.create(
                model="tts-1",
                voice="alloy",
                input=text,
                response_format="mp3"
            )
            return response.content
        except OpenAIError as e:
            logger.error(f"❌ OpenAI API error: {e}")
            return b""
        except Exception as e:
            logger.error(f"❌ Error in text-to-speech: {e}")
            return b""

    def setup_routes(self):
        @self.app.route('/')
        def index():
            return render_template_string(HTML_TEMPLATE)

        @self.app.route('/upload_csv', methods=['POST'])
        def upload_csv():
            try:
                if 'file' not in request.files:
                    return jsonify({'success': False, 'error': 'No file uploaded'})
                
                file = request.files['file']
                if file.filename == '':
                    return jsonify({'success': False, 'error': 'No file selected'})
                
                if not file.filename.lower().endswith('.csv'):
                    return jsonify({'success': False, 'error': 'Only CSV files are allowed'})
                
                # Read file content
                file_content = file.read()
                
                # Create analyzer and load data
                self.analyzer = CSVAnalyzer(self.api_key)
                if not self.analyzer.load_data_from_file(file_content, file.filename):
                    return jsonify({'success': False, 'error': 'Failed to load CSV file'})
                
                if not self.analyzer.create_agent():
                    return jsonify({'success': False, 'error': 'Failed to create analysis agent'})
                
                # Get basic info about the dataset
                info = {
                    'rows': len(self.analyzer.df),
                    'columns': len(self.analyzer.df.columns),
                    'column_names': list(self.analyzer.df.columns)
                }
                
                return jsonify({
                    'success': True, 
                    'message': f'CSV loaded successfully! {info["rows"]} rows, {info["columns"]} columns',
                    'info': info
                })
                
            except Exception as e:
                logger.error(f"Error uploading CSV: {e}")
                return jsonify({'success': False, 'error': str(e)})

        @self.app.route('/voice_query', methods=['POST'])
        def voice_query():
            try:
                if not self.analyzer:
                    return jsonify({'success': False, 'error': 'No CSV file loaded. Please upload a CSV first.'})
                
                # Get audio data from request
                audio_data = request.files['audio'].read()
                
                # Convert speech to text
                question = self.speech_to_text(audio_data)
                if not question or question.startswith('Error'):
                    return jsonify({'success': False, 'error': f'Speech recognition failed: {question}'})
                
                # Query the CSV
                result = self.analyzer.query(question)
                
                if result['success']:
                    # Convert answer to speech
                    audio_response = self.text_to_speech(result['answer'])
                    audio_b64 = base64.b64encode(audio_response).decode() if audio_response else ""
                    
                    return jsonify({
                        'success': True,
                        'question': question,
                        'answer': result['answer'],
                        'audio': audio_b64
                    })
                else:
                    return jsonify({
                        'success': False,
                        'error': result['error'],
                        'question': question
                    })
                
            except Exception as e:
                logger.error(f"Error in voice query: {e}")
                return jsonify({'success': False, 'error': str(e)})

        @self.app.route('/text_query', methods=['POST'])
        def text_query():
            try:
                if not self.analyzer:
                    return jsonify({'success': False, 'error': 'No CSV file loaded. Please upload a CSV first.'})
                
                data = request.get_json()
                question = data.get('question', '').strip()
                
                if not question:
                    return jsonify({'success': False, 'error': 'No question provided'})
                
                # Query the CSV
                result = self.analyzer.query(question)
                
                if result['success']:
                    # Convert answer to speech
                    audio_response = self.text_to_speech(result['answer'])
                    audio_b64 = base64.b64encode(audio_response).decode() if audio_response else ""
                    
                    return jsonify({
                        'success': True,
                        'question': question,
                        'answer': result['answer'],
                        'audio': audio_b64
                    })
                else:
                    return jsonify({
                        'success': False,
                        'error': result['error'],
                        'question': question
                    })
                
            except Exception as e:
                logger.error(f"Error in text query: {e}")
                return jsonify({'success': False, 'error': str(e)})

    def run(self, host='0.0.0.0', port=5000, debug=False):
        self.app.run(host=host, port=port, debug=debug)

# HTML Template with embedded CSS and JavaScript
HTML_TEMPLATE = '''

<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Voice CSV Assistant</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            background: #1a1a1a;
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            color: #ffffff;
        }
        .container {
            background: #2a2a2a;
            border-radius: 16px;
            padding: 40px;
            max-width: 500px;
            width: 90%;
            text-align: center;
            box-shadow: 0 20px 40px rgba(0, 0, 0, 0.3);
            border: 1px solid #3a3a3a;
        }
        .logo {
            width: 60px;
            height: 60px;
            margin: 0 auto 30px;
            background: linear-gradient(135deg, #00ff88, #00cc6a);
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            animation: spin 8s linear infinite;
        }
        @keyframes spin {
            from { transform: rotate(0deg); }
            to { transform: rotate(360deg); }
        }
        .logo::before {
            content: '';
            width: 30px;
            height: 30px;
            background: conic-gradient(from 0deg, transparent, #ffffff, transparent, #ffffff, transparent);
            border-radius: 50%;
        }
        h1 {
            margin-bottom: 15px;
            font-size: 2.2em;
            font-weight: 300;
            color: #ffffff;
        }
        .subtitle {
            color: #888888;
            margin-bottom: 40px;
            font-size: 0.9em;
            line-height: 1.4;
        }
        .features {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 20px;
            margin-bottom: 40px;
        }
        .feature {
            background: #3a3a3a;
            padding: 20px 15px;
            border-radius: 12px;
            text-align: center;
            border: 1px solid #4a4a4a;
            transition: all 0.3s ease;
        }
        .feature:hover {
            background: #444444;
            border-color: #00ff88;
        }
        .feature-icon {
            width: 32px;
            height: 32px;
            margin: 0 auto 10px;
            background: #00ff88;
            border-radius: 8px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 16px;
        }
        .feature-title {
            font-size: 0.9em;
            font-weight: 600;
            margin-bottom: 5px;
            color: #ffffff;
        }
        .feature-desc {
            font-size: 0.7em;
            color: #888888;
            line-height: 1.3;
        }
        .upload-section {
            margin-bottom: 30px;
        }
        .file-input-wrapper {
            position: relative;
            overflow: hidden;
            display: inline-block;
            width: 100%;
        }
        .file-input {
            position: absolute;
            left: -9999px;
        }
        .file-input-button {
            background: #3a3a3a;
            border: 2px dashed #666666;
            border-radius: 12px;
            padding: 25px;
            cursor: pointer;
            transition: all 0.3s ease;
            display: block;
            width: 100%;
            color: #ffffff;
            font-size: 14px;
        }
        .file-input-button:hover {
            background: #444444;
            border-color: #00ff88;
        }
        .input-section {
            position: relative;
            margin-bottom: 20px;
        }
        .input-container {
            background: #3a3a3a;
            border: 1px solid #4a4a4a;
            border-radius: 25px;
            padding: 4px;
            display: flex;
            align-items: center;
            transition: all 0.3s ease;
        }
        .input-container:focus-within {
            border-color: #00ff88;
            box-shadow: 0 0 0 2px rgba(0, 255, 136, 0.1);
        }
        .text-input {
            flex: 1;
            padding: 12px 20px;
            border: none;
            background: transparent;
            color: #ffffff;
            font-size: 14px;
            outline: none;
        }
        .text-input::placeholder {
            color: #888888;
        }
        .mic-button {
            width: 40px;
            height: 40px;
            border-radius: 50%;
            background: #00ff88;
            border: none;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            margin-right: 4px;
            transition: all 0.3s ease;
            color: #000000;
            font-weight: bold;
        }
        .mic-button:hover {
            background: #00cc6a;
            transform: scale(1.05);
        }
        .mic-button.recording {
            background: #ff4757;
            animation: pulse 1s infinite;
        }
        @keyframes pulse {
            0% { transform: scale(1); box-shadow: 0 0 0 0 rgba(255, 71, 87, 0.7); }
            70% { transform: scale(1.05); box-shadow: 0 0 0 10px rgba(255, 71, 87, 0); }
            100% { transform: scale(1); box-shadow: 0 0 0 0 rgba(255, 71, 87, 0); }
        }
        .send-button {
            background: #00ff88;
            border: none;
            padding: 10px 20px;
            border-radius: 20px;
            color: #000000;
            cursor: pointer;
            font-size: 14px;
            font-weight: 600;
            transition: all 0.3s ease;
            margin-left: 10px;
        }
        .send-button:hover:not(:disabled) {
            background: #00cc6a;
            transform: translateY(-1px);
        }
        .send-button:disabled {
            background: #666666;
            cursor: not-allowed;
        }
        .tabs {
            display: flex;
            justify-content: center;
            gap: 20px;
            margin-bottom: 30px;
        }
        .tab {
            padding: 8px 16px;
            background: transparent;
            border: 1px solid #4a4a4a;
            border-radius: 20px;
            color: #888888;
            cursor: pointer;
            font-size: 13px;
            transition: all 0.3s ease;
        }
        .tab.active {
            background: #00ff88;
            color: #000000;
            border-color: #00ff88;
        }
        .tab:hover:not(.active) {
            border-color: #666666;
            color: #ffffff;
        }
        .status {
            margin-top: 20px;
            padding: 12px 16px;
            border-radius: 8px;
            display: none;
            font-size: 14px;
        }
        .status.success {
            background: rgba(0, 255, 136, 0.1);
            border: 1px solid rgba(0, 255, 136, 0.3);
            color: #00ff88;
        }
        .status.error {
            background: rgba(255, 71, 87, 0.1);
            border: 1px solid rgba(255, 71, 87, 0.3);
            color: #ff4757;
        }
        .response {
            margin-top: 20px;
            padding: 20px;
            background: #3a3a3a;
            border-radius: 12px;
            text-align: left;
            display: none;
            border: 1px solid #4a4a4a;
        }
        .question {
            font-weight: 600;
            margin-bottom: 12px;
            color: #00ff88;
            font-size: 14px;
        }
        .answer {
            line-height: 1.6;
            color: #ffffff;
            font-size: 14px;
        }
        .wave-animation {
            display: none;
            margin: 15px 0;
            height: 40px;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 3px;
        }
        .wave-bar {
            width: 3px;
            height: 20px;
            background: #00ff88;
            border-radius: 2px;
            animation: wave 1.2s ease-in-out infinite;
        }
        .wave-bar:nth-child(2) { animation-delay: 0.1s; }
        .wave-bar:nth-child(3) { animation-delay: 0.2s; }
        .wave-bar:nth-child(4) { animation-delay: 0.3s; }
        .wave-bar:nth-child(5) { animation-delay: 0.4s; }
        @keyframes wave {
            0%, 100% { height: 20px; }
            50% { height: 40px; }
        }
        .voice-status {
            margin-top: 10px;
            font-size: 12px;
            color: #888888;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="logo"></div>
       
        <h1>How can I help you today?</h1>
        <div class="subtitle">Upload your CSV file and ask questions using voice or text. I'll analyze your data and provide insights.</div>
       
        <div class="features">
            <div class="feature">
                <div class="feature-icon">📊</div>
                <div class="feature-title">CSV Analysis</div>
                <div class="feature-desc">Upload and analyze CSV data with intelligent insights</div>
            </div>
            <div class="feature">
                <div class="feature-icon">🎤</div>
                <div class="feature-title">Voice Queries</div>
                <div class="feature-desc">Ask questions naturally using voice commands</div>
            </div>
            <div class="feature">
                <div class="feature-icon">💬</div>
                <div class="feature-title">Smart Responses</div>
                <div class="feature-desc">Get detailed answers with audio feedback</div>
            </div>
        </div>
        <div class="upload-section">
            <div class="file-input-wrapper">
                <input type="file" id="csvFile" accept=".csv" class="file-input">
                <label for="csvFile" class="file-input-button">
                    📊 Upload CSV File
                    <br><small style="color: #888888;">Click to select your CSV file</small>
                </label>
            </div>
        </div>
        <div class="tabs">
            <div class="tab active" data-tab="all">All</div>
            <div class="tab" data-tab="text">Text</div>
            <div class="tab" data-tab="voice">Voice</div>
            <div class="tab" data-tab="analysis">Analysis</div>
        </div>
        <div class="input-section">
            <div class="input-container">
                <button id="micButton" class="mic-button" title="Click to record voice question">🎤</button>
                <input type="text" id="textInput" class="text-input" placeholder="Type your prompt here..." disabled>
                <button id="sendButton" class="send-button" disabled>↗</button>
            </div>
            <div class="voice-status" id="voiceStatus">Upload a CSV file to get started</div>
        </div>
        <div class="wave-animation" id="waveAnimation">
            <div class="wave-bar"></div>
            <div class="wave-bar"></div>
            <div class="wave-bar"></div>
            <div class="wave-bar"></div>
            <div class="wave-bar"></div>
        </div>
        <div class="status" id="status"></div>
        <div class="response" id="response"></div>
    </div>
    <script>
        let isRecording = false;
        let mediaRecorder = null;
        let audioChunks = [];
        let csvUploaded = false;
        const micButton = document.getElementById('micButton');
        const voiceStatus = document.getElementById('voiceStatus');
        const status = document.getElementById('status');
        const response = document.getElementById('response');
        const csvFile = document.getElementById('csvFile');
        const textInput = document.getElementById('textInput');
        const sendButton = document.getElementById('sendButton');
        const waveAnimation = document.getElementById('waveAnimation');
        const tabs = document.querySelectorAll('.tab');
        // Tab functionality
        tabs.forEach(tab => {
            tab.addEventListener('click', function() {
                tabs.forEach(t => t.classList.remove('active'));
                this.classList.add('active');
            });
        });
        // File upload handler
        csvFile.addEventListener('change', async function(e) {
            const file = e.target.files[0];
            if (!file) return;
            const formData = new FormData();
            formData.append('file', file);
            showStatus('Uploading CSV...', 'success');
           
            try {
                const response = await fetch('/upload_csv', {
                    method: 'POST',
                    body: formData
                });
               
                const result = await response.json();
               
                if (result.success) {
                    showStatus(result.message, 'success');
                    csvUploaded = true;
                    textInput.disabled = false;
                    sendButton.disabled = false;
                    voiceStatus.textContent = 'CSV loaded! Ask me anything about your data';
                   
                    // Update upload button
                    document.querySelector('.file-input-button').innerHTML = `
                        ✓ ${file.name} loaded
                        <br><small style="color: #00ff88;">Ready for analysis</small>
                    `;
                } else {
                    showStatus(`Error: ${result.error}`, 'error');
                }
            } catch (error) {
                showStatus(`Upload failed: ${error.message}`, 'error');
            }
        });
        // Voice recording
        micButton.addEventListener('click', async function() {
            if (!csvUploaded) {
                showStatus('Please upload a CSV file first', 'error');
                return;
            }
            if (!isRecording) {
                try {
                    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                    mediaRecorder = new MediaRecorder(stream);
                    audioChunks = [];
                    mediaRecorder.ondataavailable = event => {
                        audioChunks.push(event.data);
                    };
                    mediaRecorder.onstop = async () => {
                        const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
                        await sendVoiceQuery(audioBlob);
                        stream.getTracks().forEach(track => track.stop());
                    };
                    mediaRecorder.start();
                    isRecording = true;
                    micButton.classList.add('recording');
                    micButton.textContent = '⏸';
                    voiceStatus.textContent = 'Recording... Click again to stop';
                    showWaveAnimation();
                } catch (error) {
                    showStatus('Microphone access denied or not available', 'error');
                }
            } else {
                mediaRecorder.stop();
                isRecording = false;
                micButton.classList.remove('recording');
                micButton.textContent = '🎤';
                voiceStatus.textContent = 'Processing your question...';
                hideWaveAnimation();
            }
        });
        // Text query
        sendButton.addEventListener('click', async function() {
            const question = textInput.value.trim();
            if (!question) {
                showStatus('Please enter a question', 'error');
                return;
            }
            await sendTextQuery(question);
        });
        textInput.addEventListener('keypress', function(e) {
            if (e.key === 'Enter') {
                sendButton.click();
            }
        });
        async function sendVoiceQuery(audioBlob) {
            const formData = new FormData();
            formData.append('audio', audioBlob, 'question.wav');
            try {
                const response = await fetch('/voice_query', {
                    method: 'POST',
                    body: formData
                });
                const result = await response.json();
                handleQueryResponse(result);
            } catch (error) {
                showStatus(`Voice query failed: ${error.message}`, 'error');
                voiceStatus.textContent = 'Click to record your question';
            }
        }
        async function sendTextQuery(question) {
            try {
                console.log('Sending text query:', question);
                showStatus('Processing your question...', 'success');
               
                const response = await fetch('/text_query', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({ question: question })
                });
                console.log('Response status:', response.status);
                const result = await response.json();
                console.log('Raw response from server:', result);
               
                handleQueryResponse(result);
            } catch (error) {
                console.error('Text query error:', error);
                showStatus(`Text query failed: ${error.message}`, 'error');
            }
        }
        function handleQueryResponse(result) {
            if (result.success) {
                showResponse(result.question, result.answer);
                if (result.audio) {
                    playAudio(result.audio);
                }
                hideStatus();
            } else {
                showStatus(`Error: ${result.error}`, 'error');
            }
            voiceStatus.textContent = 'Ask me another question about your data';
            textInput.value = '';
        }
        function showStatus(message, type) {
            status.textContent = message;
            status.className = `status ${type}`;
            status.style.display = 'block';
        }
        function hideStatus() {
            status.style.display = 'none';
        }
        function showResponse(question, answer) {
            const questionDiv = response.querySelector('.question') || document.createElement('div');
            const answerDiv = response.querySelector('.answer') || document.createElement('div');
           
            questionDiv.className = 'question';
            questionDiv.textContent = `Q: ${question}`;
           
            answerDiv.className = 'answer';
            answerDiv.textContent = `A: ${answer}`;
           
            response.innerHTML = '';
            response.appendChild(questionDiv);
            response.appendChild(answerDiv);
            response.style.display = 'block';
        }
        function playAudio(base64Audio) {
            if (!base64Audio) return;
            
            const audioBlob = new Blob([Uint8Array.from(atob(base64Audio), c => c.charCodeAt(0))], 
                                     { type: 'audio/mp3' });
            const audioUrl = URL.createObjectURL(audioBlob);
            const audio = new Audio(audioUrl);
            audio.play().catch(e => console.log('Could not play audio:', e));
        }
        function showWaveAnimation() {
            waveAnimation.style.display = 'flex';
        }
        function hideWaveAnimation() {
            waveAnimation.style.display = 'none';
        }
    </script>
</body>
</html>
'''
webapp = VoiceCSVWebApp()
app = webapp.app

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=False)


