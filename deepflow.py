import asyncio
import sys
from flask import Flask, render_template_string, request, jsonify
import sounddevice as sd
from scipy.io.wavfile import write
from browser_use import Agent
import numpy as np
from pathlib import Path
import base64
import threading
from openai import OpenAI
from typing import Optional
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os,time
from browser_use.browser.context import BrowserContextConfig, BrowserContext
from browser_use import Agent, Browser, BrowserConfig
from langchain_openai import ChatOpenAI
from elevenlabs.client import ElevenLabs
from elevenlabs import play, save, stream, Voice, VoiceSettings
import cv2

os.environ["OPENAI_API_KEY"] = ""


def analyze_image_simple(image_path, prompt=None):
    """Quick image analysis using GPT-4 Vision"""
    start_time = time.time()
    client = OpenAI(api_key="")

    with open(image_path, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode('utf-8')

    # Use custom prompt if provided, otherwise use default

    default_prompt = "Use one sentence to describe the thing shown in the image. Do not use any other words and do not include sentence like 'This is a picture of' or 'A person is holding balabala'"
    analysis_prompt = prompt if prompt is not None else default_prompt
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{

            "role": "user",

            "content": [

                {"type": "text", "text": analysis_prompt},

                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
            ]
        }],
        max_tokens=300
    )
    
    return response.choices[0].message.content, time.time() - start_time



def run_search_tasks(search_tasks, model="gpt-4o"):
    """
    Runs multiple web searches concurrently using the provided tasks, prints
    the first result for each, and provides optional voice output. It uses
    `ChatOpenAI` and `Agent` classes.

    Returns:
        bool: True if all tasks completed successfully, False if an exception occurred.
    """

    # ---------------------------
    # Original Web Search Functionality
    # ---------------------------
    # Fix for Windows asyncio subprocess issue
    # if sys.platform == "win32":
    #     asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

    if sys.platform == "win32":
        # Create a new event loop for each run
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    else:
        loop = asyncio.get_event_loop()

    async def run_search(task):
        try:
            config = BrowserContextConfig(
                # browser_window_size={'width': 500, 'height': 500},
                user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/85.0.4183.102 Safari/537.36',
                highlight_elements=False,
                viewport_expansion=500,
                locale='en-US',
                # device_scale_factor=2.0,
            )

            browser = Browser(
                config=BrowserConfig(
                    chrome_instance_path='C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe',
                    # headless=True,
                )
            )

            context = BrowserContext(browser=browser, config=config)
            agent = Agent(
                browser_context=context,
                task=f"{task}",
                llm=ChatOpenAI(model=model),
            )
            result = await agent.run()
            print(f"{task}': {result}")
            return (task, result)
            # Optional: provide voice output for each task result
        except Exception as e:
            # Print and optionally speak the error for this individual task
            print(f"Error in search task '{task}': {str(e)}")
            return (task, f"I apologize, but I encountered an error while searching. Error: {str(e)}")

    async def main():
        try:
            # Convert search_tasks to a string if it's a list
            search_tasks_str = ' '.join(search_tasks) if isinstance(search_tasks, list) else str(search_tasks)
            
            # Gather all searches concurrently with extended timeout using the parsed task list
            collected_results = await asyncio.wait_for(
                asyncio.gather(*(run_search(task) for task in search_tasks)),
                timeout=200  # Extended from 30 to 120 seconds
            )

            print("COLLECTED RESULTS", collected_results)
            return True, collected_results  # Return both success status and results

        except Exception as e:
            print(f"Error occurred while performing searches: {e}")
            return False, []  # Return failure status and empty results list


    # ---------------------------
    # Exception Handling & Voice Output
    # ---------------------------
     # Handle exceptions and return a boolean
    try:
        conversation(search_tasks,
                     "You are a helpful AI assistant called DeepFlow. You are going to be doing these tasks. "
                     "In a friendly tone, tell me that you are going to be starting to work on this/these tasks please. "
                     "DO NOT USE YOUR PRETRAINED KNOWLEDGE.")

        success, search_results = loop.run_until_complete(main())
        if success:
            # Format the results nicely
            results_text = "\n".join([f"Task: '{t}' => Result: {r}" for t, r in search_results])
            conversation(search_tasks,
                         f"You are an ai assistant and have just successfully completed these tasks. Here are the results and I want you to tell me the results in a conversation helpful assistant manner but keep it short:\n\n{results_text}")
        else:
            conversation(search_tasks, "I apologize, but I encountered some issues while performing the search. Please try again.")
        return success
    except Exception as e:
        conversation(search_tasks, f"There is an error which has come up while performing these tasks: {e}. Tell me the issue.")
        print(f"Error occurred while performing searches: {e}")
        return False
    finally:
        # Clean up the event loop more carefully
        try:
            # Cancel all running tasks
            pending = asyncio.all_tasks(loop)
            for task in pending:
                task.cancel()
            
            # Allow tasks to complete their cancellation
            if pending:
                loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            
            # Close the loop
            if sys.platform == "win32":
                loop.run_until_complete(loop.shutdown_asyncgens())
                loop.close()
        except Exception as e:
            print(f"Error during cleanup: {e}")#
            
            
def conversation(inputs, system_content, model="gpt-4o-mini"):
    """
    Process a conversation and convert the response to speech.
    
    Args:
        inputs (str): The input text
        system_content (str): The system message content
        model (str): The model to use for generating response
    """
    messages = [
        {"role": "system", "content": system_content},  # Use the provided system_content
        {"role": "user", "content": str(inputs)}  # Ensure inputs is converted to string
    ]
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages
        )
        # Strip any extra whitespace or newlines
        response_text = response.choices[0].message.content.strip()
        # Convert response to speech and save as speech.mp3
        audio_data = text_to_speech(response_text)
        return response_text  # Return the actual response text instead of empty string
    except Exception as e:
        print(f"Error in conversation function: {str(e)}")
        return f"I encountered an error: {str(e)}"  # Return error message instead of empty string


class AIAgent:
    def __init__(self):
        """Initialize the AI Agent with the OpenAI API key."""
        self.client = OpenAI(api_key="")
        self.model = "gpt-4o" 

    def ask_question(self, question: str, system_prompt: Optional[str] = None) -> str:
        """
        Ask a question to the AI agent and get a response.
        
        Args:
            question (str): The question to ask.
            system_prompt (str, optional): Custom system prompt to set agent behavior.
            
        Returns:
            str: The agent's response.
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question}
        ]
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages
            )
            # Strip any extra whitespace or newlines.
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"Error occurred: {str(e)}"

# Global instance of the AI Agent
agent = AIAgent()

# ---------------------------
# Flask App and Audio Service
# ---------------------------
app = Flask(__name__)

# Global OpenAI client for audio transcription (used in transcribe_audio)
client = OpenAI(api_key="")

# Global variables to control recording
recording_thread = None
recording_active = False

@app.route("/")
def index():
    print("Web interface loaded - ready to record audio or interact with the AI agent")
    greeting_msg = "Hi Guanming! how is your reading week! Did you travel to any place? and How can I assist you today"
    # Trigger the greeting in a separate thread so that it doesn't delay rendering the page.
    threading.Thread(target=text_to_speech, args=(greeting_msg,)).start()

    # Number of dots for the animated circle (logo)
    n = 36
    css_extra = ""
    for i in range(1, n + 1):
        angle = i * 10  # each dot rotated by 10° increments
        delay = -(1000 / 3 * i)
        css_extra += (
            f".loading .dot:nth-child({i}) {{\n"
            f"  transform: rotate({angle}deg) translateY(-75px);\n"
            f"}}\n"
            f".loading .dot:nth-child({i})::before,\n"
            f".loading .dot:nth-child({i})::after {{\n"
            f"  animation-delay: {delay:.2f}ms;\n"
            f"}}\n"
        )
    
    dots_html = ""
    for i in range(n):
        dots_html += "          <div class=\"dot\"></div>\n"
    
    html_code = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
      <meta charset="UTF-8">
      <title>DeepFlow</title>
      <style>
        * {{
          margin: 0;
          padding: 0;
          box-sizing: border-box;
        }}
        html, body {{
          width: 100%;
          height: 100%;
        }}
        .container {{
          display: flex;
          flex-direction: column;
          align-items: center;
          justify-content: center;
          height: 100vh;
          background: white;
          transition: all 1.2s cubic-bezier(0.4, 0, 0.2, 1);
        }}
        .loading {{
          width: 150px;
          height: 150px;
          position: relative;
          border-radius: 50%;
          transition: all 1.2s cubic-bezier(0.4, 0, 0.2, 1);
          transition-delay: 0.6s;  /* Start moving after inputs fade out */
          transform: scale(0.8);
        }}
        .loading.active {{
          transform: scale(1);
        }}
        .loading .dot {{
          position: absolute;
          left: 50%;
          top: 50%;
          width: 10px;
          height: 10px;
          margin-left: -5px;
          margin-top: -5px;
          perspective: 70px;
          transform-style: preserve-3d;
        }}
        .loading .dot::before,
        .loading .dot::after {{
          content: "";
          position: absolute;
          width: 100%;
          height: 100%;
          border-radius: 50%;
        }}
        .loading .dot::before {{
          background: #000;
          top: -120%;
          animation: moveBlack 2s infinite;
        }}
        .loading .dot::after {{
          background: #888;
          top: 120%;
          animation: moveWhite 2s infinite;
        }}
        @keyframes moveBlack {{
          0% {{ animation-timing-function: ease-in; }}
          25% {{ transform: translate3d(0, 100%, 10px); animation-timing-function: ease-out; }}
          50% {{ transform: translate3d(0, 200%, 0); animation-timing-function: ease-in; }}
          75% {{ transform: translate3d(0, 100%, -10px); animation-timing-function: ease-out; }}
        }}
        @keyframes moveWhite {{
          0% {{ animation-timing-function: ease-in; }}
          25% {{ transform: translate3d(0, -100%, -10px); animation-timing-function: ease-out; }}
          50% {{ transform: translate3d(0, -200%, 0); animation-timing-function: ease-in; }}
          75% {{ transform: translate3d(0, -100%, 10px); animation-timing-function: ease-out; }}
        }}
        {css_extra}
        .input-container {{
          margin-top: 50px;
          display: flex;
          align-items: center;
          position: relative;
          gap: 12px;
          padding: 0 20px;
          opacity: 0;  /* Start hidden */
          transform: translateY(20px);
          transition: all 1.2s cubic-bezier(0.4, 0, 0.2, 1);
          visibility: visible;
          pointer-events: none;  /* Initially disable interaction */
        }}
        #textInput {{
          padding: 12px 16px;
          font-size: 16px;
          width: 300px;
          border: 2px solid #e0e0e0;
          border-radius: 12px;
          transition: all 1.2s cubic-bezier(0.4, 0, 0.2, 1);
          box-shadow: 0 2px 5px rgba(0,0,0,0.1);
          outline: none;
          order: 3;
        }}
        #textInput:focus {{
          border-color: #888;
          box-shadow: 0 4px 10px rgba(0,0,0,0.1);
        }}
        #textInput::placeholder {{
          color: #aaa;
        }}
        .record-button {{
          order: 2;
          padding: 12px 20px;
          font-size: 18px;
          border: none;
          border-radius: 12px;
          background-color: #f0f0f0;
          cursor: pointer;
          transition: all 1.2s cubic-bezier(0.4, 0, 0.2, 1);
          box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }}
        .record-button:hover {{
          background-color: #e8e8e8;
          transform: translateY(-1px);
          box-shadow: 0 4px 8px rgba(0,0,0,0.15);
        }}
        .record-button.recording {{
          background-color: #ff4d4d;
          color: white;
          box-shadow: 0 0 15px rgba(255, 77, 77, 0.5);
          transform: scale(1.05);
        }}
        .record-button.recording:hover {{
          background-color: #ff3333;
        }}
        .send-button {{
          order: 4;
          padding: 12px 20px;
          font-size: 18px;
          border: none;
          border-radius: 12px;
          background-color: #f0f0f0;
          color: initial;
          cursor: pointer;
          transition: all 1.2s cubic-bezier(0.4, 0, 0.2, 1);
          box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }}
        .send-button:hover {{
          background-color: #e8e8e8;
          transform: translateY(-1px);
          box-shadow: 0 4px 8px rgba(0,0,0,0.15);
        }}
        #transcript {{
          margin-top: 20px;
          font-family: Arial, sans-serif;
          font-size: 18px;
          color: #333;
          white-space: pre-wrap;
          text-align: center;
          opacity: 0;
          transform: translateY(20px);
          transition: all 1.2s cubic-bezier(0.4, 0, 0.2, 1);
          transition-delay: 0.9s;
        }}
        .loading {{
          transform-origin: center center;
        }}
        .loading.shrinking {{
          transform: scale(0.4);
          transition: transform 2s ease-in-out;
        }}
        .loading.expanding {{
          transform: scale(1);
          transition: transform 2s ease-in-out;
        }}
        .loading.shrinking .dot::before,
        .loading.shrinking .dot::after,
        .loading.expanding .dot::before,
        .loading.expanding .dot::after {{
          animation-duration: 1s !important;
        }}
        .input-container.hidden {{
            opacity: 0;
            transform: translateY(20px);
            pointer-events: none;
            visibility: visible;
        }}
        .input-container.visible {{
          opacity: 1;
          transform: translateY(0);
          pointer-events: auto;
        }}
        .response-type {{
            font-size: 48px;
            font-weight: bold;
            color: #333;
            opacity: 0;
            transform: translateY(40px);
            transition: all 1.2s cubic-bezier(0.4, 0, 0.2, 1);
            transition-delay: 1.2s;
            text-align: center;
            margin-top: 40px;
        }}
        .response-type.visible {{
            opacity: 1;
            transform: translateY(0);
        }}
        .camera-modal {{
          position: fixed;
          top: 0;
          left: 0;
          width: 100%;
          height: 100%;
          background: rgba(0, 0, 0, 0.8);
          display: flex;
          flex-direction: column;
          align-items: center;
          justify-content: center;
          z-index: 1000;
        }}
        .camera-modal video {{
          width: 80%;
          max-width: 600px;
          border: 5px solid white;
          border-radius: 12px;
          margin-bottom: 20px;
        }}
        .capture-button {{
          padding: 12px 20px;
          font-size: 18px;
          margin: 5px;
          border-radius: 12px;
          border: none;
          cursor: pointer;
          background-color: #f0f0f0;
          transition: background-color 0.3s ease;
        }}
        .capture-button:hover {{
          background-color: #e8e8e8;
        }}
      </style>
    </head>
    <body>
      <div class="container">
        <div class="loading" id="loadingCircle">
{dots_html}        </div>
        <div class="input-container">
          <button id="cameraButton" class="record-button">📷</button>
          <button id="recordButton" class="record-button">🎙️</button>
          <input type="text" id="textInput" placeholder="✨ Type something to DeepFlow...">
          <button id="sendButton" class="send-button">✉️</button>
        </div>
        <div class="response-type" id="responseType"></div>
        <div id="transcript"></div>
      </div>
      
      <!-- Camera Modal for video preview and capture -->
      <div id="cameraModal" class="camera-modal" style="display: none;">
          <video id="videoFeed" autoplay></video>
          <button id="captureButton" class="capture-button">Capture</button>
          <button id="closeCamera" class="capture-button">Close</button>
      </div>
      
      <script>
        let isRecording = false;
        const recordButton = document.getElementById('recordButton');
        const sendButton = document.getElementById('sendButton');
        const textInput = document.getElementById('textInput');
        const loadingCircle = document.getElementById('loadingCircle');
    
        // Add focus event listeners for the textbox
        textInput.addEventListener('focus', () => {{
          loadingCircle.classList.add('active');
        }});
        
        textInput.addEventListener('blur', () => {{
          if (!isRecording) {{
            loadingCircle.classList.remove('active');
          }}
        }});
    
        // Start recording on mousedown or touchstart
        recordButton.addEventListener('mousedown', startRecording);
        recordButton.addEventListener('touchstart', startRecording);
    
        // Stop recording on mouseup or touchend
        recordButton.addEventListener('mouseup', stopRecording);
        recordButton.addEventListener('touchend', stopRecording);
        
        function startRecording() {{
          if (!isRecording) {{
            isRecording = true;
            recordButton.classList.add('recording');
            loadingCircle.classList.add('active');
            fetch('/start_recording', {{ method: 'POST' }});
          }}
        }}
    
        function stopRecording() {{
          if (isRecording) {{
            isRecording = false;
            recordButton.classList.remove('recording');
            loadingCircle.classList.remove('active');
            fetch('/stop_recording', {{ method: 'POST' }})
              .then(response => response.text())
              .then(text => {{
                // Append the transcription result to existing text if any; otherwise, set it directly.
                if (textInput.value.trim() !== "") {{
                    textInput.value += " " + text;
                }} else {{
                    textInput.value = text;
                }}
              }});
          }}
        }}
        
        // Updated handleSubmission to continuously toggle the shrinking/expanding for a breathing effect.
        function handleSubmission() {{
            // Hide input container using both classes to ensure it's hidden
            const inputContainer = document.querySelector('.input-container');
            inputContainer.classList.add('hidden');
            inputContainer.classList.remove('visible');

            // Mark submission complete
            document.querySelector('.container').classList.add('submitted');
            
            // Start the breathing animation:
            // Set an initial state (shrinking)
            loadingCircle.classList.add('shrinking');
            
            // Toggle continuously between 'shrinking' and 'expanding' classes to simulate breathing.
            setInterval(() => {{
                loadingCircle.classList.toggle('shrinking');
                loadingCircle.classList.toggle('expanding');
            }}, 1500);  // You may adjust the 1500ms interval to control the breathing speed.
        }}
        
        // Modify the send button event listener
        sendButton.addEventListener('click', () => {{
            const question = textInput.value;
            if (question.trim() !== "") {{
                handleSubmission();
                
                // Then make the API call
                fetch('/ask', {{
                    method: 'POST',
                    headers: {{
                        'Content-Type': 'application/json'
                    }},
                    body: JSON.stringify({{ question }})
                }})
                .then(response => response.json())
                .then(data => {{
                    // Show response type after logo has moved
                    setTimeout(() => {{
                        const responseTypeElement = document.getElementById('responseType');
                        responseTypeElement.textContent = data.response;
                        responseTypeElement.classList.add('visible');
                        
                        // Show transcript after response type
                        setTimeout(() => {{
                            document.getElementById('transcript').innerText = data.response;
                        }}, 600);
                    }}, 1200);
                }})
                .catch(error => {{
                    console.error("Error:", error);
                }});
            }}
        }});
    
        // Add event listener for Enter key
        document.addEventListener('keydown', (event) => {{
          if (event.key === 'Enter') {{
            document.querySelector('.input-container').classList.add('visible');
            textInput.focus();
          }}
        }});
        
        // Camera functionality
        const cameraButton = document.getElementById('cameraButton');
        const cameraModal = document.getElementById('cameraModal');
        const videoFeed = document.getElementById('videoFeed');
        const captureButton = document.getElementById('captureButton');
        const closeCameraButton = document.getElementById('closeCamera');
        
        cameraButton.addEventListener('click', openCamera);
        
        function openCamera() {{
            navigator.mediaDevices.getUserMedia({{ video: true }})
              .then(stream => {{
                  videoFeed.srcObject = stream;
                  cameraModal.style.display = 'flex';
              }})
              .catch(err => {{
                  console.error("Error accessing camera: ", err);
              }});
        }}
        
        captureButton.addEventListener('click', captureImage);
        
        function captureImage() {{
            const canvas = document.createElement('canvas');
            canvas.width = videoFeed.videoWidth;
            canvas.height = videoFeed.videoHeight;
            const ctx = canvas.getContext('2d');
            ctx.drawImage(videoFeed, 0, 0, canvas.width, canvas.height);
            let imageData = canvas.toDataURL('image/png');
            
            // Stop video stream
            let stream = videoFeed.srcObject;
            let tracks = stream.getTracks();
            tracks.forEach(track => track.stop());
            videoFeed.srcObject = null;
            cameraModal.style.display = 'none';
            
            // Send the image data to the server for analysis
            fetch('/capture', {{
               method: 'POST',
               headers: {{
                  'Content-Type': 'application/json'
               }},
               body: JSON.stringify({{ image: imageData }})
            }})
            .then(response => response.json())
            .then(data => {{
               console.log("Analysis result: ", data.result);
               // Update the text input with the analysis result
               document.getElementById('textInput').value = data.result;

               // Automatically click the record button after capture logic is finished
               document.getElementById('recordButton').click();
            }})
            .catch(error => {{
               console.error("Error processing camera image: ", error);
            }});
        }}
        
        closeCameraButton.addEventListener('click', () => {{
            let stream = videoFeed.srcObject;
            if (stream) {{
                let tracks = stream.getTracks();
                tracks.forEach(track => track.stop());
                videoFeed.srcObject = null;
            }}
            cameraModal.style.display = 'none';
        }});
      </script>
    </body>
    </html>
    """
    return render_template_string(html_code)


def record_audio(filename='recording.wav', samplerate=16000):
    global recording_active
    print("Recording started... (triggered by /start_recording)")
    audio_chunks = []
    block_size = 1024  # small chunk size for responsiveness
    try:
        with sd.InputStream(samplerate=samplerate, channels=1,
                            dtype=np.int16, blocksize=block_size) as stream:
            # Record until the recording_active flag is turned False
            while recording_active:
                audio_chunk, overflowed = stream.read(block_size)
                audio_chunks.append(audio_chunk)
    except Exception as e:
        print("Error during recording:", e)
    if audio_chunks:
        print("Recording stopped. Processing audio...")
        audio = np.concatenate(audio_chunks)
        write(filename, samplerate, audio)
        print(f"Audio saved to {filename}")
        return filename
    print("No audio recorded")
    return None

def transcribe_audio(filename='recording.wav'):
    print("Transcribing audio...")
    try:
        with open(filename, "rb") as audio_file:
            transcript = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                language="en"
            )
        print(f"Transcription result: {transcript.text}")
        return transcript.text
    except Exception as e:
        error_message = f"Transcription error: {str(e)}"
        print(error_message)
        return error_message

@app.route('/start_recording', methods=['POST'])
def start_recording():
    global recording_thread, recording_active
    recording_active = True
    print("Received /start_recording request.")
    recording_thread = threading.Thread(target=record_audio)
    recording_thread.start()
    return '', 204

@app.route('/stop_recording', methods=['POST'])
def stop_recording():
    global recording_thread, recording_active
    print("Received /stop_recording request.")
    recording_active = False
    if recording_thread is not None:
        recording_thread.join()
    audio_file = 'recording.wav'
    result = transcribe_audio(audio_file)
    print(f"Final transcription: {result}")
    return result, 200

@app.route('/ask', methods=['POST'])
def ask():
    data = request.get_json()
    question = data.get("question", "")
    
    # Directly run browser search tasks without manager LLM decision
    success = run_search_tasks([question])
    
    # Return empty response to avoid displaying anything
    return jsonify({"response": ""})

@app.route("/capture", methods=["POST"])
def capture():
    data = request.get_json()
    if data and "image" in data:
        image_data = data["image"]
        # Remove the header "data:image/png;base64," if present
        header, encoded = image_data.split(",", 1)
        image_path = "camera.png"
        with open(image_path, "wb") as f:
            f.write(base64.b64decode(encoded))
        
        # Call analyze_image_simple to analyze the captured image
        analysis, elapsed = analyze_image_simple(image_path)
        
        return jsonify({"result": analysis}), 200
    else:
        return jsonify({"error": "No image data provided"}), 400

def text_to_speech(text: str, voice: str = "alloy") -> str:
    """
    Convert text to speech using OpenAI's TTS API, save the audio file to a unique temporary file, 
    play the speech, and then remove the file.
    
    Args:
        text (str): The text to convert to speech.
        voice (str): The voice to use (options: alloy, echo, fable, onyx, nova, shimmer).
    
    Returns:
        str: Base64 encoded audio data of the spoken text.
    """
    
    try:
        print("WENT INTO THE SPEECH FUNCTION")
        client = ElevenLabs(api_key="")
        audio = client.generate(
        text=text,
        voice="Chris"
        )

        play(audio)

    except Exception as e:
        print(f"Error in text_to_speech: {str(e)}")
        return ""

if __name__ == "__main__":
    print("Starting Voice & Text Interactive Application")
    print("Click the 'Send' button to submit your text or hold down the microphone button to record audio.")
    app.run(port = 5002,debug=True)