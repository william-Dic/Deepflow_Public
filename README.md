# DeepFlow - Fully Autonomous Multimodal Input Agent

## How to Use

1. **Install Dependencies**:
   - Clone the repo:
     ```bash
     git clone https://github.com/your-username/Deepflow_Public.git
     cd Deepflow_Public
     ```
   - Install requirements:
     ```bash
     pip install -r requirements.txt
     ```

2. **Set Up Environment**:
   - Option 1: **Using `.env` file** (recommended for security):
     - Create a `.env` file in the root of the project with your OpenAI and Eleven Labs API keys:
       ```
       OPENAI_API_KEY=your-openai-api-key
       ELEVENLABS_API_KEY=your-elevenlabs-api-key
       ```

   - Option 2: **Directly in `deepflow.py`** (not recommended for production):
     - Alternatively, you can set your API keys directly in the `deepflow.py` file:
       ```python
       import os
       os.environ["OPENAI_API_KEY"] = "your-openai-api-key"
       os.environ["ELEVENLABS_API_KEY"] = "your-elevenlabs-api-key"

3. **Run the Application**:
   - Start the Flask app:
     ```bash
     python app.py
     ```
   - Access the app at `http://localhost:5002`.
