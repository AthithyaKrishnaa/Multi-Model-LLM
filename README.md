#  Multimodal AI Assistant

A powerful multimodal AI chatbot that can process and understand text, images, videos, and audio. Built with cutting-edge open-source models and powered by Gradio for an intuitive user interface.

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![Gradio](https://img.shields.io/badge/Gradio-UI-orange.svg)

## ✨ Features

- **💬 Text Chat**: Engage in natural conversations powered by the Phi-3 language model
- **🖼️ Image Captioning**: Generate detailed descriptions of images using BLIP
- **🎥 Video Analysis**: Process videos frame-by-frame with intelligent summarization
- **🎤 Audio Transcription**: Convert speech to text with Whisper (supports multiple languages)
- **🎨 Beautiful UI**: Modern gradient interface with WhatsApp-inspired chat design
- **☁️ Cloud Ready**: Designed to run on Google Colab with GPU acceleration

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended for optimal performance)
- 8GB+ RAM

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/multimodal-ai-assistant.git
   cd multimodal-ai-assistant
   ```

2. **Install dependencies**
   ```bash
   pip install -q -U transformers accelerate pillow faster-whisper gradio opencv-python
   ```

3. **Run the notebook**
   - Open `MultiModel_LLM.ipynb` in Jupyter or Google Colab
   - Execute all cells
   - The Gradio interface will launch automatically

### Google Colab (Recommended)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)

1. Upload `MultiModel_LLM.ipynb` to Google Colab
2. Set runtime to **GPU** (Runtime → Change runtime type → T4 GPU)
3. Run all cells
4. Access the public Gradio link generated

## 🛠️ Models Used

| Capability | Model | Description |
|------------|-------|-------------|
| **Text Generation** | [microsoft/Phi-3-mini-4k-instruct](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct) | Compact 3.8B parameter language model |
| **Image Captioning** | [Salesforce/blip-image-captioning-base](https://huggingface.co/Salesforce/blip-image-captioning-base) | BLIP model for image understanding |
| **Audio Transcription** | [Whisper (base)](https://github.com/guillaumekln/faster-whisper) | Fast speech-to-text with multilingual support |
| **Video Processing** | OpenCV + Custom frame extraction | Intelligent frame sampling and summarization |

## 📖 Usage Examples

### Text Chat
Simply type your message and press **Send** or hit Enter.

```
User: What is machine learning?
Assistant: Machine learning is a subset of artificial intelligence...
```

### Image Analysis
Upload an image using the **📷 Image** input and click **Send**.

```
Output: 🖼️ Image: A golden retriever playing in a park with a red ball during sunset
```

### Video Processing
Upload a video file (MP4 recommended) to get a frame-by-frame analysis.

```
Output: 
🎬 Video Analysis
Frame 0: Person walking in a city street
Frame 1: Close-up of traffic lights
...
Simple summary: A person navigating urban traffic
```

### Audio Transcription
Upload an audio file (WAV, MP3, etc.) to transcribe speech.

```
Output:
🎤 Audio Transcription
Language: English
Duration: ~45 seconds
Full transcript: Hello, this is a test recording...
```

## 🎨 Interface

The application features:
- **Gradient Purple Background**: Eye-catching modern design
- **Chat History**: Scrollable conversation log
- **Multi-input Support**: Send text, images, videos, or audio in one interface
- **Clear Button**: Reset conversation with one click
- **Real-time Processing**: See results as they're generated

## ⚙️ Configuration

### Adjust Model Parameters

In the notebook, you can modify:

```python
# Text generation settings
text_pipeline = pipeline("text-generation",
    model="microsoft/Phi-3-mini-4k-instruct",
    device_map="auto",
    torch_dtype=torch.bfloat16
)

# Whisper transcription quality
segments, info = whisper.transcribe(audio_path, beam_size=3)  # Increase for better accuracy
```

### Video Frame Sampling

```python
sample_interval = max(1, total_frames // 5)  # Modify to extract more/fewer frames
```

## 🔧 Troubleshooting

### Out of Memory Errors
- Reduce video resolution or length
- Use smaller models (e.g., Whisper tiny instead of base)
- Restart the runtime and clear cache

### Slow Performance
- Ensure GPU is enabled (Colab: Runtime → Change runtime type)
- Reduce `max_tokens` in text generation
- Process shorter videos

### Audio Not Detected
- Check audio file format (WAV works best)
- Ensure audio contains speech
- Try increasing Whisper beam_size

## 📦 Dependencies

```
transformers >= 4.30.0
accelerate >= 0.20.0
pillow >= 9.0.0
faster-whisper >= 0.9.0
gradio >= 3.50.0
opencv-python >= 4.7.0
torch >= 2.0.0
```



---

⭐ If you find this project helpful, please consider giving it a star!
