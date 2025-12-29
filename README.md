# Translation App

A translation application powered by Babel-9B-Chat with a FastAPI backend.

## Features

- Translate text between multiple languages
- Real-time translation using Bable-9B-Chat model
- RESTful API with FastAPI


## Project Structure

```
tower-bable-ml-system/
├── main.py                 # Core translation logic
├── api.py                  # FastAPI backend
├── requirements.txt        # Python dependencies
```

## Setup Instructions

### Backend Setup

1. Install Python dependencies:
```bash
pip install -r requirements.txt
```

2. Run the FastAPI server:
```bash
python api.py
```

The API will be available at `http://localhost:9000`

#### API Endpoints

- `GET /` - API information
- `GET /health` - Health check
- `POST /translate` - Translate text
  ```json
  {
    "text": "Hello world",
    "source_language": "English",
    "target_language": "Bangla"
  }
  ```

## Usage

1. Start the backend server (runs on port 8000)

## Supported Languages

- English
- Bangla
- Hindi
- Spanish
- French
- German
- Chinese
- Japanese
- Arabic
- Portuguese

## Requirements

### Backend
- Python 3.8+
- CUDA-compatible GPU (recommended for faster inference)
- 16GB+ RAM recommended


## Development

### Backend Development

To run the backend with auto-reload:
```bash
uvicorn api:app --reload --host 0.0.0.0 --port 9000
```

## Production Build

### Frontend
```bash
cd frontend
npm run build
npm start
```

## Notes

- The model will be downloaded automatically on first run (approximately 15GB)
- Translation speed depends on your hardware (GPU recommended)
- The backend uses greedy decoding to minimize hallucinations


## Troubleshooting

### Backend Issues
- Ensure all Python dependencies are installed
- Check if the Qwen model is downloaded correctly
- Verify CUDA is available if using GPU

## License

This project uses the Bable-9B-Chat model. Please refer to the model's license for usage terms.
