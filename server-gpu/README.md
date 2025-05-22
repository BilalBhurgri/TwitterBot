# Flask Server for Paper Processing

This server provides endpoints for processing papers and generating summaries using language models.

## Setup

1. Make sure you're in the project root directory
2. Activate the virtual environment:
   ```bash
   source .venv/bin/activate
   ```
3. Ensure all dependencies are installed:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Server

To run the server:

```bash
python server/app.py
```

The server will start on port 5000 by default. You can change this by setting the `PORT` environment variable.

## Available Endpoints

### Health Check
- **URL**: `/health`
- **Method**: `GET`
- **Description**: Check if the server is running and if the model is loaded
- **Response**: 
  ```json
  {
    "status": "healthy",
    "message": "Server is running",
    "model_loaded": true
  }
  ```

### Process Paper
- **URL**: `/process-paper/<key>`
- **Method**: `POST`
- **Description**: Process a paper and generate a summary using the s3 key, which is in the format papers/arxiv_id.txt
- **Request Body**:
  ```json
  {
    "paper_text": "Your paper text here...",
    "evaluate": true  // Optional: whether to generate an evaluation
  }
  ```
- **Response**: 
  ```json
  {
    "status": "success",
    "summary": "Generated summary...",
    "evaluation": "Evaluation scores..."  // Only if evaluate=true
  }
  ```

### Set Model
- **URL**: `/set-model`
- **Method**: `POST`
- **Description**: Change the model being used for processing
- **Request Body**:
  ```json
  {
    "model_name": "Qwen/Qwen3-1.7B"  // or "Qwen/Qwen3-4B"
  }
  ```
- **Response**: 
  ```json
  {
    "status": "success",
    "message": "Model set to Qwen/Qwen3-1.7B"
  }
  ```

## Example Usage

1. Check server health:
   ```bash
   curl http://localhost:5000/health
   ```

2. Process a paper:
   ```bash
   curl -X POST http://localhost:5000/process-paper \
     -H "Content-Type: application/json" \
     -d '{"paper_text": "Your paper text here...", "evaluate": true}'
   ```

3. Change the model:
   ```bash
   curl -X POST http://localhost:5000/set-model \
     -H "Content-Type: application/json" \
     -d '{"model_name": "Qwen/Qwen3-4B"}'
   ```

## Notes

- The server uses Qwen3-1.7B by default
- Models are loaded lazily (only when needed)
- The server supports both summary generation and evaluation
- All endpoints return JSON responses
- Error handling is implemented for all endpoints

## Development

To add new endpoints or modify existing ones, edit `app.py`. Make sure to:
1. Add proper error handling
2. Document the endpoint in this README
3. Add appropriate logging
4. Consider rate limiting for production use 