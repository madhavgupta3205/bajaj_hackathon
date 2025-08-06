# Vercel Deployment Guide

## 🚀 Deploy to Vercel

### Prerequisites

1. Install Vercel CLI: `npm install -g vercel`
2. Login to Vercel: `vercel login`

### Deployment Steps

1. **Add Environment Variables in Vercel Dashboard:**

   - Go to your Vercel project dashboard
   - Navigate to Settings > Environment Variables
   - Add these secrets:
     - `GEMINI_API_KEY`: Your Google Gemini API key
     - `HACKRX_BEARER_TOKEN`: Your authentication token

2. **Deploy:**
   ```bash
   vercel --prod
   ```

### Environment Variables Needed

- `GEMINI_API_KEY`: Your Google Gemini API key (REQUIRED)
- `HACKRX_BEARER_TOKEN`: Bearer token for API authentication

### API Endpoints

Once deployed, your API will be available at:

- `GET /api/v1/health` - Health check
- `POST /api/v1/hackrx/run` - Main processing endpoint

Base URL: `https://your-app.vercel.app/api/v1`

### Sample Request

```bash
curl -X POST "https://your-app.vercel.app/api/v1/hackrx/run" \
-H "Content-Type: application/json" \
-H "Authorization: Bearer a928ab38f03560bdb4b9c3930ca021cf0f1c753febc6a637fb996cb4f30c35c8" \
-d '{
  "documents": "https://example.com/document.pdf",
  "questions": ["What is this document about?"]
}'
```

### Troubleshooting

- **Cold starts**: First request may take longer due to model loading
- **Timeouts**: Function timeout is set to 60 seconds
- **Memory**: Large models may require more memory allocation

### Local Development

1. Copy `.env.template` to `.env`
2. Fill in your API keys
3. Run: `uvicorn app.main:app --reload`

Base URL: `http://localhost:8000/api/v1`

Test endpoints:
- Health: `curl http://localhost:8000/api/v1/health`
- Main: `curl -X POST http://localhost:8000/api/v1/hackrx/run ...`
