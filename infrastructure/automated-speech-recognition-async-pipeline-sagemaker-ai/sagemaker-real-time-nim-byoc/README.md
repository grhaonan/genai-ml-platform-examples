# Combined NVIDIA NIM ASR for SageMaker (HTTP + gRPC)

Single SageMaker endpoint exposing both transports to a NVIDIA NIM ASR container:
- GET /ping – health check
- POST /invocations – auto route (prefers HTTP; routes to gRPC for large payloads or speaker diarization)
- POST /invocations/http – force NIM HTTP path (simple transcription only)
- POST /invocations/grpc – force NIM gRPC path (supports speaker diarization)

Container runs a lightweight aiohttp server that starts NIM, waits until ready, then intelligently routes to:
- NIM HTTP: http://127.0.0.1:9000/v1/audio/transcriptions (simple transcription, <5MB files)
- NIM gRPC: 127.0.0.1:50051 via nvidia-riva-client (speaker diarization, any file size)

## Prerequisites
- Docker and AWS CLI configured
- ECR permissions
- SageMaker execution role ARN
- NGC_API_KEY exported (NVIDIA NGC)
- Region: us-east-1 (adjust if needed)

## Build & Push
```bash
# From this directory
docker build -t nim-sagemaker-asr .

# Create repo once (ignore error if exists)
aws ecr create-repository --repository-name nim-sagemaker-asr --region us-east-1 || true

# Tag & push
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
REGION=us-east-1
ECR=${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com

docker tag nim-sagemaker-asr:latest ${ECR}/nim-sagemaker-asr:latest
aws ecr get-login-password --region ${REGION} | docker login --username AWS --password-stdin ${ECR}
docker push ${ECR}/nim-sagemaker-asr:latest
```

## Deploy (ml.g5.48xlarge)
```bash
# Get existing role from current deployment or use your role ARN
ROLE_ARN=$(aws sagemaker describe-model --model-name nim-asr-combined-model --region us-east-1 --query 'ExecutionRoleArn' --output text 2>/dev/null || echo "arn:aws:iam::<ACCOUNT_ID>:role/service-role/AmazonSageMaker-ExecutionRole-XXXXXXXXXXXX")
IMAGE_URI="${ECR}/nim-sagemaker-asr:latest"
REGION=us-east-1
TIMESTAMP=$(date +%s)

# Create new model with timestamp to avoid conflicts
aws sagemaker create-model \
  --region ${REGION} \
  --model-name nim-asr-combined-model-${TIMESTAMP} \
  --primary-container "Image=${IMAGE_URI},Environment={NGC_API_KEY=$NGC_API_KEY}" \
  --execution-role-arn ${ROLE_ARN}

# Create new endpoint config
aws sagemaker create-endpoint-config \
  --region ${REGION} \
  --endpoint-config-name nim-asr-combined-config-${TIMESTAMP} \
  --production-variants VariantName=primary,ModelName=nim-asr-combined-model-${TIMESTAMP},InitialInstanceCount=1,InstanceType=ml.g5.48xlarge,ContainerStartupHealthCheckTimeoutInSeconds=2400,ModelDataDownloadTimeoutInSeconds=2400

# Create or update endpoint
aws sagemaker create-endpoint \
  --region ${REGION} \
  --endpoint-name nim-asr-combined-endpoint \
  --endpoint-config-name nim-asr-combined-config-${TIMESTAMP} 2>/dev/null || \
aws sagemaker update-endpoint \
  --region ${REGION} \
  --endpoint-name nim-asr-combined-endpoint \
  --endpoint-config-name nim-asr-combined-config-${TIMESTAMP}

# Wait for endpoint to be ready
aws sagemaker wait endpoint-in-service \
  --endpoint-name nim-asr-combined-endpoint \
  --region ${REGION}

echo "✅ Endpoint ready: nim-asr-combined-endpoint"
```

## Working Example (Tested ✅)
Both HTTP and gRPC routes confirmed working with the same multipart payload.

⚠️ **IMPORTANT: File size limit is ~5MB for HTTP route**
- HTTP route: Files >5MB will fail with "Request Entity Too Large" 
- gRPC route: Supports larger files (tested up to 25MB)
- For files >25MB: Use S3 + async inference

### Speaker Diarization Support (gRPC only)
The gRPC route supports speaker diarization with the following parameters:
- `speaker_diarization`: Set to "true" to enable (default: false)
- `max_speakers`: Maximum number of speakers to detect (default: 10)
- **Note**: Use `language_code` field (not `language`) for proper parsing
```bash
# Setup
REGION=us-east-1
ENDPOINT=nim-asr-combined-endpoint
AUDIO=/home/ubuntu/workspace/asr/test.wav  # Use smaller file
BOUNDARY="----WebKitFormBoundary$(uuidgen | tr -d '-')"
MP=/home/ubuntu/workspace/asr/asr_diarization.bin

{
  printf "%s\r\n" "--$BOUNDARY"
  printf "%s\r\n" "Content-Disposition: form-data; name=\"file\"; filename=\"test.wav\""
  printf "%s\r\n\r\n" "Content-Type: audio/wav"
  cat "$AUDIO"
  printf "\r\n%s\r\n" "--$BOUNDARY"
  printf "%s\r\n\r\n" "Content-Disposition: form-data; name=\"language_code\""
  printf "%s\r\n" "en-US"
  printf "\r\n%s\r\n" "--$BOUNDARY"
  printf "%s\r\n\r\n" "Content-Disposition: form-data; name=\"speaker_diarization\""
  printf "%s\r\n" "true"
  printf "\r\n%s\r\n" "--$BOUNDARY"
  printf "%s\r\n\r\n" "Content-Disposition: form-data; name=\"max_speakers\""
  printf "%s\r\n" "4"
  printf "%s\r\n" "--$BOUNDARY--"
} > "$MP"

```

### Intelligent Routing
The endpoint automatically chooses the best protocol:
```bash
# Auto-routing (recommended) - chooses HTTP for small files, gRPC for large files or diarization
aws sagemaker-runtime invoke-endpoint \
  --region ${REGION} \
  --endpoint-name ${ENDPOINT} \
  --content-type "multipart/form-data; boundary=$BOUNDARY" \
  --body fileb://$MP \
  result.json

# Force HTTP route (simple transcription only, <5MB)
aws sagemaker-runtime invoke-endpoint \
  --region ${REGION} \
  --endpoint-name ${ENDPOINT} \
  --custom-attributes "X-Amzn-SageMaker-Custom-Attributes=/invocations/http" \
  --content-type "multipart/form-data; boundary=$BOUNDARY" \
  --body fileb://$MP \
  result_http.json

# Force gRPC route (speaker diarization, any file size)
aws sagemaker-runtime invoke-endpoint \
  --region ${REGION} \
  --endpoint-name ${ENDPOINT} \
  --custom-attributes "X-Amzn-SageMaker-Custom-Attributes=/invocations/grpc" \
  --content-type "multipart/form-data; boundary=$BOUNDARY" \
  --body fileb://$MP \
  result_grpc.json
```


### Speaker Diarization Output Format
When speaker diarization is enabled, the response includes word-level timing and speaker information:
```json
{
  "predictions": [{
    "results": [{
      "alternatives": [{
        "transcript": "I will try to, uh, okay, so let's off video.",
        "confidence": 0.18058110773563385,
        "words": [
          {
            "word": "I",
            "start_time": 2.08,
            "end_time": 2.16,
            "confidence": 0.2953037917613983,
            "speaker_tag": 0
          },
          {
            "word": "will",
            "start_time": 2.56,
            "end_time": 2.64,
            "confidence": 0.1699565052986145,
            "speaker_tag": 0
          }
        ]
      }],
      "is_final": true,
      "channel_tag": 1
    }],
    "model_version": "parakeet-1-1b-ctc-en-us"
  }]
}
```

## Notes

### Protocol Selection
- **HTTP route**: Simple transcription, <5MB files, faster for small files
- **gRPC route**: Speaker diarization, any file size, word-level timing
- **Auto-routing**: Automatically selects best protocol based on file size and parameters

### Supported Input Formats
- **Multipart form-data** (recommended): Up to 25MB, supports all parameters
- **JSON with base64**: <4.5MB only (base64 overhead), simple integration
- **Raw binary**: Direct audio upload, HTTP route only

### Endpoints Exposed
- `GET /ping` – Health check
- `POST /invocations` – Auto-routing (recommended)
- `POST /invocations/http` – Force HTTP route
- `POST /invocations/grpc` – Force gRPC route

### Environment Variables
- `NGC_API_KEY` – NVIDIA NGC API key (required)
- `NIM_HOST` – NIM service host (default: 127.0.0.1)
- `NIM_HTTP_PORT` – NIM HTTP port (default: 9000)
- `RIVA_GRPC_PORT` – NIM gRPC port (default: 50051)
- `SAGEMAKER_BIND_TO_PORT` – SageMaker port (default: 8080)
- `NIM_TAGS_SELECTOR` – Model configuration (default: name=parakeet-1-1b-ctc-en-us,mode=ofl)

### Troubleshooting
- **"Request Entity Too Large"**: File >5MB on HTTP route → Use gRPC route
- **"Unavailable model"**: Check language_code format (use "en-US", not "en")
- **No speaker diarization**: Ensure `speaker_diarization=true` and using gRPC route
- **Parsing errors**: Check multipart field names (use `language_code`, not `language`)

## Cleanup
```bash
REGION=us-east-1

# Delete endpoint
aws sagemaker delete-endpoint --endpoint-name nim-asr-combined-endpoint --region ${REGION}

# List and delete all endpoint configs (handles timestamped versions)
aws sagemaker list-endpoint-configs --region ${REGION} --name-contains nim-asr-combined-config --query 'EndpointConfigs[].EndpointConfigName' --output text | \
xargs -n1 -I{} aws sagemaker delete-endpoint-config --endpoint-config-name {} --region ${REGION}

# List and delete all models (handles timestamped versions)
aws sagemaker list-models --region ${REGION} --name-contains nim-asr-combined-model --query 'Models[].ModelName' --output text | \
xargs -n1 -I{} aws sagemaker delete-model --model-name {} --region ${REGION}

echo "✅ Cleanup complete"
```
