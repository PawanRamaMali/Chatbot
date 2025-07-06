# 🤖 Neural Chatbot

**Neural Chatbot** is a production-ready, intelligent AI chatbot built with TensorFlow/Keras, featuring advanced neural networks, modern web interface, and comprehensive training capabilities — completely self-hosted and customizable.

> ✅ 100% self-hosted & customizable  
> 🧠 Neural network-powered intent classification  
> 🌐 Modern web interface with real-time chat  
> 🚀 Production-ready with Docker & REST API  
> 📊 Advanced training with visualization  
> 🔧 Extensible architecture

---

## 📦 Features

- **Neural Network Intelligence**: Multi-layer perceptron with dropout and batch normalization
- **Intent Classification**: Advanced NLP preprocessing with confidence scoring
- **Modern Web UI**: Responsive chat interface with session management
- **REST API**: Comprehensive API with OpenAPI documentation
- **Production Ready**: Docker deployment, health monitoring, and logging
- **Training Pipeline**: Advanced training with visualization and evaluation
- **Conversation Management**: User-specific history and analytics
- **Data Augmentation**: Automatic pattern expansion for better accuracy

---

## 🚀 Quickstart

### 1. Clone the repo
```bash
git clone https://github.com/PawanRamaMali/Chatbot.git
cd Chatbot
```

### 2. Install dependencies
```bash
pip install -e .
```

### 3. Train the model
```bash
neural-chatbot train --plot
```

### 4. Start chatting
```bash
# Interactive chat
neural-chatbot chat

# Or start web server
neural-chatbot serve
```

### 5. Test the API
```bash
# Send a message
curl -X POST 'http://localhost:5000/api/chat' \
  -H 'Content-Type: application/json' \
  -d '{"message": "Hello, how are you?"}'

# Check status
curl http://localhost:5000/api/status

# Get conversation history
curl http://localhost:5000/api/conversation
```

---

## 🐳 Using Docker

```bash
# Quick start with Docker Compose
docker-compose up -d

# Check logs
docker-compose logs -f chatbot

# Production with monitoring
docker-compose --profile monitoring up -d

# Scale for production
docker-compose up -d --scale chatbot=3
```

**Access the services:**
- Chatbot API: http://localhost:5000
- Web Interface: http://localhost:5000
- Grafana (monitoring): http://localhost:3000
- Prometheus (metrics): http://localhost:9090

---

## 📁 Project Structure

```
Chatbot/
├── src/neural_chatbot/           # Main package
│   ├── core/                     # Core components
│   │   ├── chatbot.py           # Main chatbot orchestrator
│   │   ├── model.py             # Neural network model
│   │   ├── processor.py         # Data preprocessing
│   │   └── trainer.py           # Training pipeline
│   ├── api/                      # REST API
│   │   ├── app.py               # Flask application
│   │   ├── routes.py            # API endpoints
│   │   └── middleware.py        # Request middleware
│   ├── config/                   # Configuration
│   │   ├── settings.py          # Settings management
│   │   └── config.yaml          # Default configuration
│   ├── utils/                    # Utilities
│   ├── web/                      # Web interface
│   │   ├── templates/           # HTML templates
│   │   └── static/              # CSS/JS assets
│   └── data/                     # Training data
│       └── intents.json         # Intent patterns
├── tests/                        # Test suite
├── deployment/                   # Deployment configs
├── docker-compose.yml
├── Dockerfile
└── README.md
```

---

## ⚙️ API Endpoints

### `GET /`
Get API information and available endpoints

### `GET /health`
Health check endpoint with component status

### `GET /api/status`
Detailed chatbot status and statistics

### `POST /api/chat`
Send message to chatbot and get response

**Body:**
```json
{
  "message": "Hello, how are you?",
  "user_id": "optional-user-id",
  "context": {}
}
```

**Response:**
```json
{
  "response": "Hello! I'm doing great, thank you for asking!",
  "intent": "greeting",
  "confidence": 0.95,
  "timestamp": "2024-01-15T10:30:00",
  "user_id": "user-123"
}
```

### `GET /api/conversation`
Get conversation history for current user

### `DELETE /api/conversation`
Clear conversation history for current user

### `POST /api/training`
Train or retrain the chatbot model

**Body:**
```json
{
  "intents_data": {
    "intents": [
      {
        "tag": "greeting",
        "patterns": ["hi", "hello", "hey"],
        "responses": ["Hello!", "Hi there!"]
      }
    ]
  },
  "retrain": true
}
```

### `POST /api/predict`
Predict intent for a message without storing conversation

**Body:**
```json
{
  "message": "What's the weather like?"
}
```

### `POST /api/evaluate`
Evaluate model performance

### `GET /api/docs`
API documentation

---

## 🧠 Intent Configuration

### Basic Intent Structure
```json
{
  "intents": [
    {
      "tag": "greeting",
      "patterns": [
        "Hi", "Hello", "Good morning", "Hey there",
        "What's up", "How are you"
      ],
      "responses": [
        "Hello! How can I help you today?",
        "Hi there! What can I do for you?",
        "Good to see you! How may I assist you?"
      ],
      "context_set": "greeting"
    }
  ]
}
```

### Supported Intent Types
- **Greetings**: Hello, hi, good morning
- **Goodbyes**: Bye, see you later, farewell
- **Questions**: What, how, why, when
- **Commands**: Show, tell me, help
- **Small Talk**: Weather, time, jokes
- **Business Logic**: Custom domain-specific intents

### Adding New Intents
1. Edit `src/neural_chatbot/data/intents.json`
2. Add new intent with patterns and responses
3. Retrain: `neural-chatbot train`
4. Test: `neural-chatbot chat`

---

## 🎯 Training & Evaluation

### Basic Training
```bash
# Train with default settings
neural-chatbot train

# Train with visualization
neural-chatbot train --plot

# Advanced training
neural-chatbot train --epochs 300 --batch-size 16 --learning-rate 0.0005 --plot --evaluate
```

### Model Evaluation
```bash
# Evaluate current model
neural-chatbot evaluate --save-report

# Get detailed statistics
neural-chatbot status --detailed
```

### Training Features
- **Data Augmentation**: Automatic pattern expansion
- **Early Stopping**: Prevents overfitting
- **Learning Rate Scheduling**: Adaptive learning rates
- **Validation Split**: Model performance monitoring
- **Visualization**: Training plots and confusion matrices
- **Comprehensive Metrics**: Accuracy, precision, recall, F1-score

---

## ⚙️ Configuration

### Environment Variables
```bash
# Application
NEURAL_CHATBOT_SECRET_KEY=your-secret-key
NEURAL_CHATBOT_DEBUG=false
NEURAL_CHATBOT_LOG_LEVEL=INFO

# API Configuration
NEURAL_CHATBOT_HOST=0.0.0.0
NEURAL_CHATBOT_PORT=5000

# Model Configuration
NEURAL_CHATBOT_LEARNING_RATE=0.001
NEURAL_CHATBOT_EPOCHS=200
NEURAL_CHATBOT_BATCH_SIZE=8
NEURAL_CHATBOT_CONFIDENCE_THRESHOLD=0.25
```

### YAML Configuration
```yaml
# config.yaml
model:
  learning_rate: 0.001
  epochs: 200
  batch_size: 8
  dropout_rate: 0.5
  confidence_threshold: 0.25
  hidden_layers: [128, 64, 32]

data:
  intents_file: "intents.json"
  max_response_length: 500

api:
  host: "0.0.0.0"
  port: 5000
  debug: false
  cors_enabled: true

logging:
  level: "INFO"
  file: "logs/chatbot.log"
```

---

## 🚀 Command Line Interface

### Chat Commands
```bash
# Interactive chat
neural-chatbot chat

# Chat with specific user ID
neural-chatbot chat --user-id my-user-123
```

### Training Commands
```bash
# Basic training
neural-chatbot train

# Custom training
neural-chatbot train --intents data/custom_intents.json --epochs 500 --plot

# Training with evaluation
neural-chatbot train --evaluate
```

### Server Commands
```bash
# Development server
neural-chatbot serve --debug

# Production server
neural-chatbot serve --host 0.0.0.0 --port 8080 --workers 4
```

### Utility Commands
```bash
# Check status
neural-chatbot status --detailed

# Export model
neural-chatbot export --type model --output model.json

# Export conversations
neural-chatbot export --type conversations --output chat_history.json
```

---

## 🔧 Performance & Optimization

### Model Optimization
- **Batch Normalization**: Stable training
- **Dropout Regularization**: Prevents overfitting
- **Early Stopping**: Optimal training duration
- **Learning Rate Scheduling**: Adaptive learning

### Production Optimizations
- **Gunicorn**: WSGI server with worker processes
- **Redis Caching**: Session and response caching
- **Nginx**: Reverse proxy and load balancing
- **Health Checks**: Automatic failover

### Memory Management
```bash
# Monitor memory usage
docker stats neural-chatbot

# Scale horizontally
docker-compose up -d --scale chatbot=5
```

---

## 📊 Monitoring & Analytics

### Built-in Metrics
- Response times and latency
- Intent prediction accuracy
- User engagement statistics
- Conversation flow analysis
- Model confidence distributions

### Health Monitoring
```bash
# Application health
curl http://localhost:5000/health

# Detailed statistics
curl http://localhost:5000/api/statistics
```

### Conversation Analytics
- Intent distribution analysis
- User engagement patterns
- Response confidence tracking
- Performance bottlenecks

---

## 🔒 Security Features

### Input Validation
- Message length limits
- HTML escaping and sanitization
- SQL injection prevention
- Rate limiting

### API Security
- CORS configuration
- Request size limits
- Session management
- Error message sanitization

### Deployment Security
- Non-root Docker containers
- Secret management
- HTTPS enforcement
- Security headers

---

## 📋 Troubleshooting

### Common Issues

1. **Model Not Training**
   - Check intents.json format
   - Verify sufficient training data
   - Check memory availability

2. **Low Accuracy**
   - Add more training patterns
   - Increase training epochs
   - Adjust confidence threshold

3. **API Errors**
   - Check if model is trained
   - Verify input format
   - Check logs for details

### Debug Commands
```bash
# Verbose logging
neural-chatbot --verbose train

# Check model status
neural-chatbot status

# View logs
tail -f logs/chatbot.log
```

---

## 📌 Roadmap

- [x] Neural network-based intent classification
- [x] Modern web interface
- [x] REST API with documentation
- [x] Docker deployment
- [x] Training visualization
- [x] Conversation management
- [ ] Multi-language support
- [ ] Voice interface
- [ ] Integration with external APIs
- [ ] Advanced NLU with transformers
- [ ] Sentiment analysis
- [ ] Custom entity recognition

---

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Install development dependencies (`pip install -e ".[dev]"`)
4. Run tests (`pytest`)
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

### Development Setup
```bash
# Clone and install
git clone https://github.com/PawanRamaMali/Chatbot.git
cd Chatbot

# Install in development mode
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install

# Run tests
pytest

# Format code
black src/ tests/
isort src/ tests/
```

---

## 📄 License

MIT License. See `LICENSE` for details.

---

## ✨ Credits

Built with ❤️ by [Pawan Rama Mali](https://github.com/PawanRamaMali)

### Acknowledgments
- **TensorFlow/Keras** - Deep learning framework
- **NLTK** - Natural language processing
- **Flask** - Web framework
- **scikit-learn** - Machine learning utilities
- **Docker** - Containerization platform

**Neural Chatbot** - Making intelligent conversational AI accessible and customizable for everyone.