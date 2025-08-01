# 🚀 AI Assistant Quick Start Guide

## ⚡ Get Your AI Assistant Running in 5 Minutes

### Step 1: Create Environment Files

**Create `backend/.env`:**
```bash
# AI Assistant Backend Configuration
OPENROUTER_API_KEY=your_openrouter_api_key_here
DATABASE_URL=sqlite:///./ai_assistant.db
ALLOWED_ORIGINS=http://localhost:3000,http://127.0.0.1:3000
MAX_FILE_SIZE=10485760
DB_ECHO=false
DB_POOL_SIZE=20
DB_MAX_OVERFLOW=30
SECRET_KEY=your-super-secret-key-change-this-in-production
ENVIRONMENT=development
DEBUG=false
LOG_LEVEL=INFO
```

**Create `frontend/.env`:**
```bash
# AI Assistant Frontend Configuration
REACT_APP_API_URL=http://localhost:8000
REACT_APP_WS_URL=ws://localhost:8000
GENERATE_SOURCEMAP=false
```

### Step 2: Get Free API Key

1. Visit: https://openrouter.ai/keys
2. Sign up (no credit card required)
3. Copy your API key
4. Replace `your_openrouter_api_key_here` in `backend/.env`

### Step 3: Install Dependencies

**Backend:**
```bash
cd backend
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate
pip install -r requirements.txt
```

**Frontend:**
```bash
cd frontend
npm install
```

### Step 4: Start Services

**Terminal 1 - Backend:**
```bash
cd backend
python main.py
```
✅ Backend will run on http://localhost:8000

**Terminal 2 - Frontend:**
```bash
cd frontend
npm start
```
✅ Frontend will run on http://localhost:3000

### Step 5: Test Your AI Assistant

1. Open http://localhost:3000
2. You'll see a VS Code-like interface
3. Try asking: "Write a Python function to calculate fibonacci numbers"
4. Select different AI models from the dropdown
5. Upload code files for analysis

## 🎯 What You Can Do

### 💬 **Chat with AI**
- Ask coding questions
- Get code explanations
- Debug issues
- Get best practices

### 🔧 **Code Actions**
- **Explain Code**: Select code → Ctrl+Shift+E
- **Debug Code**: Select code → Ctrl+Shift+D  
- **Refactor Code**: Select code → Ctrl+Shift+R
- **Generate Code**: Ctrl+Shift+G

### 📁 **File Management**
- Upload code files
- Browse project structure
- Get context-aware help

### 🎨 **Auto Language Detection**
- Supports 15+ languages
- No manual selection needed
- Smart context handling

## 🆘 Troubleshooting

### "API Key not configured"
- Check `backend/.env` file exists
- Verify OpenRouter API key is set
- Restart backend server

### "Backend not responding"
- Check if backend is running on port 8000
- Verify Python dependencies installed
- Check console for error messages

### "Frontend not loading"
- Check if frontend is running on port 3000
- Verify Node.js dependencies installed
- Check browser console for errors

### "Models not working"
- Try different free models from dropdown
- Check internet connection
- Verify API key is valid

## 🎉 Success!

Once everything is running, you'll have:
- ✅ **Free AI models** (no API costs)
- ✅ **VS Code-like interface**
- ✅ **Auto language detection**
- ✅ **Context-aware conversations**
- ✅ **File management**
- ✅ **Code execution**

**Happy coding with your AI assistant! 🚀** 