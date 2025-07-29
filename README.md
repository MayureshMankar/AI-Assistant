# AI Coding Assistant

A next-generation, full-stack AI coding assistant with chat, code generation, code execution, and more.

## Tech Stack
- **Backend:** FastAPI (Python), OpenAI API, Docker (for code execution)
- **Frontend:** Next.js (React), Monaco Editor
- **Database:** PostgreSQL (future)
- **Other:** Docker, WebSockets, Redis (future)

## Project Structure
```
ai-assistant/
├── backend/         # FastAPI app
├── frontend/        # Next.js React app
├── db/              # Database (future)
├── vscode-extension/ # (future)
├── scripts/         # DevOps, deployment
└── README.md
```

## Getting Started

### Backend
```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload
```

### Frontend
```bash
cd frontend
npm install
npm run dev
```

---

## Features (Planned)
- Chat with AI (GPT-4/4o)
- Code generation, refactoring, and explanation
- Secure code execution (Docker)
- Real-time code analysis
- Web search, GitHub integration
- Memory and context retention
- Diagrams (UML, flowcharts)
- VSCode extension 