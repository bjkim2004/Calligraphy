# 한글 캘리그래피 생성기 🖌️

고품질 한글 붓글씨를 생성하는 웹 서비스입니다.

## 특징

- ✅ **100% 가독성 보장** - 한글 폰트를 직접 사용
- 🎨 **다양한 스타일** - 붓글씨/펜글씨 선택
- 🖼️ **붓질 효과** - OpenCV를 이용한 자연스러운 효과
- ⚡ **빠른 생성** - AI 없이도 고품질 결과
- 🤖 **AI 질감 개선** - GPU/CPU 모두 지원

## 설치 및 실행

### 1. 백엔드 설정

```bash
cd backend
python -m venv venv
venv\Scripts\activate  # Windows
pip install -r requirements.txt
python download_fonts.py
python main.py
```

### 2. 프론트엔드 설정

```bash
cd frontend
npm install
npm run dev
```

### 3. 빠른 실행 (Windows)

```bash
start-all.bat  # 전체 서버 시작
stop-all.bat   # 전체 서버 종료
```

## 접속

- 백엔드 API: http://localhost:8000
- 프론트엔드: http://localhost:5173

## 기술 스택

- **Backend**: FastAPI, OpenCV, Pillow, PyTorch (선택)
- **Frontend**: React, Vite
- **Fonts**: Google Fonts 한글 손글씨

## 라이선스

MIT License
