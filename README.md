# 💕 Patra App 1.0

<div align="center">

![Flutter](https://img.shields.io/badge/Flutter-02569B?style=for-the-badge&logo=flutter&logoColor=white)
![Firebase](https://img.shields.io/badge/Firebase-039BE5?style=for-the-badge&logo=Firebase&logoColor=white)
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Dart](https://img.shields.io/badge/dart-%230175C2.svg?style=for-the-badge&logo=dart&logoColor=white)

**A Modern Dating App with AI-Powered Matching**

[🌐 Live Demo](https://patra1-0.vercel.app) 

</div>

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Architecture](#architecture)
- [Installation](#installation)
- [Configuration](#configuration)
- [ML Service](#ml-service)
- [Deployment](#deployment)
- [Contributing](#contributing)
- [Team](#team)
- [License](#license)

## 🎯 Overview

Patra is a modern dating application built with Flutter and powered by AI-driven matching algorithms. The app combines intuitive user experience with intelligent recommendation systems to help users find meaningful connections.

### Key Highlights
- 🤖 **AI-Powered Matching**: Smart recommendations using machine learning
- 🔥 **Real-time Chat**: Instant messaging with Firebase integration
- 📱 **Cross-Platform**: Works on iOS, Android, and Web
- ☁️ **Cloud Integration**: Seamless image uploads with Cloudinary
- 🔒 **Secure Authentication**: Firebase Auth with multiple providers

## ✨ Features

### Core Features
- ✅ **User Authentication** (College Mail Id)
- ✅ **Profile Management** with photo uploads
- ✅ **Smart Matching Algorithm** using ML recommendations
- ✅ **Swipe Interface** (Like, Pass, Super Like)
- ✅ **Real-time Messaging** 
- ✅ **Discovery Page** with advanced filters
- ✅ **Location-based Matching**
- ✅ **Push Notifications**

### Advanced Features
- 🔄 **ML Data Refresh** for improved recommendations
- 📊 **Analytics Integration** with Firebase Analytics
- 🎨 **Modern UI/UX** with smooth animations
- 🌓 **Theme Support** (Light/Dark mode)
- 🔐 **Privacy Controls** and data encryption
- 📱 **Responsive Design** for all screen sizes

## 🛠️ Tech Stack

### Frontend
- **Framework**: Flutter 3.0+
- **Language**: Dart
- **State Management**: Provider / Riverpod
- **UI Components**: Material Design 3
- **Navigation**: Go Router

### Backend & Services
- **Authentication**: Firebase Auth
- **Database**: Cloud Firestore
- **Storage**: Firebase Storage + Cloudinary
- **Analytics**: Firebase Analytics
- **Notifications**: Firebase Cloud Messaging

### AI/ML Service
- **Language**: Python 3.8+
- **Framework**: Flask
- **ML Libraries**: scikit-learn, pandas, numpy
- **Database**: Firebase Admin SDK


### Deployment
- **Web**: Vercel
- **Mobile**: Firebase App Distribution
- **ML Service**: Flask + Docker

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Client Layer                         │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐      │
│  │     iOS     │  │   Android   │  │     Web     │      │
│  └─────────────┘  └─────────────┘  └─────────────┘      │
└─────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────┐
│                 Flutter Application                     │
│  ┌─────────────────────────────────────────────────┐    │
│  │              ConfigService                      │    │
│  │         (Environment Management)                │    │
│  └─────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────┐
│                   Service Layer                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐   │
│  │   Firebase   │  │  Cloudinary  │  │  ML Service  │   │
│  │   Services   │  │   (Images)   │  │ (Python API) │   │
│  └──────────────┘  └──────────────┘  └──────────────┘   │
└─────────────────────────────────────────────────────────┘
```

## 🚀 Installation

### Prerequisites
- Flutter SDK (3.0+)
- Dart SDK (2.17+)
- Firebase CLI
- Python 3.8+ (for ML service)
- Git

### 1. Clone Repository
```bash
git clone https://github.com/utkarshshukla03/Patra1.0.git
cd Patra1.0/patra_initial
```

### 2. Install Flutter Dependencies
```bash
flutter pub get
```

### 3. Environment Setup
Create environment files:
```bash
# Copy example files
cp .env.example .env
cp api_keys.env.example api_keys.env

# Edit with your actual keys
nano .env
nano api_keys.env
```

### 4. Firebase Setup
```bash
# Install Firebase CLI
npm install -g firebase-tools

# Login and configure
firebase login
flutterfire configure
```

### 5. Run the Application
```bash
# Development mode
flutter run

# Web development
flutter run -d chrome

# Release build
flutter build apk --release  # Android
flutter build ios --release  # iOS
flutter build web --release  # Web
```

## ⚙️ Configuration

### Environment Variables

#### Required Variables
```env
# Firebase Configuration
FIREBASE_PROJECT_ID=your-project-id
FIREBASE_WEB_API_KEY=your-api-key
FIREBASE_WEB_APP_ID=your-app-id
FIREBASE_AUTH_DOMAIN=your-domain.firebaseapp.com

# Cloudinary Configuration
CLOUDINARY_CLOUD_NAME=your-cloud-name
CLOUDINARY_API_KEY=your-api-key
CLOUDINARY_API_SECRET=your-api-secret
CLOUDINARY_UPLOAD_PRESET=your-preset

# Environment
ENVIRONMENT=development # development, staging, production
DEBUG_MODE=true
ML_MATCHING_ENABLED=true
```

#### Optional Variables
```env
# Feature Flags
PREMIUM_FEATURES_ENABLED=false
VIDEO_CALL_ENABLED=false
PUSH_NOTIFICATIONS_ENABLED=true

# Limits
MAX_PHOTOS_PER_USER=6
MAX_SWIPES_PER_DAY=100
MAX_MATCHES_PER_REQUEST=20

# External Services
GOOGLE_ANALYTICS_ID=your-ga-id
SENTRY_DSN=your-sentry-dsn
```

### Firebase Configuration
1. Create a new Firebase project
2. Enable Authentication (Email, Google, Facebook)
3. Create Firestore database
4. Configure storage rules
5. Add your app (iOS/Android/Web)

### Cloudinary Setup
1. Create Cloudinary account
2. Get cloud name and API credentials
3. Create upload preset for image processing

## 🤖 ML Service

The AI-powered recommendation system runs as a separate Python service.

### Features
- User behavior analysis
- Compatibility scoring
- Preference learning
- Real-time recommendations

### Setup
```bash
cd ml_backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt

# Setup Firebase credentials
export GOOGLE_APPLICATION_CREDENTIALS="path/to/firebase-key.json"

# Run service
python app.py
```

### API Endpoints
```
GET  /health                    # Service status
POST /recommendations/{user_id} # Get recommendations
POST /refresh-data              # Refresh ML model
```

### Docker Deployment
```bash
# Build image
docker build -t patra-ml-service .

# Run container
docker run -p 5000:5000 -e GOOGLE_APPLICATION_CREDENTIALS=/app/firebase-key.json patra-ml-service
```

## 🌐 Deployment

### Web Deployment (Vercel)
```bash
# Install Vercel CLI
npm install -g vercel

# Deploy
flutter build web --release
vercel --prod
```

### Mobile Deployment
```bash
# Android
flutter build appbundle --release

# iOS (requires Mac)
flutter build ipa --release

# Firebase App Distribution
firebase appdistribution:distribute build/app/outputs/bundle/release/app-release.aab \
  --app 1:123456789:android:abcdef \
  --groups "testers"
```

### Environment-Specific Builds
```bash
# Staging
flutter build web --dart-define=ENVIRONMENT=staging

# Production
flutter build web --dart-define=ENVIRONMENT=production \
  --dart-define=DEBUG_MODE=false
```

## 📱 App Structure

```
lib/
├── main.dart                    # App entry point
├── auth/                        # Authentication logic
├── pages/                       # UI screens
│   ├── homePage.dart           # Main app interface
│   ├── loginPage.dart          # User login
│   ├── signUpPage.dart         # User registration
│   └── discoveryPage.dart      # Match discovery
├── services/                    # Business logic
│   ├── config_service.dart     # Environment management
│   ├── auth_service.dart       # Authentication service
│   ├── firestore_service.dart  # Database operations
│   ├── ml_service.dart         # ML API integration
│   └── cloudinary_service.dart # Image uploads
├── models/                      # Data models
├── widgets/                     # Reusable components
└── utils/                       # Helper functions
```

## 🔧 Development

### Code Quality
- **Linting**: `flutter analyze`
- **Formatting**: `dart format .`
- **Testing**: `flutter test`

### Git Workflow
```bash
# Feature branch
git checkout -b feature/new-feature
git commit -m "feat: add new feature"
git push origin feature/new-feature

# Create pull request
```

### Debugging
```bash
# Debug mode
flutter run --debug

# Profile mode
flutter run --profile

# Release mode
flutter run --release
```

## 🤝 Contributing

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Commit** changes (`git commit -m 'Add amazing feature'`)
4. **Push** to branch (`git push origin feature/amazing-feature`)
5. **Open** a Pull Request

### Development Guidelines
- Follow Flutter/Dart style guide
- Write unit tests for new features
- Update documentation
- Ensure CI/CD passes

## 👥 Team

<div align="center">

### Development Team

<table>
<tr>
<td align="center">
<img src="https://github.com/utkarshshukla03.png" width="100px;" alt="Utkarsh Shukla"/><br />
<b>Utkarsh Shukla</b><br />
<i>Full Stack Engineer</i><br />
<a href="https://github.com/utkarshshukla03">💻</a>
</td>
<td align="center">
<img src="https://github.com/5umitpandey.png" width="100px;" alt="Sumit Pandey"/><br />
<b>Sumit Pandey</b><br />
<i>AI/ML Engineer</i><br />
<a href="#">🤖</a>
</td>
</tr>
</table>



</div>

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Flutter team for the amazing framework
- Firebase for backend services
- Cloudinary for image management
- Open source community for inspiration

<div align="center">

**Built with ❤️ using Flutter**

[⭐ Star this repo](https://github.com/utkarshshukla03/Patra1.0) • [🐛 Report Bug](https://github.com/utkarshshukla03/Patra1.0/issues) • [💡 Request Feature](https://github.com/utkarshshukla03/Patra1.0/issues)

</div>
