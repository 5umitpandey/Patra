# Firebase ML Matching Algorithm Setup Guide

## Prerequisites
1. Python 3.8 or higher
2. Firebase project with Firestore enabled
3. Firebase service account key

## Setup Instructions

### 1. Install Dependencies
```bash
cd ml_backend/scripts
pip install -r requirements.txt
```

### 2. Firebase Service Account Setup
1. Go to Firebase Console (https://console.firebase.google.com)
2. Select your project
3. Go to Project Settings > Service Accounts
4. Click "Generate new private key"
5. Download the JSON file
6. Rename it to `firebase-service-account.json`
7. Place it in the `ml_backend/scripts/` directory

### 3. Test the Algorithm
```bash
# Test mode - analyze data and show sample matches
python reject_superlike_like.py

# Server mode - start Flask API
python reject_superlike_like.py server
```

## API Endpoints

### Health Check
```
GET http://localhost:5000/health
```

### Get Recommendations
```
GET http://localhost:5000/recommendations/<user_uid>?count=10
```

### Refresh Data
```
POST http://localhost:5000/refresh-data
```

## Integration with Flutter App

### Option 1: Direct Python Integration
Use the `get_recommendations_api(user_uid, count)` function directly in your Python backend.

### Option 2: HTTP API Integration
Call the Flask API endpoints from your Flutter app using HTTP requests.

Example Flutter code:
```dart
Future<List<Map<String, dynamic>>> getMLRecommendations(String userUid) async {
  try {
    final response = await http.get(
      Uri.parse('http://localhost:5000/recommendations/$userUid?count=10'),
    );
    
    if (response.statusCode == 200) {
      final data = json.decode(response.body);
      return List<Map<String, dynamic>>.from(data['recommendations']);
    }
  } catch (e) {
    print('Error getting ML recommendations: $e');
  }
  return [];
}
```

## Algorithm Features

- **Bio Similarity**: Uses TF-IDF vectorization and cosine similarity
- **Age Compatibility**: Weights users within 10 years age difference
- **Location Matching**: Prioritizes same city/state users
- **Interest Overlap**: Uses Jaccard similarity for shared interests
- **Swipe History**: Considers previous like/superlike actions
- **Gender Preferences**: Basic orientation-based filtering

## Customization

You can adjust the matching weights in the `match_score()` function:
```python
final_score = (0.25*age_score + 0.15*loc_score + 0.25*hobby_score + 
              0.10*lf_score + 0.15*bio_score + 0.10*gender_score)
```

## Troubleshooting

1. **Firebase Import Error**: Install firebase-admin: `pip install firebase-admin`
2. **No Users Found**: Check Firebase connection and user collection
3. **Permission Denied**: Verify service account key has Firestore read permissions
4. **Empty Recommendations**: Check if user exists and has valid data

## Security Notes

- Keep `firebase-service-account.json` secure and never commit to version control
- Add the API server behind authentication in production
- Consider rate limiting for the API endpoints
- Use HTTPS in production deployment