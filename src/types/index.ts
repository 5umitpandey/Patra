export interface User {
  id: string;
  name: string;
  age: number;
  bio: string;
  photos: string[];
  interests: string[];
  location: {
    lat: number;
    lng: number;
    city: string;
  };
  status?: {
    text: string;
    timestamp: number;
    expiresAt: number;
  };
  preferences: {
    ageRange: [number, number];
    maxDistance: number;
    interests: string[];
  };
  stats: {
    rightSwipes: number;
    leftSwipes: number;
    matches: number;
    profileViews: number;
  };
  eloScore: number;
  attractivenessScore: number;
  activityScore: number;
  lastActive: number;
}

export interface SwipeAction {
  userId: string;
  targetUserId: string;
  action: 'like' | 'pass';
  timestamp: number;
  confidence?: number;
}

export interface Match {
  id: string;
  users: [string, string];
  timestamp: number;
  compatibility: number;
  algorithm: string;
}

export interface RecommendationResult {
  userId: string;
  score: number;
  reasons: string[];
  algorithm: string;
  confidence: number;
}

export interface MLFeatures {
  age: number;
  interestSimilarity: number;
  locationDistance: number;
  attractivenessScore: number;
  activityScore: number;
  eloScore: number;
  photoQuality: number;
  profileCompleteness: number;
}

export interface AlgorithmWeights {
  collaborative: number;
  contentBased: number;
  elo: number;
  location: number;
  attractiveness: number;
  activity: number;
}