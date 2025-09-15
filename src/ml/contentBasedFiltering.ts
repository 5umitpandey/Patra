import { User, RecommendationResult, MLFeatures } from '../types';

export class ContentBasedFilteringEngine {
  private users: User[];

  constructor(users: User[]) {
    this.users = users;
  }

  private extractFeatures(user: User): MLFeatures {
    return {
      age: user.age,
      interestSimilarity: 0, // Will be calculated per comparison
      locationDistance: 0, // Will be calculated per comparison
      attractivenessScore: user.attractivenessScore,
      activityScore: user.activityScore,
      eloScore: user.eloScore / 2000, // Normalize to 0-1
      photoQuality: this.calculatePhotoQuality(user),
      profileCompleteness: this.calculateProfileCompleteness(user)
    };
  }

  private calculatePhotoQuality(user: User): number {
    // Simulate photo quality analysis
    const photoCount = user.photos.length;
    const qualityScore = Math.random() * 0.3 + 0.7; // 0.7-1.0
    const countBonus = Math.min(photoCount / 5, 1) * 0.2;
    return Math.min(qualityScore + countBonus, 1);
  }

  private calculateProfileCompleteness(user: User): number {
    let score = 0;
    if (user.bio && user.bio.length > 20) score += 0.3;
    if (user.interests.length >= 3) score += 0.3;
    if (user.photos.length >= 2) score += 0.2;
    if (user.status) score += 0.1;
    if (user.location.city) score += 0.1;
    return score;
  }

  private calculateInterestSimilarity(userA: User, userB: User): number {
    const interestsA = new Set(userA.interests);
    const interestsB = new Set(userB.interests);
    const intersection = new Set([...interestsA].filter(x => interestsB.has(x)));
    const union = new Set([...interestsA, ...interestsB]);
    
    return union.size > 0 ? intersection.size / union.size : 0;
  }

  private calculateDistance(userA: User, userB: User): number {
    const R = 3959; // Earth's radius in miles
    const dLat = (userB.location.lat - userA.location.lat) * Math.PI / 180;
    const dLng = (userB.location.lng - userA.location.lng) * Math.PI / 180;
    
    const a = Math.sin(dLat / 2) * Math.sin(dLat / 2) +
              Math.cos(userA.location.lat * Math.PI / 180) * Math.cos(userB.location.lat * Math.PI / 180) *
              Math.sin(dLng / 2) * Math.sin(dLng / 2);
    
    const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
    return R * c;
  }

  private cosineSimilarity(featuresA: MLFeatures, featuresB: MLFeatures): number {
    const vectorA = Object.values(featuresA);
    const vectorB = Object.values(featuresB);
    
    let dotProduct = 0;
    let normA = 0;
    let normB = 0;

    for (let i = 0; i < vectorA.length; i++) {
      dotProduct += vectorA[i] * vectorB[i];
      normA += vectorA[i] * vectorA[i];
      normB += vectorB[i] * vectorB[i];
    }

    if (normA === 0 || normB === 0) return 0;
    return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
  }

  private euclideanDistance(featuresA: MLFeatures, featuresB: MLFeatures): number {
    const vectorA = Object.values(featuresA);
    const vectorB = Object.values(featuresB);
    
    let sum = 0;
    for (let i = 0; i < vectorA.length; i++) {
      sum += Math.pow(vectorA[i] - vectorB[i], 2);
    }
    
    return Math.sqrt(sum);
  }

  public getContentBasedRecommendations(currentUser: User, topK: number = 10): RecommendationResult[] {
    const currentFeatures = this.extractFeatures(currentUser);
    const recommendations: RecommendationResult[] = [];

    for (const candidate of this.users) {
      if (candidate.id === currentUser.id) continue;

      // Check basic preferences
      if (candidate.age < currentUser.preferences.ageRange[0] || 
          candidate.age > currentUser.preferences.ageRange[1]) continue;

      const distance = this.calculateDistance(currentUser, candidate);
      if (distance > currentUser.preferences.maxDistance) continue;

      // Calculate enhanced features
      const candidateFeatures = this.extractFeatures(candidate);
      candidateFeatures.interestSimilarity = this.calculateInterestSimilarity(currentUser, candidate);
      candidateFeatures.locationDistance = 1 - Math.min(distance / currentUser.preferences.maxDistance, 1);

      // Calculate similarity scores
      const cosineSim = this.cosineSimilarity(currentFeatures, candidateFeatures);
      const euclideanDist = this.euclideanDistance(currentFeatures, candidateFeatures);
      const euclideanSim = 1 / (1 + euclideanDist); // Convert distance to similarity

      // Weighted combination
      const finalScore = (cosineSim * 0.6) + (euclideanSim * 0.4);

      const reasons = [];
      if (candidateFeatures.interestSimilarity > 0.3) {
        reasons.push(`${Math.round(candidateFeatures.interestSimilarity * 100)}% interest match`);
      }
      if (distance < 5) {
        reasons.push('Very close by');
      }
      if (candidate.attractivenessScore > 0.8) {
        reasons.push('High attractiveness score');
      }
      if (candidate.activityScore > 0.8) {
        reasons.push('Very active user');
      }

      recommendations.push({
        userId: candidate.id,
        score: Math.max(0, Math.min(1, finalScore)),
        reasons: reasons.length > 0 ? reasons : ['Good overall match'],
        algorithm: 'content-based',
        confidence: Math.min((candidateFeatures.profileCompleteness + candidateFeatures.photoQuality) / 2, 1)
      });
    }

    return recommendations
      .sort((a, b) => b.score - a.score)
      .slice(0, topK);
  }

  public getPreferenceLearningRecommendations(
    currentUser: User, 
    likedUsers: User[], 
    passedUsers: User[], 
    topK: number = 10
  ): RecommendationResult[] {
    if (likedUsers.length === 0) {
      return this.getContentBasedRecommendations(currentUser, topK);
    }

    // Learn preferences from liked users
    const likedFeatures = likedUsers.map(user => {
      const features = this.extractFeatures(user);
      features.interestSimilarity = this.calculateInterestSimilarity(currentUser, user);
      features.locationDistance = 1 - Math.min(
        this.calculateDistance(currentUser, user) / currentUser.preferences.maxDistance, 
        1
      );
      return features;
    });

    // Calculate average preferred features
    const avgPreferences: MLFeatures = {
      age: 0,
      interestSimilarity: 0,
      locationDistance: 0,
      attractivenessScore: 0,
      activityScore: 0,
      eloScore: 0,
      photoQuality: 0,
      profileCompleteness: 0
    };

    likedFeatures.forEach(features => {
      Object.keys(avgPreferences).forEach(key => {
        avgPreferences[key as keyof MLFeatures] += features[key as keyof MLFeatures];
      });
    });

    Object.keys(avgPreferences).forEach(key => {
      avgPreferences[key as keyof MLFeatures] /= likedFeatures.length;
    });

    // Score candidates based on learned preferences
    const recommendations: RecommendationResult[] = [];

    for (const candidate of this.users) {
      if (candidate.id === currentUser.id) continue;
      if (likedUsers.some(u => u.id === candidate.id)) continue;
      if (passedUsers.some(u => u.id === candidate.id)) continue;

      const candidateFeatures = this.extractFeatures(candidate);
      candidateFeatures.interestSimilarity = this.calculateInterestSimilarity(currentUser, candidate);
      candidateFeatures.locationDistance = 1 - Math.min(
        this.calculateDistance(currentUser, candidate) / currentUser.preferences.maxDistance, 
        1
      );

      const similarity = this.cosineSimilarity(avgPreferences, candidateFeatures);
      
      recommendations.push({
        userId: candidate.id,
        score: Math.max(0, Math.min(1, similarity)),
        reasons: ['Matches your learned preferences'],
        algorithm: 'preference-learning',
        confidence: Math.min(likedUsers.length / 10, 1)
      });
    }

    return recommendations
      .sort((a, b) => b.score - a.score)
      .slice(0, topK);
  }
}