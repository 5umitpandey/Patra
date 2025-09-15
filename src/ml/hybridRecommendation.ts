import { User, SwipeAction, RecommendationResult, AlgorithmWeights } from '../types';
import { CollaborativeFilteringEngine } from './collaborativeFiltering';
import { ContentBasedFilteringEngine } from './contentBasedFiltering';
import { EloRankingSystem } from './eloSystem';

export class HybridRecommendationEngine {
  private collaborativeEngine: CollaborativeFilteringEngine;
  private contentEngine: ContentBasedFilteringEngine;
  private users: User[];
  private swipeHistory: SwipeAction[];
  private weights: AlgorithmWeights;

  constructor(users: User[], swipeHistory: SwipeAction[]) {
    this.users = users;
    this.swipeHistory = swipeHistory;
    this.collaborativeEngine = new CollaborativeFilteringEngine(users, swipeHistory);
    this.contentEngine = new ContentBasedFilteringEngine(users);
    
    // Default weights - can be adjusted based on A/B testing
    this.weights = {
      collaborative: 0.25,
      contentBased: 0.30,
      elo: 0.20,
      location: 0.10,
      attractiveness: 0.10,
      activity: 0.05
    };
  }

  public setWeights(weights: Partial<AlgorithmWeights>): void {
    this.weights = { ...this.weights, ...weights };
    
    // Normalize weights to sum to 1
    const total = Object.values(this.weights).reduce((sum, weight) => sum + weight, 0);
    Object.keys(this.weights).forEach(key => {
      this.weights[key as keyof AlgorithmWeights] /= total;
    });
  }

  private getUserById(userId: string): User | undefined {
    return this.users.find(u => u.id === userId);
  }

  private getLikedUsers(currentUserId: string): User[] {
    const likedUserIds = this.swipeHistory
      .filter(action => action.userId === currentUserId && action.action === 'like')
      .map(action => action.targetUserId);
    
    return likedUserIds
      .map(id => this.getUserById(id))
      .filter(user => user !== undefined) as User[];
  }

  private getPassedUsers(currentUserId: string): User[] {
    const passedUserIds = this.swipeHistory
      .filter(action => action.userId === currentUserId && action.action === 'pass')
      .map(action => action.targetUserId);
    
    return passedUserIds
      .map(id => this.getUserById(id))
      .filter(user => user !== undefined) as User[];
  }

  private calculateLocationScore(userA: User, userB: User): number {
    const R = 3959; // Earth's radius in miles
    const dLat = (userB.location.lat - userA.location.lat) * Math.PI / 180;
    const dLng = (userB.location.lng - userA.location.lng) * Math.PI / 180;
    
    const a = Math.sin(dLat / 2) * Math.sin(dLat / 2) +
              Math.cos(userA.location.lat * Math.PI / 180) * Math.cos(userB.location.lat * Math.PI / 180) *
              Math.sin(dLng / 2) * Math.sin(dLng / 2);
    
    const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
    const distance = R * c;
    
    // Convert distance to score (closer = higher score)
    return Math.max(0, 1 - (distance / userA.preferences.maxDistance));
  }

  private combineRecommendations(
    recommendations: Map<string, RecommendationResult[]>
  ): RecommendationResult[] {
    const combinedScores = new Map<string, {
      totalScore: number;
      reasons: Set<string>;
      algorithms: Set<string>;
      confidenceSum: number;
      count: number;
    }>();

    // Combine scores from all algorithms
    recommendations.forEach((results, algorithm) => {
      const weight = this.getAlgorithmWeight(algorithm);
      
      results.forEach(result => {
        if (!combinedScores.has(result.userId)) {
          combinedScores.set(result.userId, {
            totalScore: 0,
            reasons: new Set(),
            algorithms: new Set(),
            confidenceSum: 0,
            count: 0
          });
        }

        const combined = combinedScores.get(result.userId)!;
        combined.totalScore += result.score * weight;
        result.reasons.forEach(reason => combined.reasons.add(reason));
        combined.algorithms.add(algorithm);
        combined.confidenceSum += result.confidence;
        combined.count++;
      });
    });

    // Convert to final recommendations
    const finalRecommendations: RecommendationResult[] = [];

    combinedScores.forEach((data, userId) => {
      finalRecommendations.push({
        userId,
        score: data.totalScore,
        reasons: Array.from(data.reasons),
        algorithm: 'hybrid',
        confidence: data.confidenceSum / data.count
      });
    });

    return finalRecommendations.sort((a, b) => b.score - a.score);
  }

  private getAlgorithmWeight(algorithm: string): number {
    switch (algorithm) {
      case 'user-based-cf':
      case 'item-based-cf':
      case 'matrix-factorization':
        return this.weights.collaborative;
      case 'content-based':
      case 'preference-learning':
        return this.weights.contentBased;
      case 'elo':
        return this.weights.elo;
      case 'location':
        return this.weights.location;
      case 'attractiveness':
        return this.weights.attractiveness;
      case 'activity':
        return this.weights.activity;
      default:
        return 0.1;
    }
  }

  public getHybridRecommendations(currentUserId: string, topK: number = 10): RecommendationResult[] {
    const currentUser = this.getUserById(currentUserId);
    if (!currentUser) return [];

    const likedUsers = this.getLikedUsers(currentUserId);
    const passedUsers = this.getPassedUsers(currentUserId);
    const recommendations = new Map<string, RecommendationResult[]>();

    // Get recommendations from all algorithms
    try {
      // Collaborative Filtering
      if (this.swipeHistory.length > 10) {
        recommendations.set('user-based-cf', 
          this.collaborativeEngine.getUserBasedRecommendations(currentUserId, topK * 2));
        recommendations.set('item-based-cf', 
          this.collaborativeEngine.getItemBasedRecommendations(currentUserId, topK * 2));
        recommendations.set('matrix-factorization', 
          this.collaborativeEngine.getMatrixFactorizationRecommendations(currentUserId, 10, topK * 2));
      }
    } catch (error) {
      console.warn('Collaborative filtering failed:', error);
    }

    // Content-Based Filtering
    recommendations.set('content-based', 
      this.contentEngine.getContentBasedRecommendations(currentUser, topK * 2));
    
    if (likedUsers.length > 0) {
      recommendations.set('preference-learning', 
        this.contentEngine.getPreferenceLearningRecommendations(currentUser, likedUsers, passedUsers, topK * 2));
    }

    // ELO-based recommendations
    const eloCandidates = EloRankingSystem.getEloBasedRecommendations(currentUser, this.users);
    const eloRecommendations: RecommendationResult[] = eloCandidates
      .filter(candidate => candidate.id !== currentUserId)
      .slice(0, topK * 2)
      .map(candidate => ({
        userId: candidate.id,
        score: EloRankingSystem.calculateCompatibilityScore(currentUser, candidate),
        reasons: ['Similar ELO rating'],
        algorithm: 'elo',
        confidence: 0.8
      }));
    recommendations.set('elo', eloRecommendations);

    // Location-based scoring
    const locationRecommendations: RecommendationResult[] = this.users
      .filter(user => user.id !== currentUserId)
      .map(user => ({
        userId: user.id,
        score: this.calculateLocationScore(currentUser, user),
        reasons: ['Close proximity'],
        algorithm: 'location',
        confidence: 0.9
      }))
      .filter(rec => rec.score > 0)
      .sort((a, b) => b.score - a.score)
      .slice(0, topK * 2);
    recommendations.set('location', locationRecommendations);

    // Attractiveness-based scoring
    const attractivenessRecommendations: RecommendationResult[] = this.users
      .filter(user => user.id !== currentUserId)
      .map(user => ({
        userId: user.id,
        score: user.attractivenessScore,
        reasons: ['High attractiveness score'],
        algorithm: 'attractiveness',
        confidence: 0.7
      }))
      .sort((a, b) => b.score - a.score)
      .slice(0, topK * 2);
    recommendations.set('attractiveness', attractivenessRecommendations);

    // Activity-based scoring
    const activityRecommendations: RecommendationResult[] = this.users
      .filter(user => user.id !== currentUserId)
      .map(user => ({
        userId: user.id,
        score: user.activityScore,
        reasons: ['Very active user'],
        algorithm: 'activity',
        confidence: 0.6
      }))
      .sort((a, b) => b.score - a.score)
      .slice(0, topK * 2);
    recommendations.set('activity', activityRecommendations);

    // Combine all recommendations
    const combined = this.combineRecommendations(recommendations);
    
    // Filter out already swiped users
    const swipedUserIds = new Set([
      ...likedUsers.map(u => u.id),
      ...passedUsers.map(u => u.id)
    ]);

    return combined
      .filter(rec => !swipedUserIds.has(rec.userId))
      .slice(0, topK);
  }

  public predictSwipeOutcome(currentUserId: string, targetUserId: string): number {
    const recommendations = this.getHybridRecommendations(currentUserId, 100);
    const targetRec = recommendations.find(rec => rec.userId === targetUserId);
    
    if (!targetRec) return 0.5; // Default probability
    
    // Convert recommendation score to swipe probability
    return Math.min(0.95, Math.max(0.05, targetRec.score));
  }

  public getAlgorithmPerformance(): { [algorithm: string]: number } {
    // Simulate algorithm performance metrics
    return {
      'user-based-cf': 0.72,
      'item-based-cf': 0.68,
      'matrix-factorization': 0.75,
      'content-based': 0.70,
      'preference-learning': 0.78,
      'elo': 0.65,
      'location': 0.60,
      'attractiveness': 0.55,
      'activity': 0.50,
      'hybrid': 0.82
    };
  }

  public runABTest(testWeights: AlgorithmWeights, sampleSize: number = 100): {
    controlAccuracy: number;
    testAccuracy: number;
    improvement: number;
  } {
    // Simulate A/B test results
    const originalWeights = { ...this.weights };
    
    // Control group performance
    const controlAccuracy = 0.75 + Math.random() * 0.1;
    
    // Test group performance with new weights
    this.setWeights(testWeights);
    const testAccuracy = 0.70 + Math.random() * 0.15;
    
    // Restore original weights
    this.weights = originalWeights;
    
    return {
      controlAccuracy,
      testAccuracy,
      improvement: ((testAccuracy - controlAccuracy) / controlAccuracy) * 100
    };
  }
}