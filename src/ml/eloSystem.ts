import { User, SwipeAction } from '../types';

export class EloRankingSystem {
  private static readonly K_FACTOR = 32;
  private static readonly INITIAL_RATING = 1500;

  public static calculateExpectedScore(ratingA: number, ratingB: number): number {
    return 1 / (1 + Math.pow(10, (ratingB - ratingA) / 400));
  }

  public static updateRatings(
    userA: User, 
    userB: User, 
    result: 'win' | 'loss' | 'draw'
  ): { newRatingA: number; newRatingB: number } {
    const expectedA = this.calculateExpectedScore(userA.eloScore, userB.eloScore);
    const expectedB = this.calculateExpectedScore(userB.eloScore, userA.eloScore);

    let actualA: number;
    let actualB: number;

    switch (result) {
      case 'win':
        actualA = 1;
        actualB = 0;
        break;
      case 'loss':
        actualA = 0;
        actualB = 1;
        break;
      case 'draw':
        actualA = 0.5;
        actualB = 0.5;
        break;
    }

    const newRatingA = userA.eloScore + this.K_FACTOR * (actualA - expectedA);
    const newRatingB = userB.eloScore + this.K_FACTOR * (actualB - expectedB);

    return {
      newRatingA: Math.max(800, Math.min(2400, newRatingA)),
      newRatingB: Math.max(800, Math.min(2400, newRatingB))
    };
  }

  public static processSwipeAction(
    swiper: User, 
    target: User, 
    action: 'like' | 'pass'
  ): { newSwiperRating: number; newTargetRating: number } {
    // In dating apps, a "like" is considered a win for the target
    // A "pass" is considered a loss for the target
    const result = action === 'like' ? 'loss' : 'win';
    return this.updateRatings(swiper, target, result);
  }

  public static getEloBasedRecommendations(
    currentUser: User, 
    candidates: User[], 
    tolerance: number = 200
  ): User[] {
    const userRating = currentUser.eloScore;
    
    return candidates
      .filter(candidate => {
        const ratingDiff = Math.abs(candidate.eloScore - userRating);
        return ratingDiff <= tolerance;
      })
      .sort((a, b) => {
        // Prefer users with similar ratings, but slightly favor higher ratings
        const diffA = Math.abs(a.eloScore - userRating);
        const diffB = Math.abs(b.eloScore - userRating);
        
        if (Math.abs(diffA - diffB) < 50) {
          // If ratings are similar, prefer higher rating
          return b.eloScore - a.eloScore;
        }
        
        return diffA - diffB;
      });
  }

  public static calculateCompatibilityScore(userA: User, userB: User): number {
    const ratingDiff = Math.abs(userA.eloScore - userB.eloScore);
    const maxDiff = 600; // Maximum meaningful difference
    
    // Convert rating difference to compatibility score (0-1)
    const baseCompatibility = Math.max(0, 1 - (ratingDiff / maxDiff));
    
    // Boost compatibility for users in similar rating ranges
    const avgRating = (userA.eloScore + userB.eloScore) / 2;
    let tierBonus = 0;
    
    if (avgRating > 1800) tierBonus = 0.1; // High tier
    else if (avgRating > 1600) tierBonus = 0.05; // Mid-high tier
    else if (avgRating > 1400) tierBonus = 0.02; // Mid tier
    
    return Math.min(1, baseCompatibility + tierBonus);
  }

  public static simulateSwipeOutcome(swiper: User, target: User): boolean {
    const expectedScore = this.calculateExpectedScore(target.eloScore, swiper.eloScore);
    
    // Add some randomness and other factors
    const attractivenessBonus = target.attractivenessScore * 0.3;
    const activityBonus = target.activityScore * 0.1;
    const photoQualityBonus = Math.random() * 0.1; // Simulated photo quality
    
    const finalProbability = Math.min(0.9, expectedScore + attractivenessBonus + activityBonus + photoQualityBonus);
    
    return Math.random() < finalProbability;
  }

  public static updateUserStats(user: User, swipeActions: SwipeAction[]): User {
    const userSwipes = swipeActions.filter(action => action.userId === user.id);
    const receivedSwipes = swipeActions.filter(action => action.targetUserId === user.id);
    
    const rightSwipes = userSwipes.filter(action => action.action === 'like').length;
    const leftSwipes = userSwipes.filter(action => action.action === 'pass').length;
    const receivedLikes = receivedSwipes.filter(action => action.action === 'like').length;
    
    // Calculate new attractiveness score based on received likes
    const totalReceived = receivedSwipes.length;
    if (totalReceived > 0) {
      const likeRatio = receivedLikes / totalReceived;
      user.attractivenessScore = Math.min(1, user.attractivenessScore * 0.8 + likeRatio * 0.2);
    }
    
    // Update activity score based on recent activity
    const recentActivity = userSwipes.filter(
      action => Date.now() - action.timestamp < 24 * 60 * 60 * 1000
    ).length;
    user.activityScore = Math.min(1, recentActivity / 20); // 20 swipes = max activity
    
    user.stats.rightSwipes = rightSwipes;
    user.stats.leftSwipes = leftSwipes;
    
    return user;
  }
}