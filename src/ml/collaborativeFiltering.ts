import { Matrix } from 'ml-matrix';
import { User, SwipeAction, RecommendationResult } from '../types';

export class CollaborativeFilteringEngine {
  private userItemMatrix: Matrix;
  private userSimilarityMatrix: Matrix;
  private itemSimilarityMatrix: Matrix;
  private users: User[];
  private swipeHistory: SwipeAction[];

  constructor(users: User[], swipeHistory: SwipeAction[]) {
    this.users = users;
    this.swipeHistory = swipeHistory;
    this.buildUserItemMatrix();
    this.calculateSimilarityMatrices();
  }

  private buildUserItemMatrix(): void {
    const userIds = this.users.map(u => u.id);
    const matrix = Matrix.zeros(userIds.length, userIds.length);

    // Build interaction matrix from swipe history
    this.swipeHistory.forEach(swipe => {
      const userIndex = userIds.indexOf(swipe.userId);
      const targetIndex = userIds.indexOf(swipe.targetUserId);
      
      if (userIndex !== -1 && targetIndex !== -1) {
        // 1 for like, -1 for pass, 0 for no interaction
        matrix.set(userIndex, targetIndex, swipe.action === 'like' ? 1 : -1);
      }
    });

    this.userItemMatrix = matrix;
  }

  private calculateSimilarityMatrices(): void {
    const userCount = this.users.length;
    this.userSimilarityMatrix = Matrix.zeros(userCount, userCount);
    this.itemSimilarityMatrix = Matrix.zeros(userCount, userCount);

    // Calculate user-based similarity (cosine similarity)
    for (let i = 0; i < userCount; i++) {
      for (let j = i + 1; j < userCount; j++) {
        const userI = this.userItemMatrix.getRow(i);
        const userJ = this.userItemMatrix.getRow(j);
        
        const similarity = this.cosineSimilarity(userI, userJ);
        this.userSimilarityMatrix.set(i, j, similarity);
        this.userSimilarityMatrix.set(j, i, similarity);
      }
    }

    // Calculate item-based similarity
    for (let i = 0; i < userCount; i++) {
      for (let j = i + 1; j < userCount; j++) {
        const itemI = this.userItemMatrix.getColumn(i);
        const itemJ = this.userItemMatrix.getColumn(j);
        
        const similarity = this.cosineSimilarity(itemI, itemJ);
        this.itemSimilarityMatrix.set(i, j, similarity);
        this.itemSimilarityMatrix.set(j, i, similarity);
      }
    }
  }

  private cosineSimilarity(vectorA: number[], vectorB: number[]): number {
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

  public getUserBasedRecommendations(userId: string, topK: number = 10): RecommendationResult[] {
    const userIndex = this.users.findIndex(u => u.id === userId);
    if (userIndex === -1) return [];

    const similarities = this.userSimilarityMatrix.getRow(userIndex);
    const userRatings = this.userItemMatrix.getRow(userIndex);
    
    const recommendations: RecommendationResult[] = [];

    for (let targetIndex = 0; targetIndex < this.users.length; targetIndex++) {
      if (targetIndex === userIndex || userRatings[targetIndex] !== 0) continue;

      let weightedSum = 0;
      let similaritySum = 0;

      for (let similarUserIndex = 0; similarUserIndex < this.users.length; similarUserIndex++) {
        if (similarUserIndex === userIndex) continue;

        const similarity = similarities[similarUserIndex];
        const rating = this.userItemMatrix.get(similarUserIndex, targetIndex);

        if (similarity > 0 && rating !== 0) {
          weightedSum += similarity * rating;
          similaritySum += Math.abs(similarity);
        }
      }

      if (similaritySum > 0) {
        const score = weightedSum / similaritySum;
        recommendations.push({
          userId: this.users[targetIndex].id,
          score: Math.max(0, (score + 1) / 2), // Normalize to 0-1
          reasons: ['Similar users liked this profile'],
          algorithm: 'user-based-cf',
          confidence: Math.min(similaritySum / 10, 1)
        });
      }
    }

    return recommendations
      .sort((a, b) => b.score - a.score)
      .slice(0, topK);
  }

  public getItemBasedRecommendations(userId: string, topK: number = 10): RecommendationResult[] {
    const userIndex = this.users.findIndex(u => u.id === userId);
    if (userIndex === -1) return [];

    const userRatings = this.userItemMatrix.getRow(userIndex);
    const recommendations: RecommendationResult[] = [];

    for (let targetIndex = 0; targetIndex < this.users.length; targetIndex++) {
      if (targetIndex === userIndex || userRatings[targetIndex] !== 0) continue;

      let weightedSum = 0;
      let similaritySum = 0;

      for (let ratedIndex = 0; ratedIndex < this.users.length; ratedIndex++) {
        if (userRatings[ratedIndex] === 0) continue;

        const similarity = this.itemSimilarityMatrix.get(targetIndex, ratedIndex);
        const rating = userRatings[ratedIndex];

        if (similarity > 0) {
          weightedSum += similarity * rating;
          similaritySum += Math.abs(similarity);
        }
      }

      if (similaritySum > 0) {
        const score = weightedSum / similaritySum;
        recommendations.push({
          userId: this.users[targetIndex].id,
          score: Math.max(0, (score + 1) / 2),
          reasons: ['Similar to profiles you liked'],
          algorithm: 'item-based-cf',
          confidence: Math.min(similaritySum / 5, 1)
        });
      }
    }

    return recommendations
      .sort((a, b) => b.score - a.score)
      .slice(0, topK);
  }

  public getMatrixFactorizationRecommendations(userId: string, factors: number = 10, topK: number = 10): RecommendationResult[] {
    // Simplified matrix factorization using SVD approximation
    const userIndex = this.users.findIndex(u => u.id === userId);
    if (userIndex === -1) return [];

    try {
      const svd = this.userItemMatrix.svd();
      const reducedU = svd.leftSingularVectors.subMatrix(0, this.users.length - 1, 0, Math.min(factors - 1, svd.diagonal.length - 1));
      const reducedS = Matrix.diag(svd.diagonal.slice(0, Math.min(factors, svd.diagonal.length)));
      const reducedV = svd.rightSingularVectors.subMatrix(0, this.users.length - 1, 0, Math.min(factors - 1, svd.diagonal.length - 1));

      const reconstructed = reducedU.mmul(reducedS).mmul(reducedV.transpose());
      const userPredictions = reconstructed.getRow(userIndex);
      const userRatings = this.userItemMatrix.getRow(userIndex);

      const recommendations: RecommendationResult[] = [];

      for (let i = 0; i < this.users.length; i++) {
        if (i === userIndex || userRatings[i] !== 0) continue;

        const score = Math.max(0, Math.min(1, (userPredictions[i] + 1) / 2));
        recommendations.push({
          userId: this.users[i].id,
          score,
          reasons: ['Matrix factorization prediction'],
          algorithm: 'matrix-factorization',
          confidence: 0.7
        });
      }

      return recommendations
        .sort((a, b) => b.score - a.score)
        .slice(0, topK);
    } catch (error) {
      console.warn('Matrix factorization failed, falling back to user-based CF');
      return this.getUserBasedRecommendations(userId, topK);
    }
  }
}