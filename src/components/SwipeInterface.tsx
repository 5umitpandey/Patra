import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Heart, X, RotateCcw, Zap, Settings, TrendingUp } from 'lucide-react';
import { SwipeCard } from './SwipeCard';
import { User, SwipeAction, RecommendationResult } from '../types';
import { HybridRecommendationEngine } from '../ml/hybridRecommendation';
import { EloRankingSystem } from '../ml/eloSystem';

interface SwipeInterfaceProps {
  currentUser: User;
  users: User[];
  onSwipe: (action: SwipeAction) => void;
  onMatch: (matchedUser: User) => void;
}

export const SwipeInterface: React.FC<SwipeInterfaceProps> = ({
  currentUser,
  users,
  onSwipe,
  onMatch
}) => {
  const [cardStack, setCardStack] = useState<User[]>([]);
  const [swipeHistory, setSwipeHistory] = useState<SwipeAction[]>([]);
  const [recommendations, setRecommendations] = useState<RecommendationResult[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [showStats, setShowStats] = useState(false);
  const [hybridEngine, setHybridEngine] = useState<HybridRecommendationEngine | null>(null);

  useEffect(() => {
    initializeRecommendations();
  }, [users, currentUser]);

  const initializeRecommendations = async () => {
    setIsLoading(true);
    
    // Initialize the hybrid recommendation engine
    const engine = new HybridRecommendationEngine(users, swipeHistory);
    setHybridEngine(engine);
    
    // Get initial recommendations
    const recs = engine.getHybridRecommendations(currentUser.id, 20);
    setRecommendations(recs);
    
    // Convert recommendations to user objects for the card stack
    const recommendedUsers = recs
      .map(rec => users.find(u => u.id === rec.userId))
      .filter(user => user !== undefined) as User[];
    
    setCardStack(recommendedUsers.slice(0, 10));
    setIsLoading(false);
  };

  const handleSwipe = (direction: 'left' | 'right') => {
    if (cardStack.length === 0) return;

    const swipedUser = cardStack[0];
    const action: SwipeAction = {
      userId: currentUser.id,
      targetUserId: swipedUser.id,
      action: direction === 'right' ? 'like' : 'pass',
      timestamp: Date.now()
    };

    // Update swipe history
    const newSwipeHistory = [...swipeHistory, action];
    setSwipeHistory(newSwipeHistory);
    
    // Update ELO scores
    const { newSwiperRating, newTargetRating } = EloRankingSystem.processSwipeAction(
      currentUser,
      swipedUser,
      action.action
    );
    
    currentUser.eloScore = newSwiperRating;
    swipedUser.eloScore = newTargetRating;

    // Check for match (simulate mutual like)
    if (direction === 'right' && Math.random() > 0.7) {
      onMatch(swipedUser);
    }

    // Remove swiped card and add prediction confidence
    if (hybridEngine) {
      const prediction = hybridEngine.predictSwipeOutcome(currentUser.id, swipedUser.id);
      action.confidence = prediction;
    }

    onSwipe(action);

    // Update card stack
    const newStack = cardStack.slice(1);
    setCardStack(newStack);

    // Refresh recommendations if running low on cards
    if (newStack.length <= 3) {
      refreshRecommendations(newSwipeHistory);
    }
  };

  const refreshRecommendations = (updatedHistory: SwipeAction[]) => {
    if (!hybridEngine) return;

    // Update the engine with new swipe history
    const newEngine = new HybridRecommendationEngine(users, updatedHistory);
    setHybridEngine(newEngine);

    // Get fresh recommendations
    const newRecs = newEngine.getHybridRecommendations(currentUser.id, 20);
    setRecommendations(newRecs);

    // Add new users to the stack
    const newUsers = newRecs
      .map(rec => users.find(u => u.id === rec.userId))
      .filter(user => user !== undefined && !cardStack.some(c => c.id === user.id)) as User[];

    setCardStack(prev => [...prev, ...newUsers.slice(0, 10 - prev.length)]);
  };

  const handleButtonSwipe = (direction: 'left' | 'right') => {
    handleSwipe(direction);
  };

  const undoLastSwipe = () => {
    if (swipeHistory.length === 0) return;

    const lastSwipe = swipeHistory[swipeHistory.length - 1];
    const undoneUser = users.find(u => u.id === lastSwipe.targetUserId);
    
    if (undoneUser) {
      setCardStack(prev => [undoneUser, ...prev]);
      setSwipeHistory(prev => prev.slice(0, -1));
    }
  };

  const getAlgorithmStats = () => {
    if (!hybridEngine) return {};
    return hybridEngine.getAlgorithmPerformance();
  };

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-full">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary-500 mx-auto mb-4"></div>
          <p className="text-gray-600">Finding your perfect matches...</p>
        </div>
      </div>
    );
  }

  if (cardStack.length === 0) {
    return (
      <div className="flex items-center justify-center h-full">
        <div className="text-center">
          <Heart className="w-16 h-16 text-gray-300 mx-auto mb-4" />
          <h3 className="text-xl font-semibold text-gray-700 mb-2">No more profiles!</h3>
          <p className="text-gray-500 mb-4">Check back later for new matches</p>
          <button
            onClick={() => initializeRecommendations()}
            className="bg-primary-500 text-white px-6 py-2 rounded-full hover:bg-primary-600 transition-colors"
          >
            Refresh
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="relative h-full flex flex-col">
      {/* Stats Toggle */}
      <div className="absolute top-4 right-4 z-30">
        <button
          onClick={() => setShowStats(!showStats)}
          className="bg-white/80 backdrop-blur-sm rounded-full p-2 shadow-lg hover:bg-white transition-colors"
        >
          <TrendingUp className="w-5 h-5 text-gray-700" />
        </button>
      </div>

      {/* Algorithm Stats Panel */}
      <AnimatePresence>
        {showStats && (
          <motion.div
            initial={{ opacity: 0, y: -20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            className="absolute top-16 right-4 z-30 bg-white rounded-lg shadow-xl p-4 w-64"
          >
            <h4 className="font-semibold mb-3">Algorithm Performance</h4>
            <div className="space-y-2 text-sm">
              {Object.entries(getAlgorithmStats()).map(([algorithm, performance]) => (
                <div key={algorithm} className="flex justify-between">
                  <span className="capitalize">{algorithm.replace('-', ' ')}</span>
                  <span className="font-medium">{(performance * 100).toFixed(1)}%</span>
                </div>
              ))}
            </div>
            <div className="mt-3 pt-3 border-t">
              <div className="flex justify-between text-sm">
                <span>Total Swipes</span>
                <span>{swipeHistory.length}</span>
              </div>
              <div className="flex justify-between text-sm">
                <span>Like Rate</span>
                <span>
                  {swipeHistory.length > 0 
                    ? ((swipeHistory.filter(s => s.action === 'like').length / swipeHistory.length) * 100).toFixed(1)
                    : 0}%
                </span>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Card Stack */}
      <div className="flex-1 relative card-stack px-4 py-8">
        <div className="relative w-full max-w-sm mx-auto h-full">
          <AnimatePresence>
            {cardStack.slice(0, 3).map((user, index) => (
              <SwipeCard
                key={user.id}
                user={user}
                onSwipe={handleSwipe}
                isTop={index === 0}
                style={{
                  zIndex: 3 - index,
                  scale: 1 - index * 0.05,
                  y: index * 10,
                }}
              />
            ))}
          </AnimatePresence>
        </div>
      </div>

      {/* Action Buttons */}
      <div className="flex justify-center items-center space-x-6 pb-8">
        <button
          onClick={undoLastSwipe}
          disabled={swipeHistory.length === 0}
          className="bg-yellow-500 hover:bg-yellow-600 disabled:bg-gray-300 disabled:cursor-not-allowed text-white rounded-full p-4 shadow-lg transition-all duration-200 hover:scale-110"
        >
          <RotateCcw className="w-6 h-6" />
        </button>

        <button
          onClick={() => handleButtonSwipe('left')}
          disabled={cardStack.length === 0}
          className="bg-red-500 hover:bg-red-600 disabled:bg-gray-300 disabled:cursor-not-allowed text-white rounded-full p-5 shadow-lg transition-all duration-200 hover:scale-110"
        >
          <X className="w-8 h-8" />
        </button>

        <button
          onClick={() => handleButtonSwipe('right')}
          disabled={cardStack.length === 0}
          className="bg-green-500 hover:bg-green-600 disabled:bg-gray-300 disabled:cursor-not-allowed text-white rounded-full p-5 shadow-lg transition-all duration-200 hover:scale-110 animate-pulse-glow"
        >
          <Heart className="w-8 h-8" />
        </button>

        <button
          onClick={() => {/* Super like functionality */}}
          disabled={cardStack.length === 0}
          className="bg-blue-500 hover:bg-blue-600 disabled:bg-gray-300 disabled:cursor-not-allowed text-white rounded-full p-4 shadow-lg transition-all duration-200 hover:scale-110"
        >
          <Zap className="w-6 h-6" />
        </button>
      </div>

      {/* Current Recommendation Info */}
      {cardStack.length > 0 && recommendations.length > 0 && (
        <div className="absolute bottom-32 left-4 right-4 z-20">
          <div className="bg-black/50 backdrop-blur-sm text-white rounded-lg p-3 mx-auto max-w-sm">
            <div className="text-xs">
              <div className="flex justify-between items-center">
                <span>Match Score: {(recommendations.find(r => r.userId === cardStack[0].id)?.score * 100 || 0).toFixed(0)}%</span>
                <span className="capitalize">{recommendations.find(r => r.userId === cardStack[0].id)?.algorithm || 'hybrid'}</span>
              </div>
              <div className="mt-1 text-xs opacity-75">
                {recommendations.find(r => r.userId === cardStack[0].id)?.reasons.slice(0, 2).join(', ')}
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};