import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Heart, User, MessageCircle, Sparkles } from 'lucide-react';
import { SwipeInterface } from './components/SwipeInterface';
import { ProfileView } from './components/ProfileView';
import { MatchModal } from './components/MatchModal';
import { mockUsers, currentUser } from './data/mockUsers';
import { User as UserType, SwipeAction, Match } from './types';

type TabType = 'discover' | 'profile' | 'matches';

function App() {
  const [activeTab, setActiveTab] = useState<TabType>('discover');
  const [users, setUsers] = useState<UserType[]>(mockUsers);
  const [swipeHistory, setSwipeHistory] = useState<SwipeAction[]>([]);
  const [matches, setMatches] = useState<Match[]>([]);
  const [showMatchModal, setShowMatchModal] = useState(false);
  const [currentMatch, setCurrentMatch] = useState<UserType | null>(null);
  const [user, setUser] = useState<UserType>(currentUser);

  useEffect(() => {
    // Simulate some initial swipe history for better recommendations
    const initialHistory: SwipeAction[] = [
      { userId: user.id, targetUserId: 'user-1', action: 'like', timestamp: Date.now() - 86400000 },
      { userId: user.id, targetUserId: 'user-3', action: 'pass', timestamp: Date.now() - 82800000 },
      { userId: user.id, targetUserId: 'user-5', action: 'like', timestamp: Date.now() - 79200000 },
      { userId: user.id, targetUserId: 'user-7', action: 'like', timestamp: Date.now() - 75600000 },
      { userId: user.id, targetUserId: 'user-9', action: 'pass', timestamp: Date.now() - 72000000 },
    ];
    setSwipeHistory(initialHistory);
  }, [user.id]);

  const handleSwipe = (action: SwipeAction) => {
    setSwipeHistory(prev => [...prev, action]);
  };

  const handleMatch = (matchedUser: UserType) => {
    const newMatch: Match = {
      id: `match-${Date.now()}`,
      users: [user.id, matchedUser.id],
      timestamp: Date.now(),
      compatibility: Math.floor(Math.random() * 20 + 80), // 80-100% compatibility
      algorithm: 'hybrid'
    };
    
    setMatches(prev => [...prev, newMatch]);
    setCurrentMatch(matchedUser);
    setShowMatchModal(true);
  };

  const handleUpdateStatus = (status: { text: string; expiresIn: number }) => {
    const newStatus = {
      text: status.text,
      timestamp: Date.now(),
      expiresAt: Date.now() + status.expiresIn
    };
    
    setUser(prev => ({ ...prev, status: newStatus }));
  };

  const handleSendMessage = () => {
    setShowMatchModal(false);
    setActiveTab('matches');
    // In a real app, this would navigate to the chat
  };

  const getTabIcon = (tab: TabType) => {
    switch (tab) {
      case 'discover':
        return <Heart className="w-6 h-6" />;
      case 'profile':
        return <User className="w-6 h-6" />;
      case 'matches':
        return <MessageCircle className="w-6 h-6" />;
    }
  };

  const getTabLabel = (tab: TabType) => {
    switch (tab) {
      case 'discover':
        return 'Discover';
      case 'profile':
        return 'Profile';
      case 'matches':
        return `Matches ${matches.length > 0 ? `(${matches.length})` : ''}`;
    }
  };

  return (
    <div className="h-screen bg-gradient-to-br from-pink-50 via-white to-orange-50 flex flex-col">
      {/* Header */}
      <header className="bg-white/80 backdrop-blur-sm border-b border-white/20 px-6 py-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-2">
            <div className="w-8 h-8 bg-gradient-to-r from-primary-500 to-accent-500 rounded-full flex items-center justify-center">
              <Sparkles className="w-5 h-5 text-white" />
            </div>
            <h1 className="text-2xl font-bold gradient-text">Patra</h1>
          </div>
          
          <div className="flex items-center space-x-4">
            <div className="text-right">
              <p className="text-sm font-medium text-gray-900">{user.name}</p>
              <p className="text-xs text-gray-500">ELO: {user.eloScore}</p>
            </div>
            <img
              src={user.photos[0]}
              alt={user.name}
              className="w-10 h-10 rounded-full object-cover border-2 border-primary-200"
              onError={(e) => {
                e.currentTarget.src = 'https://images.pexels.com/photos/1239291/pexels-photo-1239291.jpeg?auto=compress&cs=tinysrgb&w=400&h=600&fit=crop';
              }}
            />
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="flex-1 overflow-hidden">
        <AnimatePresence mode="wait">
          {activeTab === 'discover' && (
            <motion.div
              key="discover"
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: 20 }}
              className="h-full"
            >
              <SwipeInterface
                currentUser={user}
                users={users}
                onSwipe={handleSwipe}
                onMatch={handleMatch}
              />
            </motion.div>
          )}

          {activeTab === 'profile' && (
            <motion.div
              key="profile"
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: 20 }}
              className="h-full"
            >
              <ProfileView
                user={user}
                onUpdateStatus={handleUpdateStatus}
              />
            </motion.div>
          )}

          {activeTab === 'matches' && (
            <motion.div
              key="matches"
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: 20 }}
              className="h-full flex items-center justify-center"
            >
              <div className="text-center">
                <MessageCircle className="w-16 h-16 text-gray-300 mx-auto mb-4" />
                <h3 className="text-xl font-semibold text-gray-700 mb-2">
                  {matches.length > 0 ? `${matches.length} Matches` : 'No matches yet'}
                </h3>
                <p className="text-gray-500">
                  {matches.length > 0 
                    ? 'Start conversations with your matches!' 
                    : 'Keep swiping to find your perfect match'
                  }
                </p>
                {matches.length > 0 && (
                  <div className="mt-6 space-y-3">
                    {matches.slice(0, 5).map((match) => {
                      const matchedUser = users.find(u => 
                        match.users.includes(u.id) && u.id !== user.id
                      );
                      if (!matchedUser) return null;
                      
                      return (
                        <div key={match.id} className="bg-white rounded-lg p-4 shadow-sm flex items-center space-x-3">
                          <img
                            src={matchedUser.photos[0]}
                            alt={matchedUser.name}
                            className="w-12 h-12 rounded-full object-cover"
                            onError={(e) => {
                              e.currentTarget.src = 'https://images.pexels.com/photos/1239291/pexels-photo-1239291.jpeg?auto=compress&cs=tinysrgb&w=400&h=600&fit=crop';
                            }}
                          />
                          <div className="flex-1 text-left">
                            <p className="font-medium text-gray-900">{matchedUser.name}</p>
                            <p className="text-sm text-gray-500">{match.compatibility}% match</p>
                          </div>
                          <button className="bg-primary-500 text-white px-4 py-2 rounded-full text-sm hover:bg-primary-600 transition-colors">
                            Chat
                          </button>
                        </div>
                      );
                    })}
                  </div>
                )}
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </main>

      {/* Bottom Navigation */}
      <nav className="bg-white/80 backdrop-blur-sm border-t border-white/20 px-6 py-3">
        <div className="flex justify-around">
          {(['discover', 'matches', 'profile'] as TabType[]).map((tab) => (
            <button
              key={tab}
              onClick={() => setActiveTab(tab)}
              className={`flex flex-col items-center space-y-1 py-2 px-4 rounded-xl transition-all duration-200 ${
                activeTab === tab
                  ? 'bg-gradient-to-r from-primary-500 to-accent-500 text-white shadow-lg'
                  : 'text-gray-600 hover:text-gray-900 hover:bg-gray-100'
              }`}
            >
              {getTabIcon(tab)}
              <span className="text-xs font-medium">{getTabLabel(tab)}</span>
            </button>
          ))}
        </div>
      </nav>

      {/* Match Modal */}
      <MatchModal
        isOpen={showMatchModal}
        matchedUser={currentMatch}
        currentUser={user}
        onClose={() => setShowMatchModal(false)}
        onSendMessage={handleSendMessage}
      />
    </div>
  );
}

export default App;