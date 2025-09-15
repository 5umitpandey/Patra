import React from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Heart, MessageCircle, X, Star } from 'lucide-react';
import { User } from '../types';

interface MatchModalProps {
  isOpen: boolean;
  matchedUser: User | null;
  currentUser: User;
  onClose: () => void;
  onSendMessage: () => void;
}

export const MatchModal: React.FC<MatchModalProps> = ({
  isOpen,
  matchedUser,
  currentUser,
  onClose,
  onSendMessage
}) => {
  if (!matchedUser) return null;

  return (
    <AnimatePresence>
      {isOpen && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          className="fixed inset-0 bg-gradient-to-br from-pink-500/20 via-purple-500/20 to-orange-500/20 backdrop-blur-sm z-50 flex items-center justify-center p-4"
          onClick={onClose}
        >
          <motion.div
            initial={{ scale: 0.5, opacity: 0, y: 50 }}
            animate={{ scale: 1, opacity: 1, y: 0 }}
            exit={{ scale: 0.5, opacity: 0, y: 50 }}
            className="bg-white rounded-3xl w-full max-w-md overflow-hidden shadow-2xl"
            onClick={(e) => e.stopPropagation()}
          >
            {/* Header */}
            <div className="relative bg-gradient-to-br from-primary-500 to-accent-500 p-6 text-white">
              <button
                onClick={onClose}
                className="absolute top-4 right-4 p-2 hover:bg-white/20 rounded-full transition-colors"
              >
                <X className="w-5 h-5" />
              </button>
              
              <div className="text-center">
                <motion.div
                  initial={{ scale: 0 }}
                  animate={{ scale: 1 }}
                  transition={{ delay: 0.2, type: "spring", stiffness: 200 }}
                  className="w-20 h-20 bg-white/20 rounded-full flex items-center justify-center mx-auto mb-4"
                >
                  <Heart className="w-10 h-10 text-white" />
                </motion.div>
                
                <motion.h2
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: 0.3 }}
                  className="text-2xl font-bold mb-2"
                >
                  It's a Match! 🎉
                </motion.h2>
                
                <motion.p
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: 0.4 }}
                  className="text-white/90"
                >
                  You and {matchedUser.name} liked each other
                </motion.p>
              </div>
            </div>

            {/* Profile Photos */}
            <div className="relative p-6">
              <div className="flex items-center justify-center space-x-4 mb-6">
                <motion.div
                  initial={{ x: -50, opacity: 0 }}
                  animate={{ x: 0, opacity: 1 }}
                  transition={{ delay: 0.5 }}
                  className="relative"
                >
                  <img
                    src={currentUser.photos[0]}
                    alt={currentUser.name}
                    className="w-20 h-20 rounded-full object-cover border-4 border-white shadow-lg"
                    onError={(e) => {
                      e.currentTarget.src = 'https://images.pexels.com/photos/1239291/pexels-photo-1239291.jpeg?auto=compress&cs=tinysrgb&w=400&h=600&fit=crop';
                    }}
                  />
                  <div className="absolute -bottom-1 -right-1 bg-primary-500 rounded-full p-1">
                    <Heart className="w-3 h-3 text-white" />
                  </div>
                </motion.div>

                <motion.div
                  initial={{ scale: 0 }}
                  animate={{ scale: 1 }}
                  transition={{ delay: 0.6, type: "spring", stiffness: 200 }}
                  className="bg-gradient-to-r from-primary-500 to-accent-500 rounded-full p-3"
                >
                  <Heart className="w-6 h-6 text-white" />
                </motion.div>

                <motion.div
                  initial={{ x: 50, opacity: 0 }}
                  animate={{ x: 0, opacity: 1 }}
                  transition={{ delay: 0.5 }}
                  className="relative"
                >
                  <img
                    src={matchedUser.photos[0]}
                    alt={matchedUser.name}
                    className="w-20 h-20 rounded-full object-cover border-4 border-white shadow-lg"
                    onError={(e) => {
                      e.currentTarget.src = 'https://images.pexels.com/photos/1239291/pexels-photo-1239291.jpeg?auto=compress&cs=tinysrgb&w=400&h=600&fit=crop';
                    }}
                  />
                  <div className="absolute -bottom-1 -left-1 bg-primary-500 rounded-full p-1">
                    <Heart className="w-3 h-3 text-white" />
                  </div>
                </motion.div>
              </div>

              {/* Match Info */}
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.7 }}
                className="text-center mb-6"
              >
                <h3 className="text-lg font-semibold text-gray-900 mb-2">
                  {matchedUser.name}, {matchedUser.age}
                </h3>
                
                {/* Compatibility Score */}
                <div className="flex items-center justify-center space-x-2 mb-3">
                  <Star className="w-4 h-4 text-yellow-500" />
                  <span className="text-sm text-gray-600">
                    {Math.floor(Math.random() * 20 + 80)}% compatibility
                  </span>
                </div>

                {/* Common Interests */}
                {matchedUser.interests.length > 0 && (
                  <div className="mb-4">
                    <p className="text-xs text-gray-500 mb-2">Common interests:</p>
                    <div className="flex flex-wrap justify-center gap-1">
                      {matchedUser.interests.slice(0, 3).map((interest, index) => (
                        <span
                          key={index}
                          className="bg-primary-100 text-primary-700 px-2 py-1 rounded-full text-xs"
                        >
                          {interest}
                        </span>
                      ))}
                    </div>
                  </div>
                )}

                {/* Status */}
                {matchedUser.status && matchedUser.status.expiresAt > Date.now() && (
                  <div className="bg-gray-50 rounded-lg p-3 mb-4">
                    <p className="text-sm text-gray-700">"{matchedUser.status.text}"</p>
                  </div>
                )}
              </motion.div>

              {/* Action Buttons */}
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.8 }}
                className="space-y-3"
              >
                <button
                  onClick={onSendMessage}
                  className="w-full bg-gradient-to-r from-primary-500 to-accent-500 text-white py-4 rounded-xl font-semibold flex items-center justify-center space-x-2 hover:shadow-lg transition-all duration-200"
                >
                  <MessageCircle className="w-5 h-5" />
                  <span>Send Message</span>
                </button>
                
                <button
                  onClick={onClose}
                  className="w-full bg-gray-100 text-gray-700 py-3 rounded-xl font-medium hover:bg-gray-200 transition-colors"
                >
                  Keep Swiping
                </button>
              </motion.div>
            </div>
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>
  );
};