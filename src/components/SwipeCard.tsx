import React, { useState, useRef } from 'react';
import { motion, useMotionValue, useTransform, PanInfo } from 'framer-motion';
import { Heart, X, MapPin, Clock, Star } from 'lucide-react';
import { User } from '../types';
import { formatDistanceToNow } from 'date-fns';

interface SwipeCardProps {
  user: User;
  onSwipe: (direction: 'left' | 'right') => void;
  isTop: boolean;
  style?: React.CSSProperties;
}

export const SwipeCard: React.FC<SwipeCardProps> = ({ user, onSwipe, isTop, style }) => {
  const [currentPhotoIndex, setCurrentPhotoIndex] = useState(0);
  const [isDragging, setIsDragging] = useState(false);
  const cardRef = useRef<HTMLDivElement>(null);
  
  const x = useMotionValue(0);
  const rotate = useTransform(x, [-200, 200], [-30, 30]);
  const opacity = useTransform(x, [-200, -100, 0, 100, 200], [0, 1, 1, 1, 0]);

  const handleDragStart = () => {
    setIsDragging(true);
  };

  const handleDragEnd = (event: MouseEvent | TouchEvent | PointerEvent, info: PanInfo) => {
    setIsDragging(false);
    
    const threshold = 100;
    const velocity = info.velocity.x;
    const offset = info.offset.x;

    if (Math.abs(velocity) >= 500 || Math.abs(offset) >= threshold) {
      onSwipe(offset > 0 ? 'right' : 'left');
    }
  };

  const nextPhoto = (e: React.MouseEvent) => {
    e.stopPropagation();
    setCurrentPhotoIndex((prev) => (prev + 1) % user.photos.length);
  };

  const prevPhoto = (e: React.MouseEvent) => {
    e.stopPropagation();
    setCurrentPhotoIndex((prev) => (prev - 1 + user.photos.length) % user.photos.length);
  };

  const calculateAge = (birthDate: number) => {
    return new Date().getFullYear() - new Date(birthDate).getFullYear();
  };

  const isStatusActive = user.status && user.status.expiresAt > Date.now();

  return (
    <motion.div
      ref={cardRef}
      className={`absolute w-full h-full bg-white rounded-2xl shadow-2xl overflow-hidden cursor-grab ${
        isDragging ? 'cursor-grabbing' : ''
      } ${isTop ? 'z-20' : 'z-10'}`}
      style={{
        x,
        rotate,
        opacity,
        ...style,
      }}
      drag={isTop ? 'x' : false}
      dragConstraints={{ left: 0, right: 0 }}
      onDragStart={handleDragStart}
      onDragEnd={handleDragEnd}
      whileTap={{ scale: 0.95 }}
    >
      {/* Photo Section */}
      <div className="relative h-3/5 overflow-hidden">
        <img
          src={user.photos[currentPhotoIndex]}
          alt={user.name}
          className="w-full h-full object-cover"
          onError={(e) => {
            e.currentTarget.src = 'https://images.pexels.com/photos/1239291/pexels-photo-1239291.jpeg?auto=compress&cs=tinysrgb&w=400&h=600&fit=crop';
          }}
        />
        
        {/* Photo Navigation */}
        {user.photos.length > 1 && (
          <>
            <button
              onClick={prevPhoto}
              className="absolute left-0 top-0 w-1/2 h-full z-10 bg-transparent"
              aria-label="Previous photo"
            />
            <button
              onClick={nextPhoto}
              className="absolute right-0 top-0 w-1/2 h-full z-10 bg-transparent"
              aria-label="Next photo"
            />
            
            {/* Photo Indicators */}
            <div className="absolute top-4 left-4 right-4 flex space-x-1">
              {user.photos.map((_, index) => (
                <div
                  key={index}
                  className={`flex-1 h-1 rounded-full ${
                    index === currentPhotoIndex ? 'bg-white' : 'bg-white/30'
                  }`}
                />
              ))}
            </div>
          </>
        )}

        {/* Status Bubble */}
        {isStatusActive && (
          <div className="absolute top-4 left-4 status-bubble">
            <div className="flex items-center space-x-1">
              <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse" />
              <span className="text-xs">{user.status!.text}</span>
            </div>
          </div>
        )}

        {/* ELO Badge */}
        <div className="absolute top-4 right-4 bg-black/50 backdrop-blur-sm rounded-full px-2 py-1">
          <div className="flex items-center space-x-1">
            <Star className="w-3 h-3 text-yellow-400" />
            <span className="text-white text-xs font-medium">{user.eloScore}</span>
          </div>
        </div>

        {/* Swipe Indicators */}
        <motion.div
          className="absolute inset-0 flex items-center justify-center"
          style={{
            opacity: useTransform(x, [0, 100], [0, 1]),
          }}
        >
          <div className="bg-green-500 text-white px-6 py-3 rounded-full font-bold text-xl transform rotate-12 border-4 border-green-500">
            LIKE
          </div>
        </motion.div>
        
        <motion.div
          className="absolute inset-0 flex items-center justify-center"
          style={{
            opacity: useTransform(x, [-100, 0], [1, 0]),
          }}
        >
          <div className="bg-red-500 text-white px-6 py-3 rounded-full font-bold text-xl transform -rotate-12 border-4 border-red-500">
            NOPE
          </div>
        </motion.div>
      </div>

      {/* Info Section */}
      <div className="h-2/5 p-6 flex flex-col justify-between">
        <div>
          <div className="flex items-center justify-between mb-2">
            <h2 className="text-2xl font-bold text-gray-900">
              {user.name}, {user.age}
            </h2>
            <div className="flex items-center text-gray-500 text-sm">
              <MapPin className="w-4 h-4 mr-1" />
              {user.location.city}
            </div>
          </div>

          {user.bio && (
            <p className="text-gray-600 text-sm mb-3 line-clamp-2">{user.bio}</p>
          )}

          {/* Interests */}
          <div className="flex flex-wrap gap-2 mb-3">
            {user.interests.slice(0, 4).map((interest, index) => (
              <span
                key={index}
                className="bg-gradient-to-r from-primary-100 to-accent-100 text-primary-700 px-2 py-1 rounded-full text-xs font-medium"
              >
                {interest}
              </span>
            ))}
            {user.interests.length > 4 && (
              <span className="text-gray-400 text-xs">+{user.interests.length - 4} more</span>
            )}
          </div>
        </div>

        {/* Stats */}
        <div className="flex items-center justify-between text-xs text-gray-500">
          <div className="flex items-center space-x-4">
            <span>❤️ {user.stats.matches} matches</span>
            <span>👁️ {user.stats.profileViews} views</span>
          </div>
          <div className="flex items-center">
            <Clock className="w-3 h-3 mr-1" />
            {formatDistanceToNow(user.lastActive, { addSuffix: true })}
          </div>
        </div>
      </div>
    </motion.div>
  );
};