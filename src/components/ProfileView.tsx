import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { 
  User, 
  Settings, 
  Heart, 
  MessageCircle, 
  TrendingUp, 
  MapPin, 
  Calendar,
  Edit3,
  Plus,
  Clock
} from 'lucide-react';
import { User as UserType } from '../types';
import { StatusCreator } from './StatusCreator';
import { formatDistanceToNow } from 'date-fns';

interface ProfileViewProps {
  user: UserType;
  onUpdateStatus: (status: { text: string; expiresIn: number }) => void;
}

export const ProfileView: React.FC<ProfileViewProps> = ({ user, onUpdateStatus }) => {
  const [showStatusCreator, setShowStatusCreator] = useState(false);
  const [currentPhotoIndex, setCurrentPhotoIndex] = useState(0);

  const handleCreateStatus = (status: { text: string; expiresIn: number }) => {
    onUpdateStatus(status);
    setShowStatusCreator(false);
  };

  const nextPhoto = () => {
    setCurrentPhotoIndex((prev) => (prev + 1) % user.photos.length);
  };

  const prevPhoto = () => {
    setCurrentPhotoIndex((prev) => (prev - 1 + user.photos.length) % user.photos.length);
  };

  const isStatusActive = user.status && user.status.expiresAt > Date.now();

  return (
    <div className="h-full overflow-y-auto bg-gray-50">
      {/* Header */}
      <div className="relative">
        {/* Photo Section */}
        <div className="relative h-96 overflow-hidden">
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
              />
              <button
                onClick={nextPhoto}
                className="absolute right-0 top-0 w-1/2 h-full z-10 bg-transparent"
              />
              
              {/* Photo Indicators */}
              <div className="absolute bottom-4 left-4 right-4 flex space-x-1">
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

          {/* Settings Button */}
          <button className="absolute top-4 right-4 bg-black/50 backdrop-blur-sm rounded-full p-2">
            <Settings className="w-5 h-5 text-white" />
          </button>
        </div>

        {/* Profile Info Overlay */}
        <div className="absolute bottom-0 left-0 right-0 bg-gradient-to-t from-black/70 to-transparent p-6 text-white">
          <h1 className="text-3xl font-bold mb-1">{user.name}, {user.age}</h1>
          <div className="flex items-center text-white/90">
            <MapPin className="w-4 h-4 mr-1" />
            {user.location.city}
          </div>
        </div>
      </div>

      {/* Content */}
      <div className="p-6 space-y-6">
        {/* Status Section */}
        <div className="bg-white rounded-2xl p-6 shadow-sm">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-lg font-semibold">Status</h2>
            <button
              onClick={() => setShowStatusCreator(true)}
              className="flex items-center space-x-2 text-primary-600 hover:text-primary-700 transition-colors"
            >
              {isStatusActive ? <Edit3 className="w-4 h-4" /> : <Plus className="w-4 h-4" />}
              <span className="text-sm font-medium">
                {isStatusActive ? 'Edit' : 'Add Status'}
              </span>
            </button>
          </div>

          {isStatusActive ? (
            <div className="bg-gradient-to-r from-primary-50 to-accent-50 rounded-xl p-4">
              <p className="text-gray-800 mb-2">"{user.status!.text}"</p>
              <div className="flex items-center text-xs text-gray-500">
                <Clock className="w-3 h-3 mr-1" />
                Expires {formatDistanceToNow(user.status!.expiresAt, { addSuffix: true })}
              </div>
            </div>
          ) : (
            <div className="text-center py-8 text-gray-500">
              <div className="w-12 h-12 bg-gray-100 rounded-full flex items-center justify-center mx-auto mb-3">
                <Plus className="w-6 h-6" />
              </div>
              <p className="text-sm">Share what you're up to</p>
            </div>
          )}
        </div>

        {/* Stats */}
        <div className="bg-white rounded-2xl p-6 shadow-sm">
          <h2 className="text-lg font-semibold mb-4">Your Stats</h2>
          <div className="grid grid-cols-2 gap-4">
            <div className="text-center">
              <div className="w-12 h-12 bg-red-100 rounded-full flex items-center justify-center mx-auto mb-2">
                <Heart className="w-6 h-6 text-red-500" />
              </div>
              <p className="text-2xl font-bold text-gray-900">{user.stats.matches}</p>
              <p className="text-sm text-gray-500">Matches</p>
            </div>
            <div className="text-center">
              <div className="w-12 h-12 bg-blue-100 rounded-full flex items-center justify-center mx-auto mb-2">
                <MessageCircle className="w-6 h-6 text-blue-500" />
              </div>
              <p className="text-2xl font-bold text-gray-900">{user.stats.profileViews}</p>
              <p className="text-sm text-gray-500">Profile Views</p>
            </div>
          </div>
          
          <div className="mt-6 pt-6 border-t border-gray-100">
            <div className="flex items-center justify-between">
              <span className="text-sm text-gray-600">ELO Score</span>
              <div className="flex items-center space-x-2">
                <TrendingUp className="w-4 h-4 text-green-500" />
                <span className="font-semibold text-gray-900">{user.eloScore}</span>
              </div>
            </div>
            <div className="mt-2 bg-gray-200 rounded-full h-2">
              <div 
                className="bg-gradient-to-r from-primary-500 to-accent-500 h-2 rounded-full"
                style={{ width: `${Math.min((user.eloScore - 1000) / 1000 * 100, 100)}%` }}
              />
            </div>
          </div>
        </div>

        {/* Bio */}
        {user.bio && (
          <div className="bg-white rounded-2xl p-6 shadow-sm">
            <h2 className="text-lg font-semibold mb-3">About</h2>
            <p className="text-gray-700 leading-relaxed">{user.bio}</p>
          </div>
        )}

        {/* Interests */}
        <div className="bg-white rounded-2xl p-6 shadow-sm">
          <h2 className="text-lg font-semibold mb-4">Interests</h2>
          <div className="flex flex-wrap gap-2">
            {user.interests.map((interest, index) => (
              <span
                key={index}
                className="bg-gradient-to-r from-primary-100 to-accent-100 text-primary-700 px-3 py-2 rounded-full text-sm font-medium"
              >
                {interest}
              </span>
            ))}
          </div>
        </div>

        {/* Activity */}
        <div className="bg-white rounded-2xl p-6 shadow-sm">
          <h2 className="text-lg font-semibold mb-4">Activity</h2>
          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <span className="text-sm text-gray-600">Last Active</span>
              <span className="text-sm font-medium text-gray-900">
                {formatDistanceToNow(user.lastActive, { addSuffix: true })}
              </span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-sm text-gray-600">Activity Score</span>
              <div className="flex items-center space-x-2">
                <div className="w-16 bg-gray-200 rounded-full h-2">
                  <div 
                    className="bg-green-500 h-2 rounded-full"
                    style={{ width: `${user.activityScore * 100}%` }}
                  />
                </div>
                <span className="text-sm font-medium text-gray-900">
                  {Math.round(user.activityScore * 100)}%
                </span>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Status Creator Modal */}
      <StatusCreator
        isOpen={showStatusCreator}
        onClose={() => setShowStatusCreator(false)}
        onCreateStatus={handleCreateStatus}
      />
    </div>
  );
};