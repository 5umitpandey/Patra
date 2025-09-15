import { User } from '../types';

const interests = [
  'Photography', 'Travel', 'Fitness', 'Music', 'Art', 'Cooking', 'Reading', 
  'Dancing', 'Hiking', 'Gaming', 'Movies', 'Yoga', 'Coffee', 'Wine', 
  'Technology', 'Fashion', 'Sports', 'Nature', 'Meditation', 'Writing'
];

const cities = [
  { name: 'New York', lat: 40.7128, lng: -74.0060 },
  { name: 'Los Angeles', lat: 34.0522, lng: -118.2437 },
  { name: 'Chicago', lat: 41.8781, lng: -87.6298 },
  { name: 'Miami', lat: 25.7617, lng: -80.1918 },
  { name: 'San Francisco', lat: 37.7749, lng: -122.4194 },
  { name: 'Austin', lat: 30.2672, lng: -97.7431 },
  { name: 'Seattle', lat: 47.6062, lng: -122.3321 },
  { name: 'Boston', lat: 42.3601, lng: -71.0589 }
];

const statusMessages = [
  "Just finished an amazing workout! 💪",
  "Coffee and good vibes ☕️",
  "Exploring the city today 🌆",
  "Weekend hiking adventures 🥾",
  "Trying out a new recipe 👨‍🍳",
  "Art gallery hopping 🎨",
  "Beach day vibes 🏖️",
  "Concert tonight! 🎵",
  "Reading by the fireplace 📚",
  "Sunday brunch mood 🥐"
];

const names = [
  'Emma', 'Liam', 'Olivia', 'Noah', 'Ava', 'Ethan', 'Sophia', 'Mason',
  'Isabella', 'William', 'Mia', 'James', 'Charlotte', 'Benjamin', 'Amelia',
  'Lucas', 'Harper', 'Henry', 'Evelyn', 'Alexander', 'Abigail', 'Michael',
  'Emily', 'Daniel', 'Elizabeth', 'Matthew', 'Sofia', 'Jackson', 'Avery',
  'Sebastian', 'Ella', 'David', 'Scarlett', 'Carter', 'Grace', 'Wyatt'
];

function generateRandomUser(id: string): User {
  const city = cities[Math.floor(Math.random() * cities.length)];
  const userInterests = interests
    .sort(() => 0.5 - Math.random())
    .slice(0, Math.floor(Math.random() * 6) + 3);
  
  const age = Math.floor(Math.random() * 20) + 22; // 22-42
  const hasStatus = Math.random() > 0.6;
  const now = Date.now();
  
  return {
    id,
    name: names[Math.floor(Math.random() * names.length)],
    age,
    bio: `${userInterests.slice(0, 3).join(', ')} enthusiast. Love exploring new places and meeting interesting people!`,
    photos: [
      `https://images.pexels.com/photos/${1000000 + Math.floor(Math.random() * 2000000)}/pexels-photo-${1000000 + Math.floor(Math.random() * 2000000)}.jpeg?auto=compress&cs=tinysrgb&w=400&h=600&fit=crop`,
      `https://images.pexels.com/photos/${1000000 + Math.floor(Math.random() * 2000000)}/pexels-photo-${1000000 + Math.floor(Math.random() * 2000000)}.jpeg?auto=compress&cs=tinysrgb&w=400&h=600&fit=crop`,
      `https://images.pexels.com/photos/${1000000 + Math.floor(Math.random() * 2000000)}/pexels-photo-${1000000 + Math.floor(Math.random() * 2000000)}.jpeg?auto=compress&cs=tinysrgb&w=400&h=600&fit=crop`
    ],
    interests: userInterests,
    location: {
      lat: city.lat + (Math.random() - 0.5) * 0.1,
      lng: city.lng + (Math.random() - 0.5) * 0.1,
      city: city.name
    },
    status: hasStatus ? {
      text: statusMessages[Math.floor(Math.random() * statusMessages.length)],
      timestamp: now - Math.floor(Math.random() * 3600000), // Within last hour
      expiresAt: now + (24 * 3600000) // Expires in 24 hours
    } : undefined,
    preferences: {
      ageRange: [Math.max(18, age - 8), Math.min(50, age + 8)],
      maxDistance: Math.floor(Math.random() * 50) + 10, // 10-60 miles
      interests: userInterests.slice(0, Math.floor(Math.random() * 3) + 2)
    },
    stats: {
      rightSwipes: Math.floor(Math.random() * 100),
      leftSwipes: Math.floor(Math.random() * 200),
      matches: Math.floor(Math.random() * 30),
      profileViews: Math.floor(Math.random() * 500) + 50
    },
    eloScore: Math.floor(Math.random() * 1000) + 1000, // 1000-2000
    attractivenessScore: Math.random() * 0.4 + 0.6, // 0.6-1.0
    activityScore: Math.random() * 0.5 + 0.5, // 0.5-1.0
    lastActive: now - Math.floor(Math.random() * 86400000) // Within last day
  };
}

export const mockUsers: User[] = Array.from({ length: 50 }, (_, i) => 
  generateRandomUser(`user-${i + 1}`)
);

export const currentUser: User = {
  id: 'current-user',
  name: 'You',
  age: 28,
  bio: 'Love technology, travel, and meeting new people!',
  photos: [
    'https://images.pexels.com/photos/1239291/pexels-photo-1239291.jpeg?auto=compress&cs=tinysrgb&w=400&h=600&fit=crop'
  ],
  interests: ['Technology', 'Travel', 'Photography', 'Fitness', 'Music'],
  location: {
    lat: 40.7128,
    lng: -74.0060,
    city: 'New York'
  },
  preferences: {
    ageRange: [22, 35],
    maxDistance: 25,
    interests: ['Technology', 'Travel', 'Art']
  },
  stats: {
    rightSwipes: 45,
    leftSwipes: 120,
    matches: 12,
    profileViews: 234
  },
  eloScore: 1650,
  attractivenessScore: 0.8,
  activityScore: 0.9,
  lastActive: Date.now()
};