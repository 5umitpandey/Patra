import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Plus, X, Clock, Smile, Camera, MapPin, Coffee, Music } from 'lucide-react';

interface StatusCreatorProps {
  onCreateStatus: (status: { text: string; expiresIn: number }) => void;
  isOpen: boolean;
  onClose: () => void;
}

const statusSuggestions = [
  { icon: Coffee, text: "Coffee and good vibes ☕️", category: "mood" },
  { icon: Music, text: "Vibing to some great music 🎵", category: "activity" },
  { icon: MapPin, text: "Exploring the city today 🌆", category: "location" },
  { icon: Camera, text: "Perfect lighting for photos ✨", category: "moment" },
  { text: "Weekend adventures await! 🌟", category: "mood" },
  { text: "Just finished an amazing workout 💪", category: "activity" },
  { text: "Trying out a new recipe 👨‍🍳", category: "activity" },
  { text: "Art gallery hopping 🎨", category: "activity" },
  { text: "Beach day vibes 🏖️", category: "location" },
  { text: "Reading by the fireplace 📚", category: "mood" },
];

const expirationOptions = [
  { label: "1 hour", value: 1 * 60 * 60 * 1000 },
  { label: "4 hours", value: 4 * 60 * 60 * 1000 },
  { label: "12 hours", value: 12 * 60 * 60 * 1000 },
  { label: "24 hours", value: 24 * 60 * 60 * 1000 },
];

export const StatusCreator: React.FC<StatusCreatorProps> = ({
  onCreateStatus,
  isOpen,
  onClose
}) => {
  const [statusText, setStatusText] = useState('');
  const [selectedExpiration, setSelectedExpiration] = useState(expirationOptions[2].value);
  const [selectedCategory, setSelectedCategory] = useState<string | null>(null);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (statusText.trim()) {
      onCreateStatus({
        text: statusText.trim(),
        expiresIn: selectedExpiration
      });
      setStatusText('');
      onClose();
    }
  };

  const handleSuggestionClick = (suggestion: string) => {
    setStatusText(suggestion);
  };

  const filteredSuggestions = selectedCategory
    ? statusSuggestions.filter(s => s.category === selectedCategory)
    : statusSuggestions;

  return (
    <AnimatePresence>
      {isOpen && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          className="fixed inset-0 bg-black/50 backdrop-blur-sm z-50 flex items-end sm:items-center justify-center p-4"
          onClick={onClose}
        >
          <motion.div
            initial={{ y: 100, opacity: 0 }}
            animate={{ y: 0, opacity: 1 }}
            exit={{ y: 100, opacity: 0 }}
            className="bg-white rounded-t-3xl sm:rounded-3xl w-full max-w-md max-h-[80vh] overflow-hidden"
            onClick={(e) => e.stopPropagation()}
          >
            {/* Header */}
            <div className="flex items-center justify-between p-6 border-b border-gray-100">
              <h2 className="text-xl font-bold gradient-text">Create Status</h2>
              <button
                onClick={onClose}
                className="p-2 hover:bg-gray-100 rounded-full transition-colors"
              >
                <X className="w-5 h-5" />
              </button>
            </div>

            <div className="p-6 overflow-y-auto">
              {/* Status Input */}
              <form onSubmit={handleSubmit} className="space-y-6">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    What's happening?
                  </label>
                  <textarea
                    value={statusText}
                    onChange={(e) => setStatusText(e.target.value)}
                    placeholder="Share what you're up to..."
                    className="w-full p-4 border border-gray-200 rounded-xl resize-none focus:ring-2 focus:ring-primary-500 focus:border-transparent"
                    rows={3}
                    maxLength={100}
                  />
                  <div className="flex justify-between items-center mt-2">
                    <span className="text-xs text-gray-500">
                      {statusText.length}/100 characters
                    </span>
                    <div className="flex items-center text-xs text-gray-500">
                      <Clock className="w-3 h-3 mr-1" />
                      Expires in {expirationOptions.find(opt => opt.value === selectedExpiration)?.label}
                    </div>
                  </div>
                </div>

                {/* Expiration Options */}
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-3">
                    Status duration
                  </label>
                  <div className="grid grid-cols-2 gap-2">
                    {expirationOptions.map((option) => (
                      <button
                        key={option.value}
                        type="button"
                        onClick={() => setSelectedExpiration(option.value)}
                        className={`p-3 rounded-lg border text-sm font-medium transition-colors ${
                          selectedExpiration === option.value
                            ? 'bg-primary-50 border-primary-500 text-primary-700'
                            : 'bg-gray-50 border-gray-200 text-gray-700 hover:bg-gray-100'
                        }`}
                      >
                        {option.label}
                      </button>
                    ))}
                  </div>
                </div>

                {/* Category Filter */}
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-3">
                    Browse by category
                  </label>
                  <div className="flex flex-wrap gap-2 mb-4">
                    {['mood', 'activity', 'location', 'moment'].map((category) => (
                      <button
                        key={category}
                        type="button"
                        onClick={() => setSelectedCategory(selectedCategory === category ? null : category)}
                        className={`px-3 py-1 rounded-full text-xs font-medium transition-colors ${
                          selectedCategory === category
                            ? 'bg-primary-500 text-white'
                            : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
                        }`}
                      >
                        {category}
                      </button>
                    ))}
                  </div>
                </div>

                {/* Status Suggestions */}
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-3">
                    Quick suggestions
                  </label>
                  <div className="space-y-2 max-h-40 overflow-y-auto">
                    {filteredSuggestions.map((suggestion, index) => (
                      <button
                        key={index}
                        type="button"
                        onClick={() => handleSuggestionClick(suggestion.text)}
                        className="w-full text-left p-3 rounded-lg bg-gray-50 hover:bg-gray-100 transition-colors text-sm"
                      >
                        <div className="flex items-center space-x-2">
                          {suggestion.icon && <suggestion.icon className="w-4 h-4 text-gray-500" />}
                          <span>{suggestion.text}</span>
                        </div>
                      </button>
                    ))}
                  </div>
                </div>

                {/* Submit Button */}
                <button
                  type="submit"
                  disabled={!statusText.trim()}
                  className="w-full bg-gradient-to-r from-primary-500 to-accent-500 text-white py-4 rounded-xl font-semibold disabled:opacity-50 disabled:cursor-not-allowed hover:shadow-lg transition-all duration-200"
                >
                  Share Status
                </button>
              </form>
            </div>
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>
  );
};