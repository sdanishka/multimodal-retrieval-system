'use client';

import { useState, useCallback } from 'react';
import { useRouter } from 'next/navigation';
import { useDropzone } from 'react-dropzone';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Search,
  Upload,
  Image as ImageIcon,
  User,
  FileText,
  Sparkles,
  ArrowRight,
  Database,
  Zap,
  Layers,
  Activity,
} from 'lucide-react';
import axios from 'axios';

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

type QueryType = 'face' | 'image' | 'text';

interface SearchResult {
  // Full backend response
  entity_id?: string;
  score: number;
  confidence?: number;
  images?: string[];
  faces?: string[];
  metadata?: Record<string, any>;
  match_details?: {
    face: number;
    image: number;
    text: number;
  };
  // Colab server response
  index?: number;
  type?: string;
  image_path?: string;
}

// Helper to get image URL from result
const getImageUrl = (result: SearchResult): string | null => {
  // If we have an index, use the /image/{idx} endpoint
  if (result.index !== undefined) {
    return `${API_URL}/image/${result.index}`;
  }
  // If we have images array
  if (result.images && result.images.length > 0) {
    return result.images[0];
  }
  // If we have image_path
  if (result.image_path) {
    return result.image_path;
  }
  return null;
};

export default function HomePage() {
  const router = useRouter();
  const [queryType, setQueryType] = useState<QueryType>('text');
  const [textQuery, setTextQuery] = useState('');
  const [uploadedFile, setUploadedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [isSearching, setIsSearching] = useState(false);
  const [results, setResults] = useState<SearchResult[]>([]);
  const [searchTime, setSearchTime] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);

  const onDrop = useCallback((acceptedFiles: File[]) => {
    const file = acceptedFiles[0];
    if (file) {
      setUploadedFile(file);
      setPreviewUrl(URL.createObjectURL(file));
      setError(null);
    }
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'image/*': ['.jpeg', '.jpg', '.png', '.webp'],
    },
    maxFiles: 1,
  });

  const handleSearch = async () => {
    setError(null);
    setIsSearching(true);

    try {
      let response;

      if (queryType === 'text') {
        if (!textQuery.trim()) {
          throw new Error('Please enter a search query');
        }

        const formData = new FormData();
        formData.append('query', textQuery);
        formData.append('top_k', '20');

        response = await axios.post(`${API_URL}/search/text`, formData, {
          headers: { 'Content-Type': 'multipart/form-data' },
        });
      } else {
        if (!uploadedFile) {
          throw new Error('Please upload an image');
        }

        const formData = new FormData();
        formData.append('file', uploadedFile);
        formData.append('top_k', '20');

        const endpoint = queryType === 'face' ? '/search/face' : '/search/image';
        response = await axios.post(`${API_URL}${endpoint}`, formData, {
          headers: { 'Content-Type': 'multipart/form-data' },
        });
      }

      setResults(response.data.results);
      setSearchTime(response.data.search_time_ms);
    } catch (err: any) {
      console.error('Search error:', err);
      setError(err.response?.data?.detail || err.message || 'Search failed');
      setResults([]);
    } finally {
      setIsSearching(false);
    }
  };

  const clearSearch = () => {
    setUploadedFile(null);
    setPreviewUrl(null);
    setTextQuery('');
    setResults([]);
    setSearchTime(null);
    setError(null);
  };

  const queryTypeConfig = {
    face: {
      icon: User,
      label: 'Face Search',
      description: 'Find people by facial features',
      color: 'electric',
    },
    image: {
      icon: ImageIcon,
      label: 'Image Search',
      description: 'Find visually similar content',
      color: 'neon-purple',
    },
    text: {
      icon: FileText,
      label: 'Text Search',
      description: 'Describe what you\'re looking for',
      color: 'neon-pink',
    },
  };

  return (
    <div className="min-h-screen py-8 px-4 sm:px-6 lg:px-8">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <motion.header
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          className="text-center mb-12"
        >
          <div className="flex items-center justify-center gap-3 mb-4">
            <div className="p-3 rounded-xl bg-electric/20 border border-electric/30">
              <Sparkles className="w-8 h-8 text-electric" />
            </div>
          </div>
          <h1 className="text-4xl sm:text-5xl font-bold font-display text-pearl mb-4">
            Multimodal Retrieval
          </h1>
          <p className="text-lg text-silver max-w-2xl mx-auto">
            Search across faces, images, and text descriptions using state-of-the-art
            embeddings and vector search technology.
          </p>
        </motion.header>

        {/* Stats Bar */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1 }}
          className="glass rounded-2xl p-4 mb-8"
        >
          <div className="flex flex-wrap justify-center gap-8">
            {[
              { icon: Database, label: 'FAISS Indexes', value: '3 Active' },
              { icon: Zap, label: 'Embedding Models', value: 'CLIP + ArcFace' },
              { icon: Layers, label: 'Fusion Search', value: 'Weighted Ranking' },
              { icon: Activity, label: 'Real-time', value: '< 100ms' },
            ].map((stat, i) => (
              <div key={i} className="flex items-center gap-3">
                <div className="p-2 rounded-lg bg-steel/50">
                  <stat.icon className="w-5 h-5 text-electric" />
                </div>
                <div>
                  <p className="text-xs text-silver">{stat.label}</p>
                  <p className="text-sm font-medium text-pearl">{stat.value}</p>
                </div>
              </div>
            ))}
          </div>
        </motion.div>

        {/* Query Type Selector */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
          className="grid grid-cols-1 sm:grid-cols-3 gap-4 mb-8"
        >
          {(Object.keys(queryTypeConfig) as QueryType[]).map((type) => {
            const config = queryTypeConfig[type];
            const Icon = config.icon;
            const isSelected = queryType === type;

            return (
              <button
                key={type}
                onClick={() => {
                  setQueryType(type);
                  clearSearch();
                }}
                className={`relative p-6 rounded-xl border-2 transition-all duration-300 ${
                  isSelected
                    ? `border-${config.color} bg-${config.color}/10 shadow-glow`
                    : 'border-steel hover:border-silver bg-obsidian'
                }`}
              >
                <div className="flex items-start gap-4">
                  <div
                    className={`p-3 rounded-lg ${
                      isSelected ? `bg-${config.color}/20` : 'bg-steel'
                    }`}
                  >
                    <Icon
                      className={`w-6 h-6 ${
                        isSelected ? `text-${config.color}` : 'text-silver'
                      }`}
                    />
                  </div>
                  <div className="text-left">
                    <h3 className="font-display font-semibold text-pearl">
                      {config.label}
                    </h3>
                    <p className="text-sm text-silver mt-1">{config.description}</p>
                  </div>
                </div>
                {isSelected && (
                  <motion.div
                    layoutId="selector"
                    className="absolute inset-0 rounded-xl border-2 border-electric"
                    initial={false}
                    transition={{ type: 'spring', stiffness: 500, damping: 30 }}
                  />
                )}
              </button>
            );
          })}
        </motion.div>

        {/* Search Input */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.3 }}
          className="card mb-8"
        >
          <AnimatePresence mode="wait">
            {queryType === 'text' ? (
              <motion.div
                key="text-input"
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: 20 }}
              >
                <label className="block text-sm font-medium text-silver mb-2">
                  Search Query
                </label>
                <div className="relative">
                  <Search className="absolute left-4 top-1/2 -translate-y-1/2 w-5 h-5 text-silver" />
                  <input
                    type="text"
                    value={textQuery}
                    onChange={(e) => setTextQuery(e.target.value)}
                    onKeyDown={(e) => e.key === 'Enter' && handleSearch()}
                    placeholder="e.g., man with beard wearing black jacket at night"
                    className="input-field pl-12 text-lg"
                  />
                </div>
                <p className="mt-2 text-xs text-silver">
                  Describe the person, scene, or object you're looking for
                </p>
              </motion.div>
            ) : (
              <motion.div
                key="image-input"
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: 20 }}
              >
                <label className="block text-sm font-medium text-silver mb-2">
                  Upload {queryType === 'face' ? 'Face' : 'Image'}
                </label>
                <div
                  {...getRootProps()}
                  className={`dropzone flex flex-col items-center justify-center min-h-[200px] ${
                    isDragActive ? 'dropzone-active' : ''
                  }`}
                >
                  <input {...getInputProps()} />
                  {previewUrl ? (
                    <div className="relative">
                      <img
                        src={previewUrl}
                        alt="Preview"
                        className="max-h-48 rounded-lg object-contain"
                      />
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          clearSearch();
                        }}
                        className="absolute -top-2 -right-2 p-1 bg-error rounded-full text-white"
                      >
                        ×
                      </button>
                    </div>
                  ) : (
                    <>
                      <Upload className="w-12 h-12 text-silver mb-4" />
                      <p className="text-pearl font-medium">
                        {isDragActive
                          ? 'Drop the image here'
                          : 'Drag & drop an image, or click to select'}
                      </p>
                      <p className="text-sm text-silver mt-2">
                        Supports JPEG, PNG, WebP
                      </p>
                    </>
                  )}
                </div>
              </motion.div>
            )}
          </AnimatePresence>

          {/* Search Button */}
          <div className="flex items-center justify-between mt-6">
            <button onClick={clearSearch} className="btn-secondary">
              Clear
            </button>
            <button
              onClick={handleSearch}
              disabled={isSearching}
              className="btn-primary flex items-center gap-2"
            >
              {isSearching ? (
                <>
                  <div className="w-5 h-5 border-2 border-void border-t-transparent rounded-full animate-spin" />
                  Searching...
                </>
              ) : (
                <>
                  <Search className="w-5 h-5" />
                  Search
                  <ArrowRight className="w-4 h-4" />
                </>
              )}
            </button>
          </div>
        </motion.div>

        {/* Error Display */}
        <AnimatePresence>
          {error && (
            <motion.div
              initial={{ opacity: 0, y: -10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -10 }}
              className="mb-8 p-4 rounded-xl bg-error/20 border border-error/30 text-error"
            >
              <p className="font-medium">Search Error</p>
              <p className="text-sm mt-1">{error}</p>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Results */}
        <AnimatePresence>
          {results.length > 0 && (
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: 20 }}
            >
              {/* Results Header */}
              <div className="flex items-center justify-between mb-6">
                <div>
                  <h2 className="text-2xl font-display font-bold text-pearl">
                    Search Results
                  </h2>
                  <p className="text-sm text-silver mt-1">
                    Found {results.length} entities in {searchTime?.toFixed(2)}ms
                  </p>
                </div>
                <div className="flex items-center gap-2">
                  <span className="badge-electric">
                    {queryType.charAt(0).toUpperCase() + queryType.slice(1)} Search
                  </span>
                </div>
              </div>

              {/* Results Grid */}
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                {results.map((result, index) => (
                  <motion.div
                    key={result.entity_id || result.index || index}
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: index * 0.05 }}
                    className="result-card group"
                  >
                    {/* Image Preview */}
                    <div className="aspect-video bg-steel relative overflow-hidden">
                      {(() => {
                        const imageUrl = getImageUrl(result);
                        if (imageUrl) {
                          return (
                            <img
                              src={imageUrl}
                              alt={result.entity_id || `Result ${index}`}
                              className="w-full h-full object-cover"
                              onError={(e) => {
                                // Hide broken images
                                (e.target as HTMLImageElement).style.display = 'none';
                              }}
                            />
                          );
                        }
                        return (
                          <div className="w-full h-full flex items-center justify-center bg-gradient-to-br from-electric/20 to-neon-purple/20">
                            <div className="text-center">
                              <User className="w-12 h-12 text-silver mx-auto mb-2" />
                              <span className="text-xs text-silver">Index: {result.index ?? 'N/A'}</span>
                            </div>
                          </div>
                        );
                      })()}
                      
                      {/* Score Badge */}
                      <div className="absolute top-3 right-3">
                        <div className="px-3 py-1 rounded-full bg-void/80 backdrop-blur text-sm font-mono">
                          <span className="text-electric">{(result.score * 100).toFixed(1)}%</span>
                        </div>
                      </div>
                    </div>

                    {/* Card Content */}
                    <div className="p-4">
                      <div className="flex items-start justify-between mb-3">
                        <div>
                          <h3 className="font-display font-semibold text-pearl truncate">
                            {result.metadata?.name || result.entity_id || `Match #${result.index ?? index + 1}`}
                          </h3>
                          <p className="text-xs text-silver mt-1 font-mono">
                            {result.entity_id || `Index: ${result.index ?? index}`}
                          </p>
                        </div>
                      </div>

                      {/* Match Details */}
                      {result.match_details && (
                        <div className="space-y-2">
                          <p className="text-xs text-silver uppercase tracking-wider">
                            Match Scores
                          </p>
                          <div className="grid grid-cols-3 gap-2">
                            {Object.entries(result.match_details).map(([key, value]) => (
                              <div
                                key={key}
                                className="text-center p-2 rounded-lg bg-steel/50"
                              >
                                <p className="text-xs text-silver capitalize">{key}</p>
                                <p className="text-sm font-mono text-pearl">
                                  {((value as number) * 100).toFixed(0)}%
                                </p>
                              </div>
                            ))}
                          </div>
                        </div>
                      )}

                      {/* Simple Score Display (for Colab API) */}
                      {!result.match_details && (
                        <div className="space-y-2">
                          <p className="text-xs text-silver uppercase tracking-wider">
                            Similarity
                          </p>
                          <div className="w-full bg-steel rounded-full h-2">
                            <div 
                              className="bg-gradient-to-r from-electric to-neon-purple h-2 rounded-full"
                              style={{ width: `${result.score * 100}%` }}
                            />
                          </div>
                          <p className="text-xs text-silver text-right">
                            Type: {result.type || 'unknown'}
                          </p>
                        </div>
                      )}

                      {/* Tags */}
                      {result.metadata?.tags && result.metadata.tags.length > 0 && (
                        <div className="mt-3 flex flex-wrap gap-1">
                          {result.metadata.tags.slice(0, 3).map((tag: string) => (
                            <span
                              key={tag}
                              className="px-2 py-0.5 text-xs rounded bg-steel text-silver"
                            >
                              {tag}
                            </span>
                          ))}
                        </div>
                      )}

                      {/* Stats */}
                      <div className="mt-4 pt-3 border-t border-steel flex items-center justify-between text-xs text-silver">
                        <span>{result.images?.length ?? 0} images</span>
                        <span>{result.faces?.length ?? 0} faces</span>
                        {result.confidence !== undefined ? (
                          <span className="text-success">
                            {(result.confidence * 100).toFixed(0)}% conf
                          </span>
                        ) : (
                          <span className="text-electric">
                            {(result.score * 100).toFixed(1)}% match
                          </span>
                        )}
                      </div>
                    </div>
                  </motion.div>
                ))}
              </div>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Empty State */}
        {results.length === 0 && !isSearching && !error && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="text-center py-16"
          >
            <div className="inline-flex p-4 rounded-full bg-steel/30 mb-6">
              <Search className="w-12 h-12 text-silver" />
            </div>
            <h3 className="text-xl font-display font-semibold text-pearl mb-2">
              Ready to Search
            </h3>
            <p className="text-silver max-w-md mx-auto">
              Enter a text description or upload an image to search across your
              indexed entities using multimodal AI.
            </p>
          </motion.div>
        )}
      </div>
    </div>
  );
}
