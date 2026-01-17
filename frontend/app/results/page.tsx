'use client';

import { useSearchParams } from 'next/navigation';
import { useEffect, useState, Suspense } from 'react';
import Link from 'next/link';
import { motion } from 'framer-motion';
import {
  ArrowLeft,
  User,
  Image as ImageIcon,
  FileText,
  Clock,
  Zap,
  ExternalLink,
} from 'lucide-react';

interface SearchResult {
  entity_id: string;
  score: number;
  confidence: number;
  images: string[];
  faces: string[];
  metadata: Record<string, any>;
  match_details: {
    face: number;
    image: number;
    text: number;
  };
}

function ResultsContent() {
  const searchParams = useSearchParams();
  const [results, setResults] = useState<SearchResult[]>([]);
  const [queryType, setQueryType] = useState<string>('');
  const [searchTime, setSearchTime] = useState<number>(0);

  useEffect(() => {
    const data = searchParams.get('data');
    if (data) {
      try {
        const parsed = JSON.parse(decodeURIComponent(data));
        setResults(parsed.results || []);
        setQueryType(parsed.query_type || '');
        setSearchTime(parsed.search_time_ms || 0);
      } catch (e) {
        console.error('Failed to parse results:', e);
      }
    }
  }, [searchParams]);

  const ScoreRing = ({ score }: { score: number }) => {
    const circumference = 2 * Math.PI * 18;
    const offset = circumference - (score * circumference);
    
    return (
      <div className="relative w-14 h-14">
        <svg className="w-full h-full transform -rotate-90">
          <circle
            cx="28"
            cy="28"
            r="18"
            fill="none"
            stroke="#2a2a3a"
            strokeWidth="4"
          />
          <circle
            cx="28"
            cy="28"
            r="18"
            fill="none"
            stroke="#00d4ff"
            strokeWidth="4"
            strokeLinecap="round"
            strokeDasharray={circumference}
            strokeDashoffset={offset}
            style={{ transition: 'stroke-dashoffset 0.5s ease-out' }}
          />
        </svg>
        <div className="absolute inset-0 flex items-center justify-center">
          <span className="text-sm font-mono text-pearl">
            {Math.round(score * 100)}
          </span>
        </div>
      </div>
    );
  };

  return (
    <div className="min-h-screen py-8 px-4 sm:px-6 lg:px-8">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="flex items-center justify-between mb-8">
          <div className="flex items-center gap-4">
            <Link
              href="/"
              className="p-2 rounded-lg bg-steel hover:bg-slate-deep transition-colors"
            >
              <ArrowLeft className="w-5 h-5 text-pearl" />
            </Link>
            <div>
              <h1 className="text-2xl font-display font-bold text-pearl">
                Search Results
              </h1>
              <p className="text-sm text-silver">
                {results.length} entities found
              </p>
            </div>
          </div>

          <div className="flex items-center gap-4">
            <div className="flex items-center gap-2 px-3 py-2 rounded-lg bg-steel/50">
              {queryType === 'face' && <User className="w-4 h-4 text-electric" />}
              {queryType === 'image' && <ImageIcon className="w-4 h-4 text-neon-purple" />}
              {queryType === 'text' && <FileText className="w-4 h-4 text-neon-pink" />}
              <span className="text-sm text-pearl capitalize">{queryType} Search</span>
            </div>
            <div className="flex items-center gap-2 px-3 py-2 rounded-lg bg-steel/50">
              <Clock className="w-4 h-4 text-silver" />
              <span className="text-sm text-pearl">{searchTime.toFixed(2)}ms</span>
            </div>
          </div>
        </div>

        {/* Results Grid */}
        {results.length > 0 ? (
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {results.map((result, index) => (
              <motion.div
                key={result.entity_id}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: index * 0.05 }}
                className="card group"
              >
                <div className="flex gap-6">
                  {/* Image Preview */}
                  <div className="w-32 h-32 rounded-lg bg-steel overflow-hidden flex-shrink-0">
                    {result.images.length > 0 ? (
                      <img
                        src={result.images[0]}
                        alt={result.entity_id}
                        className="w-full h-full object-cover"
                      />
                    ) : result.faces.length > 0 ? (
                      <img
                        src={result.faces[0]}
                        alt={result.entity_id}
                        className="w-full h-full object-cover"
                      />
                    ) : (
                      <div className="w-full h-full flex items-center justify-center">
                        <User className="w-12 h-12 text-silver" />
                      </div>
                    )}
                  </div>

                  {/* Content */}
                  <div className="flex-1 min-w-0">
                    <div className="flex items-start justify-between mb-3">
                      <div>
                        <h3 className="font-display font-semibold text-pearl truncate">
                          {result.metadata?.name || 'Unknown Entity'}
                        </h3>
                        <p className="text-xs text-silver font-mono mt-1">
                          {result.entity_id}
                        </p>
                      </div>
                      <ScoreRing score={result.score} />
                    </div>

                    {/* Match Details */}
                    <div className="grid grid-cols-3 gap-2 mb-4">
                      <div className="text-center p-2 rounded bg-obsidian">
                        <div className="flex items-center justify-center gap-1 mb-1">
                          <User className="w-3 h-3 text-electric" />
                          <span className="text-xs text-silver">Face</span>
                        </div>
                        <p className="text-sm font-mono text-pearl">
                          {(result.match_details.face * 100).toFixed(0)}%
                        </p>
                      </div>
                      <div className="text-center p-2 rounded bg-obsidian">
                        <div className="flex items-center justify-center gap-1 mb-1">
                          <ImageIcon className="w-3 h-3 text-neon-purple" />
                          <span className="text-xs text-silver">Image</span>
                        </div>
                        <p className="text-sm font-mono text-pearl">
                          {(result.match_details.image * 100).toFixed(0)}%
                        </p>
                      </div>
                      <div className="text-center p-2 rounded bg-obsidian">
                        <div className="flex items-center justify-center gap-1 mb-1">
                          <FileText className="w-3 h-3 text-neon-pink" />
                          <span className="text-xs text-silver">Text</span>
                        </div>
                        <p className="text-sm font-mono text-pearl">
                          {(result.match_details.text * 100).toFixed(0)}%
                        </p>
                      </div>
                    </div>

                    {/* Metadata Tags */}
                    {result.metadata?.tags && (
                      <div className="flex flex-wrap gap-1">
                        {result.metadata.tags.slice(0, 4).map((tag: string) => (
                          <span
                            key={tag}
                            className="px-2 py-0.5 text-xs rounded bg-steel text-silver"
                          >
                            {tag}
                          </span>
                        ))}
                      </div>
                    )}
                  </div>
                </div>

                {/* Footer */}
                <div className="mt-4 pt-4 border-t border-steel flex items-center justify-between">
                  <div className="flex items-center gap-4 text-xs text-silver">
                    <span>{result.images.length} images</span>
                    <span>{result.faces.length} faces</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <span className="badge-success">
                      {(result.confidence * 100).toFixed(0)}% confidence
                    </span>
                  </div>
                </div>
              </motion.div>
            ))}
          </div>
        ) : (
          <div className="text-center py-16">
            <div className="inline-flex p-4 rounded-full bg-steel/30 mb-6">
              <Zap className="w-12 h-12 text-silver" />
            </div>
            <h3 className="text-xl font-display font-semibold text-pearl mb-2">
              No Results Found
            </h3>
            <p className="text-silver">
              Try a different search query or upload a different image.
            </p>
            <Link href="/" className="btn-primary inline-flex items-center gap-2 mt-6">
              <ArrowLeft className="w-4 h-4" />
              Back to Search
            </Link>
          </div>
        )}
      </div>
    </div>
  );
}

export default function ResultsPage() {
  return (
    <Suspense fallback={
      <div className="min-h-screen flex items-center justify-center">
        <div className="w-8 h-8 border-2 border-electric border-t-transparent rounded-full animate-spin" />
      </div>
    }>
      <ResultsContent />
    </Suspense>
  );
}
