// app/api/process-text/route.js

function calculateBLEU(reference, candidate) {
  // Tokenize the strings into words
  const tokenize = (text) => text.toLowerCase().match(/\b\w+\b/g) || [];
  const referenceTokens = tokenize(reference);
  const candidateTokens = tokenize(candidate);

  const overlap = candidateTokens.filter(token => 
    referenceTokens.includes(token)
  ).length;

  const precision = overlap / candidateTokens.length;

  const brevityPenalty = Math.exp(
    Math.min(0, 1 - (referenceTokens.length / candidateTokens.length))
  );

  return precision * brevityPenalty;
}

function calculateSimilarity(reference, candidate) {
  const tokenize = (text) => text.toLowerCase().match(/\b\w+\b/g) || [];
  const referenceTokens = new Set(tokenize(reference));
  const candidateTokens = new Set(tokenize(candidate));

  const intersection = new Set(
    [...referenceTokens].filter(x => candidateTokens.has(x))
  );
  
  const union = new Set([...referenceTokens, ...candidateTokens]);
  return intersection.size / union.size;
}

export async function POST(request) {
  try {
    const { reference, candidate } = await request.json();

    if (!reference || !candidate) {
      throw new Error('Both reference and candidate texts are required');
    }

    // Filler words array
    const filler_words = [
      "um", "uh", "like", "you know", "so", "actually", "basically",
      "literally", "right", "well", "i mean", "kind of", "sort of",
      "okay", "okay so", "hmm", "er", "ah", "ehm", "ya know"
    ];

    // Calculate filler words count
    const candidateLower = candidate.toLowerCase();
    const fillerWordsCount = filler_words.reduce((count, word) => {
      // Use a regex to match whole words to avoid partial matches
      const matches = (candidateLower.match(new RegExp(`\\b${word}\\b`, 'g')) || []).length;
      return count + matches;
    }, 0);

    // Calculate metrics
    const bleuScore = calculateBLEU(reference, candidate);
    const similarity = calculateSimilarity(reference, candidate);

    // Calculate keyword coverage
    const getKeywords = (text) => {
      // Simple keyword extraction (words longer than 4 characters)
      return new Set(
        text.toLowerCase()
          .match(/\b\w{4,}\b/g)
          ?.filter(word => !['this', 'that', 'with', 'from'].includes(word)) || []
      );
    };

    const referenceKeywords = getKeywords(reference);
    const candidateKeywords = getKeywords(candidate);
    const matchedKeywords = [...referenceKeywords]
      .filter(keyword => candidateKeywords.has(keyword));

    const keywordCoverage = referenceKeywords.size > 0 
      ? matchedKeywords.length / referenceKeywords.size
      : 0;

    return Response.json({
      success: true,
      metrics: {
        bleuScore: Math.round(bleuScore * 100),
        similarity: Math.round(similarity * 100),
        keywordCoverage: Math.round(keywordCoverage * 100),
        matchedKeywords,
        fillerWordsCount
      }
    });
  } catch (error) {
    return Response.json({
      success: false,
      error: error.message
    }, { status: 500 });
  }
}