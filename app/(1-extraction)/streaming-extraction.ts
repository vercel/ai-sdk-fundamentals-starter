import dotenvFlow from 'dotenv-flow';
dotenvFlow.config(); // Load environment variables (API keys, etc.)
import fs from 'fs';
import path from 'path';
import { generateObject, streamText } from 'ai'; // AI SDK's structured output and streaming functions
import { z } from 'zod';
import { openai } from '@ai-sdk/openai';

// Configuration
const DEFAULT_CHUNK_SIZE = 4000; // tokens per chunk (conservative estimate)
const OVERLAP_SIZE = 200; // tokens to overlap between chunks for context continuity
const CHECKPOINT_DIR = '.checkpoints';
const MAX_RETRIES = 3;

// Extraction schema for structured output
const ExtractionSchema = z.object({
  keyTakeaway: z.string().describe('Main takeaway in 50 words'),
  companies: z.array(z.string()).describe('All company names mentioned'),
  concepts: z.object({
    business: z.array(z.string()),
    technical: z.array(z.string()),
  }),
  quotes: z.array(
    z.object({
      quote: z.string(),
      speaker: z.string().nullable(),
    })
  ),
  summary: z.string().describe('Brief summary of this chunk'),
});

type ExtractionResult = z.infer<typeof ExtractionSchema>;
type ChunkResult = {
  chunkIndex: number;
  result: ExtractionResult;
  success: boolean;
  error?: string;
  processingTime: number;
};

// Progress tracking interface
interface ProgressTracker {
  totalChunks: number;
  completedChunks: number;
  failedChunks: number;
  startTime: number;
  chunkResults: ChunkResult[];
}

// Checkpoint data structure
interface Checkpoint {
  filePath: string;
  totalChunks: number;
  completedChunks: number[];
  chunkResults: ChunkResult[];
  timestamp: number;
}

/**
 * Estimate token count (rough approximation: 1 token ≈ 4 characters)
 */
function estimateTokens(text: string): number {
  return Math.ceil(text.length / 4);
}

/**
 * Split text into chunks respecting token limits with overlap
 */
function chunkText(text: string, chunkSize: number = DEFAULT_CHUNK_SIZE, overlap: number = OVERLAP_SIZE): string[] {
  const chunks: string[] = [];
  const estimatedTokens = estimateTokens(text);
  
  if (estimatedTokens <= chunkSize) {
    return [text];
  }

  // Split by sentences first for better context preservation
  const sentences = text.match(/[^.!?]+[.!?]+/g) || [text];
  let currentChunk = '';
  let currentTokens = 0;

  for (let i = 0; i < sentences.length; i++) {
    const sentence = sentences[i];
    const sentenceTokens = estimateTokens(sentence);

    if (currentTokens + sentenceTokens > chunkSize && currentChunk) {
      chunks.push(currentChunk.trim());
      
      // Start new chunk with overlap from previous chunk
      const words = currentChunk.split(/\s+/);
      const overlapWords = words.slice(-Math.floor(overlap / 4)); // Approximate overlap
      currentChunk = overlapWords.join(' ') + ' ' + sentence;
      currentTokens = estimateTokens(currentChunk);
    } else {
      currentChunk += sentence;
      currentTokens += sentenceTokens;
    }
  }

  if (currentChunk.trim()) {
    chunks.push(currentChunk.trim());
  }

  return chunks;
}

/**
 * Ensure checkpoint directory exists
 */
function ensureCheckpointDir(): void {
  if (!fs.existsSync(CHECKPOINT_DIR)) {
    fs.mkdirSync(CHECKPOINT_DIR, { recursive: true });
  }
}

/**
 * Get checkpoint file path for a given document
 */
function getCheckpointPath(filePath: string): string {
  const fileName = path.basename(filePath, path.extname(filePath));
  return path.join(CHECKPOINT_DIR, `${fileName}.checkpoint.json`);
}

/**
 * Load checkpoint if exists
 */
function loadCheckpoint(filePath: string): Checkpoint | null {
  try {
    const checkpointPath = getCheckpointPath(filePath);
    if (fs.existsSync(checkpointPath)) {
      const data = fs.readFileSync(checkpointPath, 'utf-8');
      return JSON.parse(data);
    }
  } catch (error) {
    console.warn('⚠️  Failed to load checkpoint:', error);
  }
  return null;
}

/**
 * Save checkpoint
 */
function saveCheckpoint(filePath: string, progress: ProgressTracker): void {
  try {
    ensureCheckpointDir();
    const checkpointPath = getCheckpointPath(filePath);
    const checkpoint: Checkpoint = {
      filePath,
      totalChunks: progress.totalChunks,
      completedChunks: progress.chunkResults
        .filter((r) => r.success)
        .map((r) => r.chunkIndex),
      chunkResults: progress.chunkResults,
      timestamp: Date.now(),
    };
    fs.writeFileSync(checkpointPath, JSON.stringify(checkpoint, null, 2));
  } catch (error) {
    console.warn('⚠️  Failed to save checkpoint:', error);
  }
}

/**
 * Clear checkpoint after successful completion
 */
function clearCheckpoint(filePath: string): void {
  try {
    const checkpointPath = getCheckpointPath(filePath);
    if (fs.existsSync(checkpointPath)) {
      fs.unlinkSync(checkpointPath);
    }
  } catch (error) {
    console.warn('⚠️  Failed to clear checkpoint:', error);
  }
}

/**
 * Process a single chunk with structured extraction
 * Supports both streaming (streamText) and non-streaming (generateObject) modes
 */
async function processChunk(
  chunk: string,
  chunkIndex: number,
  totalChunks: number,
  retries: number = 0,
  useStreaming: boolean = false
): Promise<ChunkResult> {
  const startTime = Date.now();
  
  try {
    let structuredOutput: ExtractionResult;

    if (useStreaming) {
      // Use streamText for real-time streaming with JSON parsing
      const streamResult = await streamText({
        model: openai('gpt-4o-mini'),
        prompt: `Extract structured information from this document chunk (${chunkIndex + 1}/${totalChunks}) and return ONLY valid JSON matching this schema:
{
  "keyTakeaway": "string (50 words)",
  "companies": ["string"],
  "concepts": {
    "business": ["string"],
    "technical": ["string"]
  },
  "quotes": [{"quote": "string", "speaker": "string or null"}],
  "summary": "string"
}

Document chunk:
${chunk}

Return ONLY the JSON object, no other text.`,
      });

      // Stream and collect text
      let fullText = '';
      for await (const textChunk of streamResult.textStream) {
        fullText += textChunk;
        process.stdout.write(`\r  📊 Chunk ${chunkIndex + 1}/${totalChunks}: Streaming... ${fullText.length} chars`);
      }
      process.stdout.write('\r' + ' '.repeat(80) + '\r'); // Clear line

      // Parse JSON from streamed text
      try {
        // Extract JSON from text (handle cases where model adds extra text)
        const jsonMatch = fullText.match(/\{[\s\S]*\}/);
        if (jsonMatch) {
          structuredOutput = JSON.parse(jsonMatch[0]);
          // Validate against schema
          structuredOutput = ExtractionSchema.parse(structuredOutput);
        } else {
          throw new Error('No JSON found in streamed response');
        }
      } catch (parseError: any) {
        throw new Error(`Failed to parse JSON from stream: ${parseError.message}`);
      }
    } else {
      // Use generateObject for reliable structured output (default)
      const result = await generateObject({
        model: openai('gpt-4o-mini'), // Using cost-effective model for large documents
        schema: ExtractionSchema,
        prompt: `Extract structured information from this document chunk (${chunkIndex + 1}/${totalChunks}).

Document chunk:
${chunk}

Extract:
- Key takeaway (50 words)
- All company names
- Business and technical concepts
- Quotes with speakers
- Brief summary`,
      });

      structuredOutput = result.object;
    }

    const processingTime = Date.now() - startTime;

    return {
      chunkIndex,
      result: structuredOutput,
      success: true,
      processingTime,
    };
  } catch (error: any) {
    const processingTime = Date.now() - startTime;
    
    if (retries < MAX_RETRIES) {
      console.log(`\n  ⚠️  Chunk ${chunkIndex + 1} failed, retrying... (${retries + 1}/${MAX_RETRIES})`);
      await new Promise((resolve) => setTimeout(resolve, 1000 * (retries + 1))); // Exponential backoff
      return processChunk(chunk, chunkIndex, totalChunks, retries + 1, useStreaming);
    }

    return {
      chunkIndex,
      result: {
        keyTakeaway: '',
        companies: [],
        concepts: { business: [], technical: [] },
        quotes: [],
        summary: `[Extraction failed: ${error.message}]`,
      },
      success: false,
      error: error.message,
      processingTime,
    };
  }
}

/**
 * Fallback summarization for failed chunks using map/reduce pattern
 */
async function summarizeFailedChunk(chunk: string): Promise<string> {
  try {
    const result = await streamText({
      model: openai('gpt-4o-mini'),
      prompt: `Provide a brief summary (2-3 sentences) of this document chunk:
${chunk}`,
    });

    let summary = '';
    for await (const text of result.textStream) {
      summary += text;
    }
    return summary;
  } catch (error: any) {
    return `[Summary generation failed: ${error.message || error}]`;
  }
}

/**
 * Reduce/aggregate chunk results into final consolidated result
 */
function reduceResults(chunkResults: ChunkResult[]): ExtractionResult {
  const successfulResults = chunkResults.filter((r) => r.success);
  
  // Aggregate companies (deduplicate)
  const allCompanies = new Set<string>();
  successfulResults.forEach((r) => {
    r.result.companies.forEach((c) => allCompanies.add(c));
  });

  // Aggregate concepts (deduplicate)
  const businessConcepts = new Set<string>();
  const technicalConcepts = new Set<string>();
  successfulResults.forEach((r) => {
    r.result.concepts.business.forEach((c) => businessConcepts.add(c));
    r.result.concepts.technical.forEach((c) => technicalConcepts.add(c));
  });

  // Aggregate quotes
  const allQuotes = successfulResults.flatMap((r) => r.result.quotes);

  // Combine summaries for key takeaway
  const summaries = successfulResults.map((r) => r.result.summary).filter(Boolean);
  const combinedSummary = summaries.join(' ');

  // Generate final key takeaway from combined summaries
  const keyTakeaway = summaries.length > 0
    ? summaries.slice(0, 3).join(' ').substring(0, 250) // Approximate 50 words
    : 'Unable to extract key takeaway from document.';

  return {
    keyTakeaway,
    companies: Array.from(allCompanies),
    concepts: {
      business: Array.from(businessConcepts),
      technical: Array.from(technicalConcepts),
    },
    quotes: allQuotes,
    summary: combinedSummary,
  };
}

/**
 * Display progress bar
 */
function displayProgress(progress: ProgressTracker): void {
  const { totalChunks, completedChunks, failedChunks, startTime } = progress;
  const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);
  const percentage = Math.round((completedChunks / totalChunks) * 100);
  const barLength = 40;
  const filled = Math.round((completedChunks / totalChunks) * barLength);
  const bar = '█'.repeat(filled) + '░'.repeat(barLength - filled);

  process.stdout.write(
    `\r📊 Progress: [${bar}] ${percentage}% | ${completedChunks}/${totalChunks} chunks | ` +
    `${failedChunks} failed | ${elapsed}s`
  );
}

/**
 * Benchmark and display metrics
 */
function displayBenchmark(progress: ProgressTracker, finalResult: ExtractionResult): void {
  const { chunkResults, startTime } = progress;
  const totalTime = (Date.now() - startTime) / 1000;
  const successfulResults = chunkResults.filter((r) => r.success);
  const avgChunkTime = successfulResults.length > 0
    ? successfulResults.reduce((sum, r) => sum + r.processingTime, 0) / successfulResults.length / 1000
    : 0;

  // Estimate memory usage (rough)
  const totalChars = chunkResults.reduce((sum, r) => sum + (r.result.summary?.length || 0), 0);
  const estimatedMemoryMB = (totalChars * 2) / (1024 * 1024); // Rough estimate

  console.log('\n\n' + '='.repeat(80));
  console.log('📈 BENCHMARK RESULTS');
  console.log('='.repeat(80));
  console.log(`⏱️  Total Processing Time: ${totalTime.toFixed(2)}s`);
  console.log(`⚡ Average Chunk Time: ${avgChunkTime.toFixed(2)}s`);
  console.log(`💾 Estimated Memory Usage: ${estimatedMemoryMB.toFixed(2)} MB`);
  console.log(`✅ Successful Chunks: ${successfulResults.length}/${chunkResults.length}`);
  console.log(`❌ Failed Chunks: ${chunkResults.filter((r) => !r.success).length}`);
  console.log(`📊 Extraction Results:`);
  console.log(`   - Companies: ${finalResult.companies.length}`);
  console.log(`   - Business Concepts: ${finalResult.concepts.business.length}`);
  console.log(`   - Technical Concepts: ${finalResult.concepts.technical.length}`);
  console.log(`   - Quotes: ${finalResult.quotes.length}`);
  console.log('='.repeat(80));
}

/**
 * Main extraction function with all features
 */
export async function extractLargeDocument(
  filePath: string,
  options: {
    chunkSize?: number;
    resume?: boolean;
    useStreaming?: boolean;
    onProgress?: (progress: ProgressTracker) => void;
  } = {}
): Promise<ExtractionResult> {
  const { chunkSize = DEFAULT_CHUNK_SIZE, resume = true, useStreaming = false, onProgress } = options;
  
  console.log('\n🚀 Starting Large Document Extraction');
  console.log('='.repeat(80));
  console.log(`📄 File: ${filePath}`);

  // Check if file exists
  if (!fs.existsSync(filePath)) {
    throw new Error(`File not found: ${filePath}`);
  }

  // Load checkpoint if resuming
  let checkpoint: Checkpoint | null = null;
  if (resume) {
    checkpoint = loadCheckpoint(filePath);
    if (checkpoint) {
      console.log(`\n🔄 Resuming from checkpoint (${checkpoint.completedChunks.length} chunks already processed)`);
    }
  }

  // Read and chunk document
  const document = fs.readFileSync(filePath, 'utf-8');
  const chunks = chunkText(document, chunkSize);
  console.log(`📦 Document split into ${chunks.length} chunks (${chunkSize} tokens/chunk)`);

  // Initialize progress tracker
  const progress: ProgressTracker = checkpoint
    ? {
        totalChunks: chunks.length,
        completedChunks: checkpoint.completedChunks.length,
        failedChunks: checkpoint.chunkResults.filter((r) => !r.success).length,
        startTime: checkpoint.timestamp,
        chunkResults: [...checkpoint.chunkResults],
      }
    : {
        totalChunks: chunks.length,
        completedChunks: 0,
        failedChunks: 0,
        startTime: Date.now(),
        chunkResults: [],
      };

  // Process chunks
  const chunksToProcess = checkpoint
    ? chunks
        .map((_, idx) => idx)
        .filter((idx) => !checkpoint!.completedChunks.includes(idx))
    : chunks.map((_, idx) => idx);

  console.log(`\n🔄 Processing ${chunksToProcess.length} remaining chunks...\n`);

  for (const chunkIndex of chunksToProcess) {
    displayProgress(progress);

    const chunkResult = await processChunk(chunks[chunkIndex], chunkIndex, chunks.length, 0, useStreaming);
    progress.chunkResults.push(chunkResult);

    if (chunkResult.success) {
      progress.completedChunks++;
    } else {
      progress.failedChunks++;
      // Apply fallback summarization
      console.log(`\n  🔄 Applying fallback summarization for chunk ${chunkIndex + 1}...`);
      const fallbackSummary = await summarizeFailedChunk(chunks[chunkIndex]);
      chunkResult.result.summary = fallbackSummary;
    }

    // Save checkpoint after each chunk
    saveCheckpoint(filePath, progress);

    // Call progress callback if provided
    if (onProgress) {
      onProgress(progress);
    }
  }

  displayProgress(progress);
  console.log('\n\n✅ All chunks processed!');

  // Reduce/aggregate results
  console.log('\n🔄 Aggregating results...');
  const finalResult = reduceResults(progress.chunkResults);

  // Display benchmark
  displayBenchmark(progress, finalResult);

  // Clear checkpoint on success
  clearCheckpoint(filePath);

  return finalResult;
}

// Main execution for testing
async function main() {
  const filePath = process.argv[2] || 'app/(1-extraction)/essay.txt';
  
  try {
    const result = await extractLargeDocument(filePath, {
      resume: true,
      onProgress: (progress) => {
        // Custom progress handler can be added here
      },
    });

    console.log('\n\n📋 FINAL EXTRACTION RESULTS');
    console.log('='.repeat(80));
    console.log('\n🎯 Key Takeaway:');
    console.log(result.keyTakeaway);
    console.log('\n🏢 Companies:');
    console.log(result.companies.length > 0 ? result.companies.join(', ') : 'None found');
    console.log('\n💼 Business Concepts:');
    console.log(result.concepts.business.length > 0 ? result.concepts.business.join(', ') : 'None found');
    console.log('\n⚙️  Technical Concepts:');
    console.log(result.concepts.technical.length > 0 ? result.concepts.technical.join(', ') : 'None found');
    console.log('\n💬 Quotes:');
    result.quotes.forEach((q, i) => {
      console.log(`  ${i + 1}. "${q.quote}" ${q.speaker ? `- ${q.speaker}` : ''}`);
    });
    console.log('\n' + '='.repeat(80));
  } catch (error: any) {
    console.error('\n❌ Extraction failed:', error.message);
    console.log('\n💡 Common issues:');
    console.log('  - Check your .env.local file has valid API keys');
    console.log('  - Verify the file path is correct');
    console.log('  - Ensure you have internet connectivity for API calls');
    process.exit(1);
  }
}

// Run if executed directly
if (require.main === module) {
  main();
}