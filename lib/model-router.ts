import { z } from 'zod';

// Router configuration schema
export const RouterConfigSchema = z.object({
  task: z.enum(['classification', 'summarization', 'reasoning', 'extraction']),
  maxLatencyMs: z.number().min(0),
  priority: z.enum(['cost', 'quality', 'balanced']),
  estimatedTokens: z.number().optional(), // Optional: helps with cost calculation
});

export type RouterConfig = z.infer<typeof RouterConfigSchema>;

// Model capability tiers
export type CapabilityTier = 'basic' | 'standard' | 'advanced' | 'reasoning';

// Model metadata interface
export interface ModelMetadata {
  provider: string;
  model: string;
  fullName: string; // e.g., 'openai/gpt-5-mini'
  baseCostPer1kTokens: number; // Input cost per 1k tokens
  baseCostPer1kOutputTokens: number; // Output cost per 1k tokens
  avgLatencyMs: number; // Average latency in milliseconds
  capabilityTier: CapabilityTier;
  supportsStreaming: boolean;
  maxTokens: number;
  // Task-specific capabilities
  taskCapabilities: {
    classification: number; // 0-1 score
    summarization: number;
    reasoning: number;
    extraction: number;
  };
}

// Telemetry data for a single call
export interface CallTelemetry {
  model: string;
  task: string;
  latencyMs: number;
  inputTokens: number;
  outputTokens: number;
  cost: number;
  timestamp: number;
  success: boolean;
}

// In-memory telemetry store (in production, use a database)
const telemetryStore: CallTelemetry[] = [];
const MAX_TELEMETRY_ENTRIES = 1000;

// Model registry with static metadata
const MODEL_REGISTRY: ModelMetadata[] = [
  {
    provider: 'openai',
    model: 'gpt-5-mini',
    fullName: 'openai/gpt-5-mini',
    baseCostPer1kTokens: 0.15, // $0.15 per 1k input tokens
    baseCostPer1kOutputTokens: 0.60, // $0.60 per 1k output tokens
    avgLatencyMs: 500,
    capabilityTier: 'standard',
    supportsStreaming: true,
    maxTokens: 128000,
    taskCapabilities: {
      classification: 0.9,
      summarization: 0.85,
      reasoning: 0.6,
      extraction: 0.9,
    },
  },
  {
    provider: 'openai',
    model: 'gpt-5',
    fullName: 'openai/gpt-5',
    baseCostPer1kTokens: 2.50, // $2.50 per 1k input tokens
    baseCostPer1kOutputTokens: 10.00, // $10.00 per 1k output tokens
    avgLatencyMs: 1500,
    capabilityTier: 'advanced',
    supportsStreaming: true,
    maxTokens: 128000,
    taskCapabilities: {
      classification: 0.95,
      summarization: 0.95,
      reasoning: 0.85,
      extraction: 0.95,
    },
  },
  {
    provider: 'openai',
    model: 'gpt-5.1-thinking',
    fullName: 'openai/gpt-5.1-thinking',
    baseCostPer1kTokens: 3.00,
    baseCostPer1kOutputTokens: 12.00,
    avgLatencyMs: 5000, // Reasoning models are slower
    capabilityTier: 'reasoning',
    supportsStreaming: false,
    maxTokens: 128000,
    taskCapabilities: {
      classification: 0.9,
      summarization: 0.85,
      reasoning: 0.98, // Excellent for reasoning
      extraction: 0.9,
    },
  },
  {
    provider: 'openai',
    model: 'gpt-4o-mini',
    fullName: 'openai/gpt-4o-mini',
    baseCostPer1kTokens: 0.15,
    baseCostPer1kOutputTokens: 0.60,
    avgLatencyMs: 400,
    capabilityTier: 'basic',
    supportsStreaming: true,
    maxTokens: 128000,
    taskCapabilities: {
      classification: 0.85,
      summarization: 0.8,
      reasoning: 0.5,
      extraction: 0.85,
    },
  },
];

/**
 * Record telemetry data for a model call
 */
export function recordTelemetry(telemetry: CallTelemetry): void {
  telemetryStore.push(telemetry);
  
  // Keep only the last MAX_TELEMETRY_ENTRIES entries
  if (telemetryStore.length > MAX_TELEMETRY_ENTRIES) {
    telemetryStore.shift();
  }
}

/**
 * Get telemetry data for a specific model
 */
export function getModelTelemetry(model: string, limit: number = 100): CallTelemetry[] {
  return telemetryStore
    .filter(t => t.model === model)
    .slice(-limit)
    .reverse();
}

/**
 * Get aggregated statistics for a model
 */
export function getModelStats(model: string) {
  const modelTelemetry = getModelTelemetry(model);
  
  if (modelTelemetry.length === 0) {
    return null;
  }

  const latencies = modelTelemetry.map(t => t.latencyMs);
  const costs = modelTelemetry.map(t => t.cost);
  
  return {
    model,
    callCount: modelTelemetry.length,
    avgLatencyMs: latencies.reduce((a, b) => a + b, 0) / latencies.length,
    minLatencyMs: Math.min(...latencies),
    maxLatencyMs: Math.max(...latencies),
    avgCost: costs.reduce((a, b) => a + b, 0) / costs.length,
    totalCost: costs.reduce((a, b) => a + b, 0),
    successRate: modelTelemetry.filter(t => t.success).length / modelTelemetry.length,
  };
}

/**
 * Get all telemetry data (for stats API)
 */
export function getAllTelemetry(limit: number = 100): CallTelemetry[] {
  return telemetryStore.slice(-limit).reverse();
}

/**
 * Calculate estimated cost for a model call
 */
function estimateCost(
  model: ModelMetadata,
  inputTokens: number = 1000,
  outputTokens: number = 500
): number {
  const inputCost = (inputTokens / 1000) * model.baseCostPer1kTokens;
  const outputCost = (outputTokens / 1000) * model.baseCostPer1kOutputTokens;
  return inputCost + outputCost;
}

/**
 * Get current latency for a model from telemetry (or fallback to static)
 */
function getCurrentLatency(model: ModelMetadata): number {
  const stats = getModelStats(model.fullName);
  if (stats && stats.callCount > 5) {
    // Use telemetry if we have enough data
    return stats.avgLatencyMs;
  }
  // Fallback to static average
  return model.avgLatencyMs;
}

/**
 * Score a model for a given task configuration
 */
function scoreModel(
  model: ModelMetadata,
  config: RouterConfig,
  estimatedInputTokens: number = 1000,
  estimatedOutputTokens: number = 500
): number {
  const task = config.task;
  const capability = model.taskCapabilities[task];
  const latency = getCurrentLatency(model);
  const cost = estimateCost(model, estimatedInputTokens, estimatedOutputTokens);
  
  // Normalize scores (0-1 range)
  const capabilityScore = capability;
  const latencyScore = Math.max(0, 1 - (latency / config.maxLatencyMs)); // Better if faster
  const costScore = 1 / (1 + cost * 100); // Lower cost = higher score
  
  // Weight factors based on priority
  let weights: { capability: number; latency: number; cost: number };
  
  if (config.priority === 'cost') {
    weights = { capability: 0.3, latency: 0.2, cost: 0.5 };
  } else if (config.priority === 'quality') {
    weights = { capability: 0.6, latency: 0.2, cost: 0.2 };
  } else {
    // balanced
    weights = { capability: 0.4, latency: 0.3, cost: 0.3 };
  }
  
  // Calculate weighted score
  const score =
    capabilityScore * weights.capability +
    latencyScore * weights.latency +
    costScore * weights.cost;
  
  return score;
}

/**
 * Select the optimal model for a given task configuration
 */
export function selectModel(config: RouterConfig): string {
  const validatedConfig = RouterConfigSchema.parse(config);
  
  const estimatedInputTokens = validatedConfig.estimatedTokens || 1000;
  const estimatedOutputTokens = Math.floor(estimatedInputTokens * 0.5);
  
  // Filter models that meet latency requirements
  const viableModels = MODEL_REGISTRY.filter(model => {
    const latency = getCurrentLatency(model);
    return latency <= validatedConfig.maxLatencyMs;
  });
  
  if (viableModels.length === 0) {
    // If no model meets latency requirements, use the fastest one
    const fastestModel = MODEL_REGISTRY.reduce((prev, curr) =>
      getCurrentLatency(curr) < getCurrentLatency(prev) ? curr : prev
    );
    console.warn(
      `⚠️  No model meets latency requirement (${validatedConfig.maxLatencyMs}ms). ` +
      `Using fastest available: ${fastestModel.fullName}`
    );
    return fastestModel.fullName;
  }
  
  // Score all viable models
  const scoredModels = viableModels.map(model => ({
    model,
    score: scoreModel(model, validatedConfig, estimatedInputTokens, estimatedOutputTokens),
  }));
  
  // Sort by score (highest first)
  scoredModels.sort((a, b) => b.score - a.score);
  
  const selected = scoredModels[0].model;
  
  return selected.fullName;
}

/**
 * Get routing statistics for visualization
 */
export function getRoutingStats() {
  const allTelemetry = getAllTelemetry(100);
  
  // Group by model
  const modelGroups = new Map<string, CallTelemetry[]>();
  for (const entry of allTelemetry) {
    if (!modelGroups.has(entry.model)) {
      modelGroups.set(entry.model, []);
    }
    modelGroups.get(entry.model)!.push(entry);
  }
  
  // Group by task
  const taskGroups = new Map<string, CallTelemetry[]>();
  for (const entry of allTelemetry) {
    if (!taskGroups.has(entry.task)) {
      taskGroups.set(entry.task, []);
    }
    taskGroups.get(entry.task)!.push(entry);
  }
  
  // Calculate statistics
  const modelStats = Array.from(modelGroups.entries()).map(([model, entries]) => {
    const latencies = entries.map(e => e.latencyMs);
    const costs = entries.map(e => e.cost);
    
    return {
      model,
      callCount: entries.length,
      avgLatencyMs: latencies.reduce((a, b) => a + b, 0) / latencies.length,
      totalCost: costs.reduce((a, b) => a + b, 0),
      successRate: entries.filter(e => e.success).length / entries.length,
    };
  });
  
  const taskStats = Array.from(taskGroups.entries()).map(([task, entries]) => {
    const modelDistribution = new Map<string, number>();
    for (const entry of entries) {
      modelDistribution.set(
        entry.model,
        (modelDistribution.get(entry.model) || 0) + 1
      );
    }
    
    return {
      task,
      callCount: entries.length,
      modelDistribution: Object.fromEntries(modelDistribution),
    };
  });
  
  return {
    totalCalls: allTelemetry.length,
    modelStats,
    taskStats,
    recentCalls: allTelemetry.slice(0, 20), // Last 20 calls
  };
}

// Demo/test function - runs when file is executed directly
function runDemo() {
  console.log('🧠 Model Router Demo\n');
  console.log('='.repeat(80));
  
  // Test case 1: Cost-optimized classification
  console.log('\n📋 Test 1: Cost-optimized classification');
  const config1: RouterConfig = {
    task: 'classification',
    maxLatencyMs: 2000,
    priority: 'cost',
  };
  const model1 = selectModel(config1);
  console.log(`   Config: ${JSON.stringify(config1)}`);
  console.log(`   Selected: ${model1}`);
  
  // Test case 2: Quality-optimized reasoning
  console.log('\n🧠 Test 2: Quality-optimized reasoning');
  const config2: RouterConfig = {
    task: 'reasoning',
    maxLatencyMs: 10000,
    priority: 'quality',
  };
  const model2 = selectModel(config2);
  console.log(`   Config: ${JSON.stringify(config2)}`);
  console.log(`   Selected: ${model2}`);
  
  // Test case 3: Balanced summarization
  console.log('\n📝 Test 3: Balanced summarization');
  const config3: RouterConfig = {
    task: 'summarization',
    maxLatencyMs: 3000,
    priority: 'balanced',
  };
  const model3 = selectModel(config3);
  console.log(`   Config: ${JSON.stringify(config3)}`);
  console.log(`   Selected: ${model3}`);
  
  // Test case 4: Fast extraction
  console.log('\n🔍 Test 4: Fast extraction (low latency)');
  const config4: RouterConfig = {
    task: 'extraction',
    maxLatencyMs: 1000,
    priority: 'balanced',
  };
  const model4 = selectModel(config4);
  console.log(`   Config: ${JSON.stringify(config4)}`);
  console.log(`   Selected: ${model4}`);
  
  console.log('\n' + '='.repeat(80));
  console.log('\n✅ Demo complete!');
}

// Run demo when file is executed directly (works with tsx)
// @ts-ignore - require.main check for direct execution
if (typeof require !== 'undefined' && require.main === module) {
  runDemo();
}
