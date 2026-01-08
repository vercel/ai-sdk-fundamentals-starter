import { generateText } from 'ai';
import 'dotenv/config';
import { recordTelemetry, getModelStats } from './lib/model-router';

const complexProblem = `
A company has 150 employees. They want to organize them into teams where:
- Each team has between 8-12 people
- No team should have exactly 10 people
- Teams should be as equal in size as possible
How should they organize the teams?
`;

// Helper function to estimate cost based on model and tokens
function estimateCost(model: string, inputTokens: number, outputTokens: number): number {
  // Cost per 1k tokens (approximate)
  const costMap: Record<string, { input: number; output: number }> = {
    'openai/gpt-5-mini': { input: 0.15, output: 0.60 },
    'openai/gpt-5': { input: 2.50, output: 10.00 },
    'openai/gpt-5.1-thinking': { input: 3.00, output: 12.00 },
    'openai/gpt-4o-mini': { input: 0.15, output: 0.60 },
  };
  
  const costs = costMap[model] || { input: 0.15, output: 0.60 };
  const inputCost = (inputTokens / 1000) * costs.input;
  const outputCost = (outputTokens / 1000) * costs.output;
  return inputCost + outputCost;
}

async function compareFastVsReasoning() {
  console.log('🚀 Starting model comparison...\n');
  console.log('Problem:', complexProblem.trim());
  console.log('\n' + '='.repeat(80) + '\n');

  // Test fast model (gpt-5.1-thinking)
  console.log('⚡ Testing fast model (gpt-5.1-thinking)...');
  const fastStartTime = Date.now();
  let fastInputTokens = 0;
  let fastOutputTokens = 0;
  let fastSuccess = true;
  
  try {
    const fastResult = await generateText({
      model: 'openai/gpt-5.1-thinking',
      prompt: complexProblem,
    });
    const fastEndTime = Date.now();
    const fastResponseTime = fastEndTime - fastStartTime;
    
    // Estimate tokens (rough approximation: ~4 chars per token)
    fastInputTokens = Math.ceil(complexProblem.length / 4);
    fastOutputTokens = Math.ceil(fastResult.text.length / 4);
    const fastCost = estimateCost('openai/gpt-5.1-thinking', fastInputTokens, fastOutputTokens);
    
    // Record telemetry
    recordTelemetry({
      model: 'openai/gpt-5.1-thinking',
      task: 'reasoning',
      latencyMs: fastResponseTime,
      inputTokens: fastInputTokens,
      outputTokens: fastOutputTokens,
      cost: fastCost,
      timestamp: Date.now(),
      success: true,
    });
    
    console.log(`⏱️  Response time: ${fastResponseTime}ms (${(fastResponseTime / 1000).toFixed(2)}s)`);
    console.log(`💰 Estimated cost: $${fastCost.toFixed(4)}`);
    console.log(`📊 Tokens: ${fastInputTokens} input + ${fastOutputTokens} output`);
    console.log(`📝 First 200 characters: ${fastResult.text.substring(0, 200)}`);
    console.log('\n' + '-'.repeat(80) + '\n');
  } catch (error) {
    fastSuccess = false;
    const fastEndTime = Date.now();
    const fastResponseTime = fastEndTime - fastStartTime;
    recordTelemetry({
      model: 'openai/gpt-5.1-thinking',
      task: 'reasoning',
      latencyMs: fastResponseTime,
      inputTokens: fastInputTokens,
      outputTokens: fastOutputTokens,
      cost: 0,
      timestamp: Date.now(),
      success: false,
    });
    console.error('❌ Error with fast model:', error);
  }

  // Test reasoning model (gpt-5-mini)
  console.log('🧠 Testing reasoning model (gpt-5-mini)...');
  const reasoningStartTime = Date.now();
  let reasoningInputTokens = 0;
  let reasoningOutputTokens = 0;
  let reasoningSuccess = true;
  
  try {
    const reasoningResult = await generateText({
      model: 'openai/gpt-5-mini',
      prompt: complexProblem,
    });
    const reasoningEndTime = Date.now();
    const reasoningResponseTime = reasoningEndTime - reasoningStartTime;
    
    // Estimate tokens
    reasoningInputTokens = Math.ceil(complexProblem.length / 4);
    reasoningOutputTokens = Math.ceil(reasoningResult.text.length / 4);
    const reasoningCost = estimateCost('openai/gpt-5-mini', reasoningInputTokens, reasoningOutputTokens);
    
    // Record telemetry
    recordTelemetry({
      model: 'openai/gpt-5-mini',
      task: 'reasoning',
      latencyMs: reasoningResponseTime,
      inputTokens: reasoningInputTokens,
      outputTokens: reasoningOutputTokens,
      cost: reasoningCost,
      timestamp: Date.now(),
      success: true,
    });
    
    console.log(`⏱️  Response time: ${reasoningResponseTime}ms (${(reasoningResponseTime / 1000).toFixed(2)}s)`);
    console.log(`💰 Estimated cost: $${reasoningCost.toFixed(4)}`);
    console.log(`📊 Tokens: ${reasoningInputTokens} input + ${reasoningOutputTokens} output`);
    console.log(`📝 First 200 characters: ${reasoningResult.text.substring(0, 200)}`);
    console.log('\n' + '-'.repeat(80) + '\n');
  } catch (error) {
    reasoningSuccess = false;
    const reasoningEndTime = Date.now();
    const reasoningResponseTime = reasoningEndTime - reasoningStartTime;
    recordTelemetry({
      model: 'openai/gpt-5-mini',
      task: 'reasoning',
      latencyMs: reasoningResponseTime,
      inputTokens: reasoningInputTokens,
      outputTokens: reasoningOutputTokens,
      cost: 0,
      timestamp: Date.now(),
      success: false,
    });
    console.error('❌ Error with reasoning model:', error);
  }

  // Compare the results and timing
  console.log('📊 Comparison Summary:');
  console.log('='.repeat(80));
  
  if (fastSuccess && reasoningSuccess) {
    const fastStats = getModelStats('openai/gpt-5.1-thinking');
    const reasoningStats = getModelStats('openai/gpt-5-mini');
    
    console.log(`⚡ Fast Model (gpt-5.1-thinking):`);
    if (fastStats) {
      console.log(`   Avg Latency: ${fastStats.avgLatencyMs.toFixed(0)}ms`);
      console.log(`   Avg Cost: $${fastStats.avgCost.toFixed(4)}`);
    }
    console.log(`🧠 Reasoning Model (gpt-5-mini):`);
    if (reasoningStats) {
      console.log(`   Avg Latency: ${reasoningStats.avgLatencyMs.toFixed(0)}ms`);
      console.log(`   Avg Cost: $${reasoningStats.avgCost.toFixed(4)}`);
    }
  }
  
  console.log('\n💡 Key Insights:');
  console.log('   - Telemetry data has been recorded for model router');
  console.log('   - Run "pnpm run model-router:demo" to see routing decisions');
  console.log('   - Check /api/model-router/stats for visualization data');
  console.log('='.repeat(80));
}

// Call the function to run your comparison
compareFastVsReasoning().catch(console.error);