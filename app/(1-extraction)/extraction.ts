import dotenvFlow from 'dotenv-flow';
dotenvFlow.config(); // Load environment variables (API keys, etc.)
import fs from 'fs';
import { generateText } from 'ai'; // AI SDK's core text generation function
 
// Read the essay file that we'll extract names from
const essay = fs.readFileSync('app/(1-extraction)/essay.txt', 'utf-8');
 
async function main() {
  // Call the LLM with our extraction prompt
  const result = await generateText({
    model: 'openai/gpt-5', // Fast, cost-effective for simple extraction tasks (non-reasoning)
                              // For complex analysis, try 'openai/gpt-5' (reasoning model, slower but more accurate)
    prompt: `What is the key takeaway of this piece in 50 words?
Essay:
${essay}`, // Instruction + the actual essay content
  });
  
  // The AI's response is in result.text
  console.log('\n--- AI Response ---');
  console.log(result.text); // This will be something like: "John Smith, Jane Doe, ..."
  console.log('-------------------');
}
 
// Run the async function and catch any errors
main().catch((error) => {
  console.error('❌ Extraction failed:', error.message);
  console.log('\n💡 Common issues:');
  console.log('  - Check your .env.local file has valid API keys');
  console.log('  - Verify essay.txt exists at app/(1-extraction)/essay.txt');
  console.log('  - Ensure you have internet connectivity for API calls');
  process.exit(1);
});