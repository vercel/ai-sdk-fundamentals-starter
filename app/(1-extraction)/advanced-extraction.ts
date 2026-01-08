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

  const companyResult = await generateText({
    model: 'openai/gpt-5', // Fast, cost-effective for simple extraction tasks (non-reasoning)
                              // For complex analysis, try 'openai/gpt-5' (reasoning model, slower but more accurate)
    prompt: `Extract all company names from this essay.
Include both explicit mentions and implied references (e.g., "the startup" referring to a previously mentioned company).
 
Format as JSON array: ["Company 1", "Company 2"]
 
Essay: ${essay}`, // Instruction + the actual essay content
  });

  const conceptsResult = await generateText({
    model: 'openai/gpt-5', // Fast, cost-effective for simple extraction tasks (non-reasoning)
                              // For complex analysis, try 'openai/gpt-5' (reasoning model, slower but more accurate)
    prompt: `Identify the main business concepts and technical terms in this essay.
Categorize them as either 'business' or 'technical' concepts.
 
Format as JSON:
{
  "business": ["concept1", "concept2"],
  "technical": ["term1", "term2"]
}
 
Essay: ${essay}`, // Instruction + the actual essay content
  });

  const quotePrompt = await generateText({
    model: 'openai/gpt-5', // Fast, cost-effective for simple extraction tasks (non-reasoning)
                              // For complex analysis, try 'openai/gpt-5' (reasoning model, slower but more accurate)
    prompt: `Extract all quotes (text in quotation marks) from this essay.
    For each quote, identify who said it if mentioned.
     
    Format as JSON array:
    [{"quote": "text here", "speaker": "name or null"}]
     
    Essay: ${essay}`, // Instruction + the actual essay content
  });
  
  // The AI's response is in result.text
  console.log('\n--- AI Response ---');
  console.log(result.text);
  console.log(companyResult.text); // This will be something like: "John Smith, Jane Doe, ..."
  console.log(conceptsResult.text);
  console.log(quotePrompt.text);
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