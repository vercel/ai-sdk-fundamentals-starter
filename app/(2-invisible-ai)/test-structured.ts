import dotenvFlow from "dotenv-flow";
dotenvFlow.config();
import { generateText, Output } from "ai";
import { z } from "zod";

// Sample data for testing
const appointmentText =
  "Team meeting tomorrow 3pm in the conference room with Guillermo and Sarah";
const namesText =
  "In the meeting, Guillermo and Lee discussed the new Vercel AI SDK with Sarah from marketing.";

async function compareOutputs() {
  console.log("\n=== Sample Data ===\n");
  console.log("Appointment text:", appointmentText);
  console.log("Names text:", namesText);

  // Replace the first TODO with this:
  console.log('\n=== Using generateText (Plain Text) ===\n');
  const { text } = await generateText({
    model: 'openai/gpt-5-mini',
    prompt: `Extract all names from this text: ${namesText}`,
  });
  console.log('Raw text output:', text);
  console.log('Output type:', typeof text);
  console.log('Need to parse string to get individual names');

  // Replace the second TODO with this:
  console.log('\n=== Using generateText with Output.object() (Structured Data) ===\n');
 
  const appointmentSchema = z.object({
    title: z.string().describe('The meeting title or subject'),
    date: z.string().describe('The date of the meeting'),
    time: z.string().nullable().describe('The time of the event'),
    location: z.string().nullable().describe('Where the event will take place'),
    attendees: z.array(z.string()).nullable().describe('People attending'),
  });
 
  const { output } = await generateText({
    model: 'openai/gpt-5.2',
    prompt: `Parse appointment details from: ${appointmentText}`,
    output: Output.object({ schema: appointmentSchema }),
  });
 
  console.log('Structured output:', JSON.stringify(output, null, 2));
  console.log('Output type:', typeof output);
  console.log('\nDirect property access:');
  console.log('- Title:', output.title);
  console.log('- Date:', output.date);
  console.log('- Time:', output.time);
  console.log('- Location:', output.location);
  console.log('- Attendees:', output.attendees?.join(', '));
  console.log('- Attendees:', output.attendees);

}

compareOutputs().catch(console.error);
