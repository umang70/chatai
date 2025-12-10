import Groq from "groq-sdk";
import dotenv from "dotenv";
import { tavily } from "@tavily/core";
import NodeCache from "node-cache";
import OpenAI from "openai";

dotenv.config();

const groq = new Groq({ apiKey: process.env.GROQ_API_KEY });
const tvly = tavily({ apiKey: process.env.TAVILY_API_KEY });

// Initialize Nebius OpenAI client for image generation
const nebius = new OpenAI({
  baseURL: "https://api.tokenfactory.nebius.com/v1/",
  apiKey: process.env.NEBIUS_API_KEY,
});

// Store conversation history per thread/user
const conversationMemory = new NodeCache({ stdTTL: 60 * 60 * 24 }); // 24 hour expiry

async function webSearch({ query }) {
  console.log("🔍 webSearch called with query:", query);
  const res = await tvly.search(query);

  if (res.answer) {
    return `Web summary:\n${res.answer}`;
  }

  const top = (res.results || []).slice(0, 3);
  const lines = top.map((r, i) => {
    return `Result ${i + 1}:
Title: ${r.title}
Snippet: ${r.content?.slice(0, 200) || ""}`;
  });

  return `Web summary (top results):\n${lines.join("\n\n")}`;
}

// 🎨 Generate image using Nebius API
async function generateImage(prompt) {
  console.log("🎨 Generating image with prompt:", prompt);
  
  try {
    const response = await nebius.images.generate({
      model: "black-forest-labs/flux-schnell",
      prompt: prompt,
      n: 1,
      size: "1024x1024",
    });

    if (response.data && response.data.length > 0) {
      return response.data[0].url;
    }
    throw new Error("No image generated");
  } catch (err) {
    console.error("❌ Image generation error:", err.message);
    throw new Error(`Image generation failed: ${err.message}`);
  }
}

async function askOnce(userQuestion, threadId) {
  // Get conversation history
  let conversationHistory = conversationMemory.get(threadId) || [];

  const baseMessages = [
    {
      role: "system",
      content: `You are a friendly, helpful AI assistant with memory of previous conversations.
Use the webSearch tool for anything that requires up-to-date or factual information.  
Do not guess or make up facts.  
Speak casually but stay accurate.  
If the user asks something personal, creative, or opinion-based, reply normally without web search.  
If the user asks about real-world events, companies, technology, dates, launches, or news, ALWAYS perform a web search first for recent information.`,
    },
    ...conversationHistory,
    {
      role: "user",
      content: userQuestion,
    },
  ];

  // 1️⃣ First call – let LLM decide if it needs webSearch
  const first = await groq.chat.completions.create({
    model: "llama-3.1-8b-instant",
    temperature: 0,
    messages: baseMessages,
    tools: [
      {
        type: "function",
        function: {
          name: "webSearch",
          description: "Search the recent info on the internet",
          parameters: {
            type: "object",
            properties: {
              query: {
                type: "string",
                description: "Search query, e.g. 'price of iPhone 17'",
              },
            },
            required: ["query"],
          },
        },
      },
    ],
    tool_choice: "auto",
  });

  // ✅ Safe null check
  const toolCalls = first.choices?.[0]?.message?.tool_calls;

  // If LLM didn't request a tool → just answer normally
  if (!toolCalls || toolCalls.length === 0) {
    const answer = first.choices[0].message.content;
    
    // Save to memory
    conversationHistory.push({ role: "user", content: userQuestion });
    conversationHistory.push({ role: "assistant", content: answer });
    conversationMemory.set(threadId, conversationHistory);
    
    return answer;
  }

  const toolCall = toolCalls[0];
  const fnName = toolCall.function.name;
  
  // ✅ Safe JSON parse with error handling
  let fnArgs;
  try {
    fnArgs = JSON.parse(toolCall.function.arguments);
  } catch (err) {
    console.error("❌ Failed to parse tool arguments:", err);
    throw new Error("Invalid tool arguments from LLM");
  }

  let toolResultText = "";

  if (fnName === "webSearch") {
    toolResultText = await webSearch(fnArgs);
  }

  // 2️⃣ Second call – use web result to answer nicely
  const secondMessages = [
    {
      role: "system",
      content: `You have access to web search results via the tool output.
Use them to answer the user's question accurately.
Do NOT call webSearch again.
If data is unclear, say so. Keep your answer casual, short, and clear.`,
    },
    ...conversationHistory,
    {
      role: "user",
      content: userQuestion,
    },
    {
      role: "assistant",
      tool_calls: [toolCall],
    },
    {
      role: "tool",
      tool_call_id: toolCall.id,
      name: fnName,
      content: toolResultText,
    },
  ];

  const second = await groq.chat.completions.create({
    model: "llama-3.1-8b-instant",
    temperature: 0,
    messages: secondMessages,
  });

  const answer = second.choices[0].message.content;
  
  // Save to memory
  conversationHistory.push({ role: "user", content: userQuestion });
  conversationHistory.push({ role: "assistant", content: answer });
  conversationMemory.set(threadId, conversationHistory);

  return answer;
}

// 🔥 Generate text response
export async function generate(userMessage, threadId = "default") {
  if (!userMessage || !userMessage.trim()) {
    throw new Error("userMessage is empty");
  }

  const answer = await askOnce(userMessage.trim(), threadId);
  return answer;
}

// 🎨 Generate image
export async function generateImageResponse(prompt) {
  if (!prompt || !prompt.trim()) {
    throw new Error("Prompt is empty");
  }

  const imageUrl = await generateImage(prompt.trim());
  return imageUrl;
}

export function clearMemory(threadId) {
  conversationMemory.del(threadId);
}
