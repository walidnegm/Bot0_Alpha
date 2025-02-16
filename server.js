import { WebSocketServer } from 'ws';
import http from 'http';
import express from 'express';
import cors from 'cors';
import { RealtimeClient } from '@openai/realtime-api-beta';
import dotenv from 'dotenv';
import fs from 'fs'
import multer from "multer";
dotenv.config();
import path from "path";
import { fileURLToPath } from "url";
const app = express();
const port = 5000;
const server = http.createServer(app);
const wss = new WebSocketServer({ server });
const audioWriteStream = fs.createWriteStream('received_audio.wav'); // File to store the audio

app.use(cors({
  origin: "http://localhost:4000",
  methods: "GET,HEAD,PUT,PATCH,POST,DELETE",
  allowedHeaders: "Content-Type,Authorization"
}));

app.use((req, res, next) => {
  res.header("Access-Control-Allow-Origin", "*");
  res.header("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS");
  res.header("Access-Control-Allow-Headers", "Content-Type, Authorization");
  next();
});

app.options("*", (req, res) => {
  res.sendStatus(200);
});

const API_KEY = process.env.OPENAI_API_KEY;
if (!API_KEY) {
  console.error('❌ Please set your OPENAI_API_KEY in your environment variables.');
  process.exit(1);
}

const client = new RealtimeClient({
  apiKey: API_KEY,
  model: 'gpt-4o-realtime-preview-2024-10-01',
});

const silenceTimeout = 5000;
let lastAudioReceived = Date.now();

setInterval(() => {
    if (!sessionActive) return; // ✅ Do NOT trigger OpenAI response if session isn't started
  
    if (isRealtimeConnected && Date.now() - lastAudioReceived > silenceTimeout) {
      console.log("🔕 No audio received, triggering OpenAI response...");
      try {
        client.createResponse();
        lastAudioReceived = Date.now();
      } catch (error) {
        console.error("❌ Error triggering OpenAI response:", error);
      }
    }
  }, 1000);
  

const testConnection = async () => {
  console.log("🔄 Testing OpenAI Realtime API Connection...");

  try {
    await client.connect();
    console.log("✅ Successfully connected to OpenAI Realtime API.");
    isRealtimeConnected = true;
  } catch (error) {
    console.error("❌ OpenAI Realtime API Connection Failed:", error);
  }
};

testConnection();

client.on('disconnect', async () => {
    console.error("❌ OpenAI Realtime API Disconnected!");
    
    if (!isRealtimeConnected) {
      console.warn("⚠️ Already attempting to reconnect...");
      return;
    }
  
    isRealtimeConnected = false;
  
    // 🔄 Retry connection after 3 seconds
    setTimeout(async () => {
      console.log("🔄 Reconnecting to OpenAI Realtime API...");
      await connectToRealtimeAPI();
    }, 3000);
  });
  

const activeConnections = new Set();
let sessionActive = false;
let isRealtimeConnected = false;

const broadcastAudio = (audioData) => {
    console.log("📢 Broadcasting OpenAI audio response to all clients...");
  
    activeConnections.forEach((ws) => {
      if (ws.readyState === ws.OPEN) {
        ws.send(JSON.stringify({ 
          type: "audio", 
          data: Array.from(audioData) // Convert Int16Array to a regular array
        }));
      }
    });
  };

client.on('conversation.item.completed', ({ item }) => {
  console.log('📩 Received response from OpenAI:', item);

  if (item.type === 'message' && item.role === 'assistant' && item.formatted && item.formatted.audio) {
    console.log('🎧 Received assistant audio response.');
    broadcastAudio(item.formatted.audio);
  } else {
    console.log('🛑 No audio in response.');
  }
});

let reconnectAttempts = 0;
const maxReconnectAttempts = 5;

const connectToRealtimeAPI = async () => {
  if (isRealtimeConnected) {
    console.log("✅ Already connected to OpenAI Realtime API. No need to reconnect.");
    return;
  }

  if (reconnectAttempts >= maxReconnectAttempts) {
    console.error("🚨 Max reconnection attempts reached. Stopping retries.");
    return;
  }

  console.log(`🔄 Attempting to reconnect to OpenAI Realtime API... (Attempt ${reconnectAttempts + 1}/${maxReconnectAttempts})`);
  reconnectAttempts++;

  try {
    await client.connect();
    isRealtimeConnected = true;
    reconnectAttempts = 0; // ✅ Reset on successful connection
    console.log("✅ Successfully reconnected to OpenAI Realtime API.");
  } catch (error) {
    console.error(`❌ Reconnection attempt ${reconnectAttempts} failed:`, error);
    isRealtimeConnected = false;

    if (reconnectAttempts < maxReconnectAttempts) {
      const retryDelay = Math.min(5000 * reconnectAttempts, 30000); // Exponential backoff (max 30s)
      console.warn(`🔁 Retrying in ${retryDelay / 1000} seconds...`);
      setTimeout(connectToRealtimeAPI, retryDelay);
    } else {
      console.error("🚨 Max reconnection attempts reached. No more retries.");
    }
  }
};


wss.on('connection', async (ws) => {
    console.log('✅ WebSocket client connected');
  
    if (!sessionActive) {
      console.log('❌ Rejecting connection: No active session');
      ws.close();
      return;
    }
  
    activeConnections.add(ws);
    let audioBuffer = Buffer.alloc(0);
    const chunkSize = 4800;
  
    // ✅ Create write streams for debugging audio (before and after OpenAI)
    const priorAudioFilePath = 'received_audio_prior.wav';
    const postAudioFilePath = 'received_audio_post.wav';
  
    const priorAudioStream = fs.createWriteStream(priorAudioFilePath);
    const postAudioStream = fs.createWriteStream(postAudioFilePath);
  
    console.log(`💾 Saving received audio to: ${priorAudioFilePath}`);
    console.log(`💾 Saving processed audio to: ${postAudioFilePath}`);
  
    // ✅ Auto-Reconnect OpenAI API if needed
    if (!isRealtimeConnected) {
      console.warn("⚠️ OpenAI Realtime API not connected. Attempting to reconnect...");
      await connectToRealtimeAPI();
  
      if (!isRealtimeConnected) {
        console.error("❌ Failed to reconnect OpenAI API. Dropping connection.");
        ws.close();
        return;
      }
    }
  
    ws.on('message', async (audioData) => {
        if (!audioData || audioData.length === 0) return;
    
        console.log(`🎤 Received audio chunk (${audioData.length} bytes)`);
        
        // 🔍 Log first few bytes to inspect the data
        const firstBytes = new Uint8Array(audioData.buffer, 0, 10);
        console.log(`🎤 First 10 bytes of received audio:`, firstBytes);
    
        // ✅ Save raw received audio before processing
        priorAudioStream.write(audioData);
    
        lastAudioReceived = Date.now();
    
        if (!isRealtimeConnected) {
            console.warn("⚠️ OpenAI Realtime API not connected. Trying to reconnect...");
            await connectToRealtimeAPI();
            
            if (!isRealtimeConnected) {
                console.error("❌ OpenAI Realtime API is still not connected. Dropping audio chunk.");
                return;
            }
        }
    
        try {
            audioBuffer = Buffer.concat([audioBuffer, audioData]);
    
            while (audioBuffer.length >= chunkSize) {
                const chunk = audioBuffer.slice(0, chunkSize);
                audioBuffer = audioBuffer.slice(chunkSize);
    
                const int16Array = new Int16Array(chunk.buffer, chunk.byteOffset, chunk.length / 2);
    
                // ✅ Save processed audio before sending to OpenAI
                postAudioStream.write(chunk);
    
                if (isRealtimeConnected) {
                    await client.appendInputAudio(int16Array);
                    console.log('📤 Sent audio chunk to OpenAI Realtime API.');
                } else {
                    console.error("⚠️ Skipping audio send: Realtime API not connected.");
                }
            }
        } catch (error) {
            console.error('❌ Error processing and sending audio data:', error);
        }
    });    
  
    ws.on('close', () => {
      console.log('❌ Client disconnected');
      activeConnections.delete(ws);
  
      if (activeConnections.size === 0) {
        console.log('⚠️ All clients disconnected, triggering final OpenAI response...');
        try {
          client.createResponse();
        } catch (error) {
          console.error("❌ Error triggering final OpenAI response:", error);
        }
      }
  
      // ✅ Ensure audio files are saved properly
      priorAudioStream.end(() => {
        console.log(`💾 Audio file ${priorAudioFilePath} successfully saved.`);
      });
  
      postAudioStream.end(() => {
        console.log(`💾 Audio file ${postAudioFilePath} successfully saved.`);
      });
    });
  });
  
  
  // Manually define __dirname in ES Modules
  const __filename = fileURLToPath(import.meta.url);
  const __dirname = path.dirname(__filename);
  
  // Ensure "uploads" directory exists
  const uploadDir = path.join(__dirname, "uploads");
  if (!fs.existsSync(uploadDir)) {
      fs.mkdirSync(uploadDir, { recursive: true });
  }
  
  // ✅ Multer configuration (Temporary storage)
  const upload = multer({ dest: uploadDir });
  
  // Serve uploaded files
  app.use("/uploads", express.static(uploadDir));
    
  // ✅ Updated `/start-session` Endpoint (Deletes old file)
  app.post('/start-session', async (req, res) => {
      if (sessionActive) {
          return res.status(400).json({ error: '⚠️ Session already started' });
      }
  
      console.log('🚀 Starting new session...');
      sessionActive = true;
  
      // ✅ Ensure old audio file is removed before starting fresh session
      const targetPath = path.join(uploadDir, "received_audio.webm");
      if (fs.existsSync(targetPath)) {
          try {
              await fs.promises.unlink(targetPath);
              console.log("🗑️ Previous received_audio.webm file removed.");
          } catch (err) {
              console.error("❌ Error removing old audio file:", err);
          }
      }
  
      await connectToRealtimeAPI();
      res.json({ success: true, message: 'Session started' });
  });
  




//above is true

app.post("/save-audio", upload.single("audio"), async (req, res) => {
    if (!req.file) {
        console.error("❌ No file uploaded.");
        return res.status(400).send("No file uploaded.");
    }

    console.log("📂 Uploaded File Details:", req.file);

    // ✅ Ensure uploaded file is WebM
    if (req.file.mimetype !== "audio/webm") {
        console.error("❌ Incorrect file type:", req.file.mimetype);
        return res.status(400).send("Invalid file type. Expected audio/webm.");
    }

    const tempPath = req.file.path;
    const targetPath = path.join(uploadDir, "received_audio.webm");

    try {
        const data = await fs.promises.readFile(tempPath);

        // ✅ Check if existing file is valid WebM
        if (fs.existsSync(targetPath)) {
            const isValid = await isValidWebM(targetPath);
            if (!isValid) {
                console.warn("⚠️ Corrupted WebM file detected. Removing old file.");
                await fs.promises.unlink(targetPath);
            }
        }

        // ✅ Append only if file is valid WebM
        await fs.promises.appendFile(targetPath, data);
        console.log("✅ Audio chunk appended to:", targetPath);

        // ✅ Delete temp file after appending
        await fs.promises.unlink(tempPath);

        res.status(200).json({ success: true, filePath: targetPath });
    } catch (err) {
        console.error("❌ Error appending audio file:", err);
        res.status(500).send("Error saving file.");
    }
});

/**
 * ✅ Check if a file is a valid WebM by verifying its magic number
 */
const isValidWebM = async (filePath) => {
    try {
        const buffer = Buffer.alloc(4);
        const fd = await fs.promises.open(filePath, "r");
        await fd.read(buffer, 0, 4, 0);
        await fd.close();

        // WebM header should start with: 1A 45 DF A3
        return buffer.equals(Buffer.from([0x1A, 0x45, 0xDF, 0xA3]));
    } catch (err) {
        console.error("❌ Error checking WebM file:", err);
        return false;
    }
};



app.post('/stop-session', (req, res) => {
    if (!sessionActive) {
      return res.status(400).json({ error: '⚠️ No active session to stop' });
    }
  
    console.log('🛑 Closing WebSocket connections.');
    activeConnections.forEach((ws) => ws.close());
    activeConnections.clear();
    sessionActive = false;
  
    // ✅ Only trigger final OpenAI response if still connected
    if (isRealtimeConnected) {
      try {
        console.log("⚠️ Triggering final OpenAI response before disconnecting...");
        client.createResponse(); // Ensure last response is handled
      } catch (error) {
        console.error("❌ Error triggering final OpenAI response:", error);
      }
    }
  
    // ✅ Ensure API is disconnected *after* the response
    setTimeout(() => {
      if (isRealtimeConnected) {
        client.disconnect();
        isRealtimeConnected = false;
        console.log("🔌 Disconnected from OpenAI Realtime API");
      }
    }, 1000); // Small delay to allow the response to complete
  
    res.json({ success: true, message: 'Session stopped' });
  });
  

server.listen(port, () => {
  console.log(`🚀 Server is listening on port ${port}`);
});
