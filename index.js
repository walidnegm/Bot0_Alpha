const express = require('express');
const cors = require('cors');
const fs = require('fs');
const path = require('path');
const multer = require('multer');  // Import multer for file uploads
const { exec } = require('child_process');

const app = express();
const port = 5000;

// Configure multer for file uploads
const upload = multer({ dest: 'uploads/' });

app.use(cors());  // Enable CORS for all routes
app.use(express.json({ limit: '10mb' }));  // Parse JSON request bodies and set large payload limit

// Serve the frames and transcriptions directories as static files
app.use('/frames', express.static(path.join(__dirname, 'frames')));
app.use('/transcriptions', express.static(path.join(__dirname, 'transcriptions')));

// Chat endpoint
app.post('/chat', (req, res) => {
  const userMessage = req.body.message;
  if (!userMessage) {
    return res.status(400).json({ response: 'No message provided!' });
  }
  const chatbotResponse = `You said: ${userMessage}`;
  res.json({ response: chatbotResponse });
});
 // Endpoint for transcribing audio using Whisper
app.post('/transcribe', upload.single('audio'), (req, res) => {
  const filePath = req.file.path;  // Path to the uploaded audio file
  const transcriptionsDir = path.join(__dirname, 'transcriptions');  // Ensure transcriptions directory exists

  if (!fs.existsSync(transcriptionsDir)) {
    fs.mkdirSync(transcriptionsDir);
  }

  console.log("Starting transcription for file:", filePath);  // Log when transcription starts

  // Run Whisper, outputting files to the transcriptions directory
  exec(`whisper ${filePath} --model small --output_dir ${transcriptionsDir}`, (error, stdout, stderr) => {
    if (error) {
      console.error(`Error: ${error.message}`);
      return res.status(500).json({ error: 'Transcription failed' });
    }

    const baseFileName = path.basename(filePath, path.extname(filePath));  // Get the base filename of audio file
    const transcriptionFile = path.join(transcriptionsDir, `${baseFileName}.txt`);

    // Read transcription content from the generated .txt file
    fs.readFile(transcriptionFile, 'utf8', (err, data) => {
      if (err) {
        console.error(`Error reading transcription file: ${err.message}`);
        return res.status(500).json({ error: 'Could not read transcription file' });
      }

      const transcriptText = data.trim();
      console.log("Transcription completed:", transcriptText);  // Log the completed transcription

      // Check for the wake-up word "bot"
      if (transcriptText.includes("bot")) {
        console.log("Wake-up word 'bot' detected in the transcription!");  // Log wake-up word detection
      }

      // Respond with the transcription text
      res.json({ transcript: transcriptText });

      // Cleanup: Remove the temporary audio file
      fs.unlinkSync(filePath);
    });
  });
});

// Endpoint to get video frame as a base64-encoded image
app.get('/video_frame', (req, res) => {
  fs.readFile('frame.jpg', (err, data) => {
    if (err) {
      console.error('Error reading frame:', err);
      res.status(500).json({ error: 'Could not read frame' });
    } else {
      const base64Image = Buffer.from(data).toString('base64');
      res.json({ frame: base64Image });
    }
  });
});

// Endpoint to save frames
app.post('/save_frame', (req, res) => {
  const { frame, frameNumber } = req.body;
  const buffer = Buffer.from(frame, 'base64');

  // Define the path to save frames
  const framesDir = path.join(__dirname, 'frames');
  if (!fs.existsSync(framesDir)) {
    fs.mkdirSync(framesDir);
  }

  const framePath = path.join(framesDir, `frame_${frameNumber}.jpg`);
  fs.writeFile(framePath, buffer, (err) => {
    if (err) {
      console.error('Error saving frame:', err);
      return res.status(500).json({ error: 'Failed to save frame' });
    }

    console.log(`Saved frame ${frameNumber} at path: ${framePath}`);
    res.status(200).json({ message: 'Frame saved successfully' });
  });
});

// Start the server
app.listen(port, () => {
  console.log(`Backend server is running on http://localhost:${port}`);
});
