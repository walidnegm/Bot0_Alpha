const express = require('express');
const cors = require('cors');
const fs = require('fs');
const path = require('path');

const app = express();
const port = 5000;

app.use(cors());  // Enable CORS for all routes
app.use(express.json({ limit: '10mb' }));  // Parse JSON request bodies and set large payload limit

// Serve the frames directory as static files
app.use('/frames', express.static(path.join(__dirname, 'frames')));

app.post('/chat', (req, res) => {
  const userMessage = req.body.message;
  if (!userMessage) {
    return res.status(400).json({ response: 'No message provided!' });
  }
  const chatbotResponse = `You said: ${userMessage}`;
  res.json({ response: chatbotResponse });
});

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

app.listen(port, () => {
  console.log(`Backend server is running on http://localhost:${port}`);
});
