import React, { useState, useRef, useEffect } from "react";
import { startSession, stopSession, sendAudioStream } from "./realtimeClient.js"; // ✅ Ensure correct path

function App() {
  const [sessionActive, setSessionActive] = useState(false);
  const sessionActiveRef = useRef(false); // ✅ Track latest session state
  const mediaRecorderRef = useRef(null);
  const audioChunksRef = useRef([]);
  const socketRef = useRef(null);

  // ✅ Keep sessionActiveRef updated
  useEffect(() => {
      sessionActiveRef.current = sessionActive;
  }, [sessionActive]);

  useEffect(() => {
      console.log("📢 sessionActive changed:", sessionActiveRef.current);

      if (!sessionActiveRef.current) {
          console.warn("⚠️ Session is not active, skipping WebSocket connection.");
          return; // ✅ Only connect WebSocket when session is active
      }

      // ✅ Ensure WebSocket is not already connected
      if (!socketRef.current || socketRef.current.readyState !== WebSocket.OPEN) {
          console.log("🔄 Establishing WebSocket connection...");
          socketRef.current = new WebSocket("ws://localhost:5000");

          socketRef.current.onopen = () => {
              console.log("✅ WebSocket connection established");
          };

          socketRef.current.onerror = (error) => {
              console.error("❌ WebSocket error:", error);
          };

          socketRef.current.onclose = () => {
              console.log("❌ WebSocket connection closed");
              sessionActiveRef.current = false; // Ensure state remains consistent
              setSessionActive(false); // Trigger re-render if needed
          };

          socketRef.current.onmessage = async (event) => {
              if (!sessionActiveRef.current) {
                  console.warn("⚠️ Ignoring message: session is not active");
                  return;
              }

              try {
                  const message = JSON.parse(event.data);
                  if (message.type === "audio") {
                      console.log("🎧 Received audio from server, playing...");
                      const audioBuffer = new Int16Array(message.data);
                      playAudioInBrowser(audioBuffer);
                  }
              } catch (error) {
                  console.error("❌ Error processing WebSocket message:", error);
              }
          };
      } else {
          console.log("⚠️ WebSocket already open. Skipping reconnection.");
      }

      return () => {
          if (socketRef.current) {
              socketRef.current.close();
              console.log("🔌 WebSocket connection closed in cleanup.");
          }
      };
  }, [sessionActive]);


  const playAudioInBrowser = async (int16Array) => {
    try {
      const audioContext = new (window.AudioContext || window.webkitAudioContext)();
      const sampleRate = 24000;
      const buffer = audioContext.createBuffer(1, int16Array.length, sampleRate);
      const channelData = buffer.getChannelData(0);
      
      for (let i = 0; i < int16Array.length; i++) {
        channelData[i] = int16Array[i] / 32768;
      }
      
      const source = audioContext.createBufferSource();
      source.buffer = buffer;
      source.connect(audioContext.destination);
      source.start();
      
      console.log("🔊 Audio playback started.");
    } catch (error) {
      console.error("❌ Error playing audio in browser:", error);
    }
  };

  useEffect(() => {
      sessionActiveRef.current = sessionActive;
  }, [sessionActive]);
  
  const handleStartSession = async () => {
    console.log("🚀 Starting OpenAI session...");

    const success = await startSession();
    if (!success) {
        console.error("❌ Failed to start session");
        return;
    }

    setSessionActive((prev) => {
        console.log("🔄 Updating sessionActive to true, previous:", prev);
        sessionActiveRef.current = true;  // ✅ Keep ref in sync
        return true;
    });

    setTimeout(() => {
        console.log("🔄 Establishing WebSocket connection...");

        if (!socketRef.current || socketRef.current.readyState !== WebSocket.OPEN) {
            socketRef.current = new WebSocket("ws://localhost:5000");

            socketRef.current.onopen = () => {
                console.log("✅ WebSocket connection established");
                setSessionActive(true);
                sessionActiveRef.current = true; // ✅ Ensure ref is updated
                startRecording();
            };

            socketRef.current.onerror = (error) => {
                console.error("❌ WebSocket error:", error);
            };

            socketRef.current.onclose = () => {
                console.log("❌ WebSocket connection closed");
                setSessionActive(false);
                sessionActiveRef.current = false; // ❌ Ensure ref is updated
            };

            socketRef.current.onmessage = async (event) => {
                try {
                    const message = JSON.parse(event.data);
                    if (message.type === "audio") {
                        console.log("🎧 Received audio from server, playing...");
                        const audioBuffer = new Int16Array(message.data);
                        playAudioInBrowser(audioBuffer);
                    }
                } catch (error) {
                    console.error("❌ Error processing WebSocket message:", error);
                }
            };
        } else {
            console.log("⚠️ WebSocket already open. Skipping reconnection.");
            setSessionActive(true);
            sessionActiveRef.current = true; // ✅ Ensure ref is updated
            startRecording();
        }
    }, 500);
};

  


  

  const handleStopSession = async () => {
    const success = await stopSession();
    if (success) {
      setSessionActive(false);
      console.log("🚦 OpenAI session stopped");
  
      // ✅ Close WebSocket when session stops
      if (socketRef.current) {
        socketRef.current.close();
        socketRef.current = null;
      }
  
      stopRecording();
    } else {
      console.error("❌ Failed to stop session");
    }
  };
  

  const startRecording = async () => {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        const mediaRecorder = new MediaRecorder(stream, { mimeType: "audio/webm" });

        mediaRecorderRef.current = mediaRecorder;
        audioChunksRef.current = [];

        mediaRecorder.ondataavailable = (event) => {
            if (event.data.size > 0) {
                console.log("🎤 Captured audio chunk:", event.data);

                // ✅ Log the size and type
                console.log("🔍 Audio Chunk Size:", event.data.size, "Type:", event.data.type);

                audioChunksRef.current.push(event.data);

                // ✅ Use sessionActiveRef to check the latest state
                if (!sessionActiveRef.current) {
                    console.warn("⚠️ Session is not active. Skipping audio send.");
                    return;
                }

                sendAudioStream(event.data, socketRef.current, sessionActiveRef);
            } else {
                console.warn("⚠️ Captured an empty audio chunk.");
            }
        };

        mediaRecorder.start(100); // ✅ Ensure audio chunks are captured every 100ms
    } catch (error) {
        console.error("❌ Error accessing microphone:", error);
    }
};

  

  const stopRecording = () => {
    if (mediaRecorderRef.current) {
      mediaRecorderRef.current.stop();
    }
  };

  return (
    <div style={{ textAlign: "center", marginTop: "50px" }}>
      <h1>Real-Time Voice Chat with OpenAI</h1>

      <button onClick={handleStartSession} disabled={sessionActive}>
        Start Session
      </button>

      <button onClick={handleStopSession} disabled={!sessionActive}>
        Stop Session
      </button>
    </div>
  );
}

export default App;
