let socket = null;
let reconnectTimeout = null;

const connectWebSocket = () => {
  if (socket) {
    console.warn("⚠️ WebSocket is already connected.");
    return;
  }

  socket = new WebSocket("ws://localhost:5000");

  socket.onopen = () => {
    console.log("✅ WebSocket connection established");
    clearTimeout(reconnectTimeout);
  };

  socket.onerror = (error) => {
    console.error("❌ WebSocket error:", error);
  };

  socket.onclose = () => {
    console.log("❌ WebSocket closed");

    reconnectTimeout = setTimeout(() => {
      console.warn("⚠️ Attempting to reconnect WebSocket...");
      connectWebSocket();
    }, 3000);
  };
};

export const startSession = async () => {
  try {
    const response = await fetch("http://localhost:5000/start-session", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
    });

    const data = await response.json();
    if (data.success) {
      console.log("✅ Session started successfully!");
      connectWebSocket();
      return true;
    }
    return false;
  } catch (error) {
    console.error("❌ Error starting session:", error);
    return false;
  }
};

export const stopSession = async () => {
  try {
    const response = await fetch("http://localhost:5000/stop-session", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
    });

    const data = await response.json();
    if (data.success) {
      console.log("✅ Session stopped successfully!");

      if (socket) {
        socket.close();
        socket = null;
      }

      clearTimeout(reconnectTimeout);
      return true;
    }
    return false;
  } catch (error) {
    console.error("❌ Error stopping session:", error);
    return false;
  }
};

export const sendAudioStream = async (audioData, socket, sessionActiveRef) => {
    if (!socket || socket.readyState !== WebSocket.OPEN) {
        console.warn("⚠️ WebSocket is not open. Audio data not sent.");
        return;
    }

    if (!sessionActiveRef.current) {
        console.warn("⚠️ Session is not active. Retrying in 100ms...");
        setTimeout(() => sendAudioStream(audioData, socket, sessionActiveRef), 100);
        return;
    }

    if (!audioData || audioData.size === 0) {
        console.warn("⚠️ Captured an empty audio chunk. Skipping send.");
        return;
    }

    console.log(`📤 Sending audio chunk (${audioData.size} bytes)`);

    try {
        // ✅ Send audio to the WebSocket
        socket.send(audioData);

        // ✅ Also send the audio to the backend for saving
        const formData = new FormData();
        formData.append("audio", audioData, "audio_chunk.webm");

        await fetch("http://localhost:5000/save-audio", {
            method: "POST",
            body: formData,
        });

        console.log("✅ Audio chunk sent to backend for saving.");
    } catch (error) {
        console.error("❌ Error sending audio data:", error);
    }
};

