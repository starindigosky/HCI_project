// ASR Variables
let websocket = null;
let mediaRecorder = null;
let audioContext = null;
let audioStream = null;

const startBtn = document.getElementById('startBtn');
const stopBtn = document.getElementById('stopBtn');
const statusDiv = document.getElementById('status');
const transcriptionBox = document.getElementById('transcriptionBox');
const languageSelect = document.getElementById('language');
const interimResultsCheckbox = document.getElementById('interimResults');

// TTS Variables
const ttsBtn = document.getElementById('ttsBtn');
const ttsText = document.getElementById('ttsText');
const ttsSourceLang = document.getElementById('ttsSourceLang');
const ttsStatus = document.getElementById('ttsStatus');
const ttsAudioPlayer = document.getElementById('ttsAudioPlayer');
const historyList = document.getElementById('historyList');

// Audio Queue Variables
const audioQueue = [];
let isPlaying = false;
let isSystemSpeaking = false; // Flag to prevent ASR from hearing TTS

async function processQueue() {
    if (isPlaying || audioQueue.length === 0) return;

    isPlaying = true;
    isSystemSpeaking = true; // Mute mic
    updateStatus('🔊 Speaking...', 'speaking');

    const { player, container } = audioQueue.shift();

    try {
        container.classList.add('playing'); // Visual indicator
        await player.play();

        player.onended = () => {
            container.classList.remove('playing');
            isPlaying = false;

            // Only unmute if queue is empty (no more speech coming immediately)
            if (audioQueue.length === 0) {
                // Add a "Cooldown" period to let echo die down
                updateStatus('⏳ Cooldown...', 'speaking');
                setTimeout(() => {
                    // Check queue again in case new audio arrived during cooldown
                    if (audioQueue.length === 0) {
                        isSystemSpeaking = false;
                        if (websocket && websocket.readyState === WebSocket.OPEN) {
                            updateStatus('🔴 Recording...', 'recording');
                        }
                    }
                }, 2000); // 2.0 second safety buffer
            }

            processQueue(); // Process next item
        };

        player.onerror = (e) => {
            console.error("Audio playback error:", e);
            container.classList.remove('playing');
            isPlaying = false;
            if (audioQueue.length === 0) {
                isSystemSpeaking = false;
                if (websocket && websocket.readyState === WebSocket.OPEN) {
                    updateStatus('🔴 Recording...', 'recording');
                }
            }
            processQueue(); // Skip to next
        };

    } catch (error) {
        console.error("Audio playback failed:", error);
        container.classList.remove('playing');
        isPlaying = false;
        processQueue(); // Skip to next
    }
}

// History Functions
async function fetchHistory() {
    try {
        const response = await fetch('http://localhost:8000/api/v1/tts/history');
        if (!response.ok) throw new Error('Failed to fetch history');
        const files = await response.json();
        renderHistory(files);
    } catch (error) {
        console.error('Error fetching history:', error);
    }
}

function renderHistory(files) {
    if (!files || files.length === 0) {
        historyList.innerHTML = '<div class="empty-history">No audio generated yet.</div>';
        return;
    }

    historyList.innerHTML = files.map(file => {
        const date = new Date(file.created_at * 1000).toLocaleString();

        let displayFilename = file.filename;
        if (displayFilename.includes('_')) {
            const parts = displayFilename.split('_');
            // If the last part looks like a UUID (hex) or similar ID, remove it
            if (parts.length > 1) {
                // Heuristic: Join all parts except the last one
                displayFilename = parts.slice(0, -1).join('_');
            }
        }


        // Ensure URL is absolute pointing to backend
        let audioUrl = file.url;
        if (audioUrl.startsWith('/')) {
            audioUrl = `http://localhost:8000${audioUrl}`;
        }

        return `
            <div class="history-item">
                <div class="history-filename" title="${file.filename}">${displayFilename}</div>
                <div class="history-meta">
                    <span>${date}</span>
                    <span>${(file.size / 1024).toFixed(1)} KB</span>
                </div>
                <audio controls preload="metadata" class="history-audio" src="${audioUrl}" type="audio/wav"></audio>
            </div>
        `;
    }).join('');
}

// Load history on startup
fetchHistory();

// API URLs
const ASR_WS_URL = 'ws://localhost:8000/api/v1/ws/asr';
const TTS_API_URL = 'http://localhost:8000/api/v1/tts/synthesize';

function updateStatus(message, className) {
    statusDiv.textContent = message;
    statusDiv.className = `status ${className}`;
}

function updateTtsStatus(message, className, show) {
    ttsStatus.textContent = message;
    ttsStatus.className = `status ${className}`;
    ttsStatus.style.display = show ? 'block' : 'none';
}

async function getSynthesizedAudio(text) {
    try {
        const response = await fetch(TTS_API_URL, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                text: text,
                source_language: languageSelect.value, // Use the same language as ASR
                target_language: 'min_nan'
            }),
        });
        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || 'Synthesis failed');
        }
        const data = await response.json();
        return `http://localhost:8000${data.audio_url}`;
    } catch (error) {
        console.error('TTS Error for transcription:', error);
        return null;
    }
}

async function addTranscription(text, isFinal = true, details = {}) {
    // Remove empty state if present
    const emptyState = transcriptionBox.querySelector('.empty-state');
    if (emptyState) emptyState.remove();

    // Determine target text and styling
    const originalText = text; // Source (e.g. Mandarin)
    const translatedText = details.translated_text || text; // Target (e.g. Min Nan) or same
    const confidence = details.confidence || 'low';
    const method = details.method || 'fallback';

    // UI Class for confidence coloring
    // High confidence (dict match) -> Green-ish text/border
    // Low confidence (fallback) -> Normal/Gray-ish
    const confidenceClass = confidence === 'high' ? 'high-confidence' : 'low-confidence';

    // Find or create transcription item
    let item;
    if (!isFinal) {
        item = transcriptionBox.querySelector('.transcription-item:last-child.interim');
        if (!item) {
            item = document.createElement('div');
            item.className = 'transcription-item interim';

            const contentDiv = document.createElement('div');
            contentDiv.className = 'transcription-content';
            item.appendChild(contentDiv);

            const metaDiv = document.createElement('div');
            metaDiv.className = 'transcription-meta';
            item.appendChild(metaDiv);

            transcriptionBox.appendChild(item);
        }
    } else {
        // If it was an interim result, finalize it. Otherwise, create a new one.
        item = transcriptionBox.querySelector('.transcription-item:last-child.interim');
        if (item) {
            item.classList.remove('interim');
        } else {
            item = document.createElement('div');
            item.className = 'transcription-item';

            const contentDiv = document.createElement('div');
            contentDiv.className = 'transcription-content';
            item.appendChild(contentDiv);

            const metaDiv = document.createElement('div');
            metaDiv.className = 'transcription-meta';
            item.appendChild(metaDiv);

            transcriptionBox.appendChild(item);
        }
    }

    // Render Dual Text Layout
    const contentDiv = item.querySelector('.transcription-content');
    contentDiv.innerHTML = `
        <div class="source-text" style="font-size: 0.85em; color: #666; margin-bottom: 4px;">${originalText}</div>
        <div class="target-text ${confidenceClass}" style="font-size: 1.1em; font-weight: bold; color: ${confidence === 'high' ? '#2e7d32' : '#d84315'};">
            ${translatedText}
        </div>
        ${confidence === 'low' && isFinal ? '<div style="font-size:0.7em; color:#999;">(原文直讀)</div>' : ''}
    `;

    item.querySelector('.transcription-meta').textContent = `${isFinal ? 'Final' : 'Interim'} • ${method && method !== 'fallback' ? 'Dict Match' : 'Neural/Raw'} • ${new Date().toLocaleTimeString()}`;


    // If it's a final transcription, fetch and embed the audio
    if (isFinal) {
        const audioContainer = document.createElement('div');
        audioContainer.className = 'transcription-audio-container';
        audioContainer.textContent = 'Synthesizing audio...';
        item.appendChild(audioContainer);

        const audioUrl = await getSynthesizedAudio(translatedText); // Synthesize the TRANSLATED text
        if (audioUrl) {
            const audioPlayer = document.createElement('audio');
            audioPlayer.src = audioUrl;
            audioPlayer.type = "audio/wav";
            audioPlayer.controls = true;
            // audioPlayer.autoplay = true; // Disable direct autoplay, use queue

            audioContainer.textContent = ''; // Clear "Synthesizing..." message
            audioContainer.appendChild(audioPlayer);

            // Add to queue
            audioQueue.push({ player: audioPlayer, container: item });
            processQueue();
        } else {
            audioContainer.textContent = 'Audio synthesis failed.';
        }
    }

    transcriptionBox.scrollTop = transcriptionBox.scrollHeight;
}

async function startRecording() {
    try {
        // Connect WebSocket
        websocket = new WebSocket(ASR_WS_URL);

        websocket.onopen = async () => {
            updateStatus('Connected', 'connected');

            // Send configuration
            websocket.send(JSON.stringify({
                type: 'config',
                language: languageSelect.value,
                interim_results: interimResultsCheckbox.checked
            }));

            // Start audio capture
            audioStream = await navigator.mediaDevices.getUserMedia({ audio: true });

            // Create audio context
            audioContext = new AudioContext({ sampleRate: 16000 });
            const source = audioContext.createMediaStreamSource(audioStream);

            // Create script processor for audio chunks
            const processor = audioContext.createScriptProcessor(4096, 1, 1);

            processor.onaudioprocess = (e) => {
                // If system is speaking, ignore microphone input (prevent feedback loop)
                if (isSystemSpeaking) return;

                if (websocket && websocket.readyState === WebSocket.OPEN) {
                    const audioData = e.inputBuffer.getChannelData(0);
                    // Convert to Int16Array
                    const int16Data = new Int16Array(audioData.length);
                    for (let i = 0; i < audioData.length; i++) {
                        int16Data[i] = Math.max(-32768, Math.min(32767, audioData[i] * 32768));
                    }
                    websocket.send(int16Data.buffer);
                }
            };

            source.connect(processor);
            processor.connect(audioContext.destination);

            updateStatus('🔴 Recording...', 'recording');
            startBtn.disabled = true;
            stopBtn.disabled = false;
        };

        websocket.onmessage = (event) => {
            const message = JSON.parse(event.data);

            if (message.type === 'transcription') {
                if (message.translated_text) {
                    // Hybrid result
                    addTranscription(message.text, message.is_final, {
                        translated_text: message.translated_text,
                        method: message.method,
                        confidence: message.confidence
                    });
                } else {
                    // Legacy/Interim simple result
                    addTranscription(message.text, message.is_final);
                }
            } else if (message.type === 'error') {
                console.error('WebSocket error:', message.message);
                alert('Error: ' + message.message);
            } else if (message.type === 'stopped') {
                updateStatus('Stopped', 'disconnected');
                if (websocket) {
                    websocket.close();
                    websocket = null;
                }
                startBtn.disabled = false;
                stopBtn.disabled = true;
            }
        };

        websocket.onerror = (error) => {
            console.error('WebSocket error:', error);
            updateStatus('Error: Connection failed', 'disconnected');
        };

        websocket.onclose = () => {
            updateStatus('Disconnected', 'disconnected');
            startBtn.disabled = false;
            stopBtn.disabled = true;
        };

    } catch (error) {
        console.error('Error starting recording:', error);
        alert('Error: ' + error.message);
        updateStatus('Error: Could not access microphone', 'disconnected');
    }
}

function stopRecording(e) {
    if (e) e.preventDefault();

    try {
        // Send stop message
        if (websocket && websocket.readyState === WebSocket.OPEN) {
            websocket.send(JSON.stringify({ type: 'stop' }));
        }

        // Stop audio stream
        if (audioStream) {
            audioStream.getTracks().forEach(track => track.stop());
            audioStream = null;
        }

        // Close audio context
        if (audioContext) {
            audioContext.close();
            audioContext = null;
        }

        // Close WebSocket - Do NOT close immediately.
        // Wait for server to send 'stopped' message to avoid race condition.
        // If we close now, server might crash trying to send final partial result.

        /* 
        if (websocket) {
            websocket.close();
            websocket = null;
        } 
        */

        // Clear Audio Queue
        audioQueue.length = 0;
        isPlaying = false;
        // Remove 'playing' class from all items if any
        document.querySelectorAll('.transcription-item.playing').forEach(el => el.classList.remove('playing'));

        startBtn.disabled = false;
        stopBtn.disabled = true;
        updateStatus('Disconnected', 'disconnected');
    } catch (error) {
        console.error("Error stopping recording:", error);
    }
}

async function synthesizeSpeech(e) {
    if (e) e.preventDefault(); // Prevent default form submission behavior
    const text = ttsText.value.trim();
    if (!text) {
        alert('Please enter some text to synthesize.');
        return;
    }

    ttsBtn.disabled = true;
    ttsAudioPlayer.style.display = 'none';
    updateTtsStatus('Synthesizing speech...', 'synthesizing', true);

    try {
        const response = await fetch(TTS_API_URL, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                text: text,
                source_language: ttsSourceLang.value,
                target_language: 'min_nan' // TTS output is always Min Nan
            }),
        });

        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || 'Synthesis failed');
        }

        const data = await response.json();

        // Construct full audio URL
        const audioUrl = `http://localhost:8000${data.audio_url}`;

        ttsAudioPlayer.src = audioUrl;
        ttsAudioPlayer.style.display = 'block';

        // Try to play automatically
        try {
            await ttsAudioPlayer.play();
            updateTtsStatus('Audio playing...', 'connected', true);
        } catch (err) {
            console.log("Autoplay prevented:", err);
            updateTtsStatus('Audio ready. Click play to listen.', 'connected', true);
        }

        // Refresh history
        fetchHistory();
    } catch (error) {
        console.error('TTS Error:', error);
        updateTtsStatus(`Error: ${error.message}`, 'disconnected', true);
    } finally {
        ttsBtn.disabled = false;
    }
}

startBtn.addEventListener('click', startRecording);
stopBtn.addEventListener('click', stopRecording);
ttsBtn.addEventListener('click', synthesizeSpeech);

// Clear transcriptions when language changes
languageSelect.addEventListener('change', () => {
    transcriptionBox.innerHTML = '<div class="empty-state">Language changed. Click "Start Recording" to begin...</div>';
});
