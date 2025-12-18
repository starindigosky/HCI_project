
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


const ttsBtn = document.getElementById('ttsBtn');
const ttsText = document.getElementById('ttsText');
const ttsSourceLang = document.getElementById('ttsSourceLang');
const ttsStatus = document.getElementById('ttsStatus');
const ttsAudioPlayer = document.getElementById('ttsAudioPlayer');
const historyList = document.getElementById('historyList');


const audioQueue = [];
let isPlaying = false;
let isSystemSpeaking = false;

async function processQueue() {
    if (isPlaying || audioQueue.length === 0) return;

    isPlaying = true;
    isSystemSpeaking = true;
    updateStatus('🔊 Speaking...', 'speaking');

    const { player, container } = audioQueue.shift();

    try {
        container.classList.add('playing');
        await player.play();

        player.onended = () => {
            container.classList.remove('playing');
            isPlaying = false;


            if (audioQueue.length === 0) {

                updateStatus('⏳ Cooldown...', 'speaking');
                setTimeout(() => {

                    if (audioQueue.length === 0) {
                        isSystemSpeaking = false;
                        if (websocket && websocket.readyState === WebSocket.OPEN) {
                            updateStatus('🔴 Recording...', 'recording');
                        }
                    }
                }, 2000);
            }

            processQueue();
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
            processQueue();
        };

    } catch (error) {
        console.error("Audio playback failed:", error);
        container.classList.remove('playing');
        isPlaying = false;
        processQueue();
    }
}


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

            if (parts.length > 1) {

                displayFilename = parts.slice(0, -1).join('_');
            }
        }



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
                <button class="delete-btn" onclick="deleteAudio('${file.filename.replace(/'/g, "\\'")}')" title="Delete">
                    <svg viewBox="0 0 24 24" width="16" height="16" fill="currentColor">
                        <path d="M6 19c0 1.1.9 2 2 2h8c1.1 0 2-.9 2-2V7H6v12zM19 4h-3.5l-1-1h-5l-1 1H5v2h14V4z"/>
                    </svg>
                </button>
            </div>
        `;
    }).join('');
}

async function deleteAudio(filename) {
    if (!confirm('Start to delete this audio?')) return;

    try {
        const response = await fetch(`http://localhost:8000/api/v1/audio/${filename}`, {
            method: 'DELETE'
        });

        if (response.ok) {
            fetchHistory();
        } else {
            const data = await response.json();
            alert('Delete failed: ' + (data.detail || 'Unknown error'));
        }
    } catch (error) {
        console.error('Error deleting audio:', error);
        alert('Error deleting audio');
    }
}


fetchHistory();


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
                source_language: languageSelect ? languageSelect.value : 'chinese',
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


let currentSpeaker = 'A';



async function addTranscription(text, isFinal = true, details = {}, speakerOverride = null) {

    const emptyState = transcriptionBox.querySelector('.empty-state');
    if (emptyState) emptyState.remove();


    const originalText = text;
    const translatedText = details.translated_text || text;
    const confidence = details.confidence || 'low';
    const method = details.method || 'fallback';



    const speaker = speakerOverride || details.speaker || currentSpeaker;


    const confidenceClass = confidence === 'high' ? 'high-confidence' : 'low-confidence';


    let item;
    const existingInterim = transcriptionBox.querySelector('.transcription-item.interim');

    if (!isFinal) {
        if (existingInterim) {
            item = existingInterim;


            if (!item.dataset.speaker) item.dataset.speaker = speaker;
        } else {
            item = document.createElement('div');

            item.dataset.speaker = speaker;

            item.className = `transcription-item interim speaker-${speaker}`;

            const contentDiv = document.createElement('div');
            contentDiv.className = 'transcription-content';
            item.appendChild(contentDiv);

            const metaDiv = document.createElement('div');
            metaDiv.className = 'transcription-meta';
            item.appendChild(metaDiv);

            transcriptionBox.appendChild(item);
        }
    } else {

        if (existingInterim) {
            item = existingInterim;
            item.classList.remove('interim');

            const lockedSpeaker = item.dataset.speaker || speaker;
            item.className = `transcription-item speaker-${lockedSpeaker}`;
        } else {
            item = document.createElement('div');

            item.dataset.speaker = speaker;
            item.className = `transcription-item speaker-${speaker}`;

            const contentDiv = document.createElement('div');
            contentDiv.className = 'transcription-content';
            item.appendChild(contentDiv);

            const metaDiv = document.createElement('div');
            metaDiv.className = 'transcription-meta';
            item.appendChild(metaDiv);

            transcriptionBox.appendChild(item);
        }
    }


    const displaySpeaker = item.dataset.speaker || speaker;


    const contentDiv = item.querySelector('.transcription-content');
    contentDiv.innerHTML = `
        <div class="speaker-label" style="font-size: 0.75em; color: #888; margin-bottom: 2px;">
            ${displaySpeaker === 'A' ? 'Speaker A' : 'Speaker B'}
        </div>
        <div class="source-text" style="font-size: 0.85em; color: #ccc; margin-bottom: 4px;">${originalText}</div>
        <div class="target-text ${confidenceClass}" style="font-size: 1.1em; font-weight: bold; color: ${confidence === 'high' ? '#2e7d32' : '#d84315'};">
            ${translatedText}
        </div>
        ${confidence === 'low' && isFinal ? '<div style="font-size:0.7em; color:#999;">(原文直讀)</div>' : ''}
    `;

    item.querySelector('.transcription-meta').textContent = `${isFinal ? 'Final' : 'Interim'} • ${method && method !== 'fallback' ? 'Dict Match' : 'Neural/Raw'} • ${new Date().toLocaleTimeString()}`;



    if (isFinal) {

        if (ttsText) {

            ttsText.value += (ttsText.value ? '\n' : '') + originalText;

            ttsText.scrollTop = ttsText.scrollHeight;
        }

        const audioContainer = document.createElement('div');
        audioContainer.className = 'transcription-audio-container';
        audioContainer.textContent = 'Synthesizing audio...';
        item.appendChild(audioContainer);

        const audioUrl = await getSynthesizedAudio(translatedText);
        if (audioUrl) {
            const audioPlayer = document.createElement('audio');
            audioPlayer.src = audioUrl;
            audioPlayer.type = "audio/wav";
            audioPlayer.controls = true;


            audioContainer.textContent = '';
            audioContainer.appendChild(audioPlayer);


            audioQueue.push({ player: audioPlayer, container: item });
            processQueue();
        } else {
            audioContainer.textContent = 'Audio synthesis failed.';
        }
    }

    transcriptionBox.scrollTop = transcriptionBox.scrollHeight;
}

async function startRecording() {
    isSystemSpeaking = false;
    try {

        websocket = new WebSocket(ASR_WS_URL);

        websocket.onopen = async () => {
            updateStatus('Connected', 'connected');


            const language = languageSelect ? languageSelect.value : 'chinese';
            const interimResults = interimResultsCheckbox ? interimResultsCheckbox.checked : false;

            websocket.send(JSON.stringify({
                type: 'config',
                language: language,
                interim_results: interimResults
            }));


            audioStream = await navigator.mediaDevices.getUserMedia({ audio: true });


            audioContext = new AudioContext({ sampleRate: 16000 });
            const source = audioContext.createMediaStreamSource(audioStream);


            const processor = audioContext.createScriptProcessor(4096, 1, 1);

            processor.onaudioprocess = (e) => {

                if (isSystemSpeaking) return;

                if (websocket && websocket.readyState === WebSocket.OPEN) {
                    const audioData = e.inputBuffer.getChannelData(0);

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
                let speakerForThisItem = null;


                if (message.cause === 'switch_speaker_forced') {


                    speakerForThisItem = (currentSpeaker === 'A' ? 'B' : 'A');
                    console.log(`Received forced finalize result. Attributing to PREVIOUS speaker: ${speakerForThisItem}`);
                }

                if (message.translated_text) {

                    addTranscription(message.text, message.is_final, {
                        translated_text: message.translated_text,
                        method: message.method,
                        confidence: message.confidence
                    }, speakerForThisItem);
                } else {

                    addTranscription(message.text, message.is_final, {}, speakerForThisItem);
                }
            } else if (message.type === 'error') {
                console.error('WebSocket error:', message.message);
                updateStatus('Error: ' + message.message, 'disconnected');
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
    isSystemSpeaking = false;


    try {

        if (websocket && websocket.readyState === WebSocket.OPEN) {
            websocket.send(JSON.stringify({ type: 'stop' }));
        }


        if (audioStream) {
            audioStream.getTracks().forEach(track => track.stop());
            audioStream = null;
        }


        if (audioContext) {
            audioContext.close();
            audioContext = null;
        }















        startBtn.disabled = false;
        stopBtn.disabled = true;
        updateStatus('Disconnected', 'disconnected');
    } catch (error) {
        console.error("Error stopping recording:", error);
    }
}

async function synthesizeSpeech(e) {
    if (e) e.preventDefault();
    const text = ttsText.value.trim();
    if (!text) {
        alert('Please enter some text to synthesize.');
        return;
    }


    if (ttsBtn.classList.contains('disabled')) return false;

    ttsBtn.classList.add('disabled');

    isSystemSpeaking = true;

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
                source_language: ttsSourceLang ? ttsSourceLang.value : 'chinese',
                target_language: 'min_nan'
            }),
        });

        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || 'Synthesis failed');
        }

        const data = await response.json();


        const audioUrl = `http://localhost:8000${data.audio_url}`;

        ttsAudioPlayer.src = audioUrl;
        ttsAudioPlayer.style.display = 'block';


        try {
            await ttsAudioPlayer.play();
            updateTtsStatus('Audio playing...', 'connected', true);
        } catch (err) {
            console.log("Autoplay prevented:", err);
            updateTtsStatus('Audio ready. Click play to listen.', 'connected', true);
        }


        fetchHistory();
    } catch (error) {
        console.error('TTS Error:', error);
        updateTtsStatus(`Error: ${error.message}`, 'disconnected', true);
    } finally {
        ttsBtn.classList.remove('disabled');






        ttsAudioPlayer.onended = () => {
            isSystemSpeaking = false;
        };

        if (ttsAudioPlayer.paused) {
            isSystemSpeaking = false;
        }
    }
    return false;
}

startBtn.addEventListener('click', startRecording);
stopBtn.addEventListener('click', stopRecording);
ttsBtn.addEventListener('click', synthesizeSpeech);



if (languageSelect) {
    languageSelect.addEventListener('change', () => {
        transcriptionBox.innerHTML = '<div class="empty-state">Language changed. Click "Start Recording" to begin...</div>';
    });
}


if (interimResultsCheckbox) {
    interimResultsCheckbox.addEventListener('change', () => {
        if (websocket && websocket.readyState === WebSocket.OPEN) {
            const language = languageSelect ? languageSelect.value : 'chinese';
            const interimResults = interimResultsCheckbox.checked;

            console.log("Updating config: interim_results =", interimResults);
            websocket.send(JSON.stringify({
                type: 'config',
                language: language,
                interim_results: interimResults
            }));
        }
    });
}