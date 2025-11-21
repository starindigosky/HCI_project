// audio-processor.js
class AudioProcessor extends AudioWorkletProcessor {
    constructor() {
        super();
        this._bufferSize = 4096;
        this._buffer = new Float32Array(this._bufferSize);
        this._bufferPos = 0;
    }

    process(inputs, outputs, parameters) {
        const input = inputs[0];
        if (input.length > 0) {
            const inputChannel = input[0];
            
            // Buffer the audio data
            for (let i = 0; i < inputChannel.length; i++) {
                this._buffer[this._bufferPos++] = inputChannel[i];
                if (this._bufferPos === this._bufferSize) {
                    // Send the buffered data to the main thread
                    this.port.postMessage(this._buffer);
                    this._bufferPos = 0;
                }
            }
        }
        return true; // Keep the processor alive
    }
}

registerProcessor('audio-processor', AudioProcessor);
