


import asyncio
import websockets
import json
import pyaudio
import sys


class RealtimeASRClient:


    def __init__(
        self,
        ws_url="ws://localhost:8000/api/v1/ws/asr",
        language="chinese",
        interim_results=False,
        sample_rate=16000,
        chunk_duration=0.5
    ):

        self.ws_url = ws_url
        self.language = language
        self.interim_results = interim_results
        self.sample_rate = sample_rate
        self.chunk_duration = chunk_duration
        self.chunk_size = int(sample_rate * chunk_duration)


        self.audio = None
        self.stream = None


        self.websocket = None

        print(f"Real-time ASR Client initialized")
        print(f"  WebSocket URL: {ws_url}")
        print(f"  Language: {language}")
        print(f"  Interim results: {interim_results}")
        print(f"  Sample rate: {sample_rate} Hz")

    async def connect(self):

        try:
            self.websocket = await websockets.connect(self.ws_url)
            print("✅ Connected to server")


            config = {
                "type": "config",
                "language": self.language,
                "interim_results": self.interim_results
            }
            await self.websocket.send(json.dumps(config))


            response = await self.websocket.recv()
            response_data = json.loads(response)
            if response_data.get("type") == "config_updated":
                print(f"✅ Configuration updated: {response_data}")
            else:
                print(f"⚠️  Unexpected response: {response_data}")

        except Exception as e:
            print(f"❌ Error connecting: {str(e)}")
            raise

    async def start_audio_capture(self):

        try:
            self.audio = pyaudio.PyAudio()


            self.stream = self.audio.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.chunk_size
            )

            print("🎤 Microphone opened, starting capture...")
            print("Press Ctrl+C to stop")

        except Exception as e:
            print(f"❌ Error opening microphone: {str(e)}")
            raise

    async def send_audio(self):

        try:
            while True:

                audio_data = self.stream.read(
                    self.chunk_size,
                    exception_on_overflow=False
                )


                await self.websocket.send(audio_data)


                await asyncio.sleep(0.01)

        except KeyboardInterrupt:
            print("\n🛑 Stopping...")
        except Exception as e:
            print(f"❌ Error sending audio: {str(e)}")
            raise

    async def receive_messages(self):

        try:
            while True:
                message = await self.websocket.recv()
                data = json.loads(message)

                if data["type"] == "transcription":
                    is_final = data.get("is_final", True)
                    text = data.get("text", "")

                    if is_final:
                        print(f"\n✅ Final: {text}")
                    else:
                        print(f"\r⏳ Interim: {text}", end="", flush=True)

                elif data["type"] == "error":
                    print(f"\n❌ Error: {data['message']}")

                elif data["type"] == "stopped":
                    print("\n✅ Session stopped")
                    break

        except websockets.exceptions.ConnectionClosed:
            print("\n⚠️  Connection closed")
        except Exception as e:
            print(f"\n❌ Error receiving: {str(e)}")

    async def stop(self):

        if self.websocket:
            try:

                await self.websocket.send(json.dumps({"type": "stop"}))
                print("\n📝 Getting final transcription...")


                await asyncio.sleep(2)

            except Exception as e:
                print(f"❌ Error stopping: {str(e)}")

    async def cleanup(self):


        if self.stream:
            self.stream.stop_stream()
            self.stream.close()


        if self.audio:
            self.audio.terminate()


        if self.websocket:
            await self.websocket.close()

        print("✅ Cleanup complete")

    async def run(self):

        try:

            await self.connect()


            await self.start_audio_capture()


            send_task = asyncio.create_task(self.send_audio())
            recv_task = asyncio.create_task(self.receive_messages())


            done, pending = await asyncio.wait(
                [send_task, recv_task],
                return_when=asyncio.FIRST_COMPLETED
            )


            for task in pending:
                task.cancel()


            await self.stop()

        except KeyboardInterrupt:
            print("\n🛑 Interrupted by user")
        except Exception as e:
            print(f"❌ Error: {str(e)}")
        finally:
            await self.cleanup()


async def main():

    print("=" * 60)
    print("  Min Nan & Chinese Real-time ASR Client")
    print("=" * 60)
    print()


    print("Select language:")
    print("  1. Chinese (中文)")
    print("  2. Min Nan (閩南語)")

    choice = input("Enter choice (1 or 2, default: 1): ").strip() or "1"

    language = "chinese" if choice == "1" else "min_nan"

    interim_input = input("Show interim results? (y/n, default: n): ").strip().lower()
    interim_results = interim_input == "y"

    print()


    client = RealtimeASRClient(
        language=language,
        interim_results=interim_results
    )

    await client.run()

    print()
    print("=" * 60)
    print("  Session ended")
    print("=" * 60)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nGoodbye!")
    except Exception as e:
        print(f"\n❌ Fatal error: {str(e)}")
        sys.exit(1)