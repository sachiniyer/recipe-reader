# Recipe Reader

A Rust CLI application that uses OpenAI's real-time API with GPT-4o to create a voice-controlled recipe assistant. This application was fully vibecoded and contains bugs :stuck_out_tongue:. Please don't use this as a judge of my coding abilities/cleanliness.

# Usage

Run the application with:

```
cargo run --release
```

The application will:
1. Connect to the OpenAI Audio Conversations WebSocket API
2. Start recording audio from your microphone
3. Stream audio data to OpenAI in real-time
4. Process AI responses including recipe search functionality
5. Display the AI's responses directly in your terminal

Optional command-line arguments:

- `-r, --recipes-path <RECIPES_PATH>`: Path to recipes directory (default: "recipes")
- `-m, --max-time <MAX_TIME>`: Maximum recording time in seconds (default: 60)

Example:

```
cargo run --release -- --recipes-path my_recipes --max-time 30
```
