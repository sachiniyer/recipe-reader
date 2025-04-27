# Recipe Reader

A Rust CLI application that uses OpenAI's real-time API with GPT-4o to create a voice-controlled recipe assistant. Talk to your computer and get recipe recommendations from your local collection.

## Features

- Voice interaction using your computer's microphone
- Real-time audio processing with OpenAI's GPT-4o audio conversations API
- Local recipe database in simple text format
- Natural language recipe search and recommendations

## Requirements

- Rust (edition 2021)
- OpenAI API key with access to GPT-4o
- Audio input device (microphone)

## Setup

1. Clone this repository
2. Create a `.env` file in the root directory with your OpenAI API key:
   ```
   OPENAI_API_KEY=your_openai_api_key_here
   ```
3. Build the project:
   ```
   cargo build --release
   ```

## Usage

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

## Recipe Format

Recipes should be stored as plain text files (.txt) in the recipes directory. Each recipe should have a title, ingredients section, and instructions section. For example:

```
# Chocolate Chip Cookies

## Ingredients
- 2 1/4 cups all-purpose flour
- 1 teaspoon baking soda
...

## Instructions
1. Preheat oven to 375°F (190°C).
2. Combine flour, baking soda, and salt in small bowl.
...
```

## License

MIT