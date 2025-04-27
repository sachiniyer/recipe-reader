use std::env;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex as StdMutex};
use std::time::Duration;

use anyhow::{Context, Result};
use base64::prelude::*;
use clap::Parser;
use console::style;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{InputCallbackInfo, OutputCallbackInfo, SampleFormat};
use dotenv::dotenv;
use futures_util::{SinkExt, StreamExt};
use http::Request;
use rand::Rng;
use serde_json::json;
use tokio::sync::{mpsc, Mutex};
use tokio_tungstenite::connect_async;
use tokio_tungstenite::tungstenite::Message;
use uuid::Uuid;
use walkdir::WalkDir;

const OPENAI_MODEL: &str = "gpt-4o-realtime-preview-2024-12-17";

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Path to recipes directory
    #[arg(short, long, default_value = "recipes")]
    recipes_path: PathBuf,

    /// Maximum recording time in seconds
    #[arg(short, long, default_value_t = 60)]
    max_time: u64,
}

// All the struct definitions were replaced with direct JSON manipulation
// using serde_json::json! macro, so they're no longer needed

/// Structure to hold recipe information
struct Recipe {
    name: String,
    content: String,
}

/// Structure to hold all recipes
struct RecipeDatabase {
    recipes: Vec<Recipe>,
}

impl RecipeDatabase {
    fn new(recipes_path: &Path) -> Result<Self> {
        let mut recipes = Vec::new();

        for entry in WalkDir::new(recipes_path)
            .follow_links(true)
            .into_iter()
            .filter_map(|e| e.ok())
        {
            if !entry.file_type().is_file()
                || entry.path().extension().map_or(true, |ext| ext != "txt")
            {
                continue;
            }

            let content = fs::read_to_string(entry.path())?;
            let name = entry
                .path()
                .file_stem()
                .unwrap()
                .to_string_lossy()
                .to_string();

            recipes.push(Recipe { name, content });
        }

        Ok(RecipeDatabase { recipes })
    }

    fn search(&self, query: &str) -> Vec<&Recipe> {
        let query = query.to_lowercase();
        self.recipes
            .iter()
            .filter(|recipe| {
                recipe.name.to_lowercase().contains(&query)
                    || recipe.content.to_lowercase().contains(&query)
            })
            .collect()
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    // Parse command line arguments
    let args = Args::parse();

    // Load environment variables from .env file
    dotenv().ok();

    // Get OpenAI API key
    let api_key = env::var("OPENAI_API_KEY")
        .context("OPENAI_API_KEY must be set in .env file or environment")?;

    // Load recipe database
    let recipe_db = Arc::new(
        RecipeDatabase::new(&args.recipes_path).context("Failed to load recipe database")?,
    );

    println!(
        "{}",
        style("Recipe Reader - Voice Assistant").bold().green()
    );
    println!(
        "Loaded {} recipes from {}",
        recipe_db.recipes.len(),
        args.recipes_path.display()
    );
    println!(
        "Press Enter to start recording (max {} seconds)",
        args.max_time
    );

    // Set up audio input
    let host = cpal::default_host();
    let device = host
        .default_input_device()
        .context("No input audio device available")?;
    let config = device
        .default_input_config()
        .context("Failed to get default input config")?;

    println!("Audio input device: {}", device.name()?);
    println!(
        "Sample format: {:?}, Sample rate: {}",
        config.sample_format(),
        config.sample_rate().0
    );

    // Create channels for audio data (input)
    let (tx, rx) = mpsc::channel::<Vec<u8>>(32);
    let rx = Arc::new(Mutex::new(rx));
    
    // Create channel for audio output
    let (audio_out_tx, audio_out_rx) = mpsc::channel::<Vec<u8>>(32);
    let audio_out_rx = Arc::new(Mutex::new(audio_out_rx));

    // Set up audio output
    let output_device = host
        .default_output_device()
        .context("No output audio device available")?;
    let output_config = output_device
        .default_output_config()
        .context("Failed to get default output config")?;

    println!("Audio output device: {}", output_device.name()?);
    println!(
        "Output format: {:?}, Sample rate: {}",
        output_config.sample_format(),
        output_config.sample_rate().0
    );

    // Generate a unique session ID
    let session_id = Uuid::new_v4().to_string();

    // Start recording and connecting to OpenAI in separate tasks
    let sample_rate = config.sample_rate().0;
    let audio_task = tokio::spawn(record_audio(device, config, tx, args.max_time));
    
    // Start audio output task
    let audio_out_task = tokio::spawn(play_audio(output_device, output_config, audio_out_rx.clone()));
    
    let openai_task = tokio::spawn(connect_to_openai(
        api_key,
        sample_rate,
        rx,
        recipe_db,
        session_id,
        audio_out_tx,
    ));

    println!(
        "{}",
        style("Recording started... Speak your recipe question.").bold()
    );
    println!("Wait for a response or press Ctrl+C to stop.");

    // Wait for all tasks to complete
    let (audio_result, audio_out_result, openai_result) = tokio::join!(audio_task, audio_out_task, openai_task);

    // Propagate any errors from the tasks
    audio_result??;
    audio_out_result??;
    openai_result??;

    Ok(())
}

async fn record_audio(
    device: cpal::Device,
    config: cpal::SupportedStreamConfig,
    tx: mpsc::Sender<Vec<u8>>,
    max_seconds: u64,
) -> Result<()> {
    let err_fn = |err| eprintln!("Audio input error: {}", err);
    let tx_clone = tx.clone();

    let stream = match config.sample_format() {
        SampleFormat::F32 => device.build_input_stream(
            &config.into(),
            move |data: &[f32], _: &InputCallbackInfo| {
                let mut bytes = Vec::with_capacity(data.len() * 4);
                for &sample in data {
                    bytes.extend_from_slice(&sample.to_le_bytes());
                }
                let _ = tx_clone.try_send(bytes);
            },
            err_fn,
            None,
        )?,
        SampleFormat::I16 => device.build_input_stream(
            &config.into(),
            move |data: &[i16], _: &InputCallbackInfo| {
                let mut bytes = Vec::with_capacity(data.len() * 2);
                for &sample in data {
                    bytes.extend_from_slice(&sample.to_le_bytes());
                }
                let _ = tx_clone.try_send(bytes);
            },
            err_fn,
            None,
        )?,
        SampleFormat::U16 => device.build_input_stream(
            &config.into(),
            move |data: &[u16], _: &InputCallbackInfo| {
                let mut bytes = Vec::with_capacity(data.len() * 2);
                for &sample in data {
                    bytes.extend_from_slice(&sample.to_le_bytes());
                }
                let _ = tx_clone.try_send(bytes);
            },
            err_fn,
            None,
        )?,
        _ => {
            return Err(anyhow::anyhow!(
                "Unsupported sample format: {:?}",
                config.sample_format()
            ))
        }
    };

    stream.play()?;

    // Stop recording after max_seconds
    let _ = tokio::time::sleep(Duration::from_secs(max_seconds));

    // Explicitly drop the stream to stop recording
    drop(stream);

    Ok(())
}

async fn connect_to_openai(
    api_key: String,
    _sample_rate: u32,
    rx: Arc<Mutex<mpsc::Receiver<Vec<u8>>>>,
    recipe_db: Arc<RecipeDatabase>,
    _session_id: String, // Unused, we get a new session ID from the API
    audio_out_tx: mpsc::Sender<Vec<u8>>,
) -> Result<()> {
    println!("Establishing connection to OpenAI API...");

    let url = format!("wss://api.openai.com/v1/realtime?model={}", OPENAI_MODEL);
    println!("WebSocket URL: {}", url);

    // Build the WebSocket request with the ephemeral key
    let req = Request::builder()
        .uri(url)
        .header("Authorization", format!("Bearer {}", api_key))
        .header("Content-Type", "application/json")
        .header("Host", "api.openai.com")
        .header("Connection", "Upgrade")
        .header("Upgrade", "websocket")
        .header("Sec-WebSocket-Version", "13")
        .header("Sec-WebSocket-Key", generate_websocket_key())
        .header("OpenAI-Beta", "realtime=v1")
        .body(())
        .context("Failed to build WebSocket request")?;

    println!("Connecting to OpenAI Audio WebSocket API...");

    let (ws_stream, response) = connect_async(req)
        .await
        .context("Failed to connect to OpenAI WebSocket API")?;

    println!("Connected to OpenAI API (HTTP {})", response.status());

    println!("Connected to OpenAI API");

    // Use two MPSC channels for communication:
    // 1. For sending audio data
    // 2. For sending tool results
    let (audio_tx, mut audio_rx) = mpsc::channel::<String>(32);
    let (tool_result_tx, mut tool_result_rx) = mpsc::channel::<String>(32);

    // The session is already initialized through the REST API
    // Now we'll just set up the WebSocket for sending audio and receiving responses

    // Split the WebSocket stream
    let (mut write, mut read) = ws_stream.split();

    println!("WebSocket connection established, ready to stream audio");

    // Spawn a task to handle WebSocket writes
    let writer_task = tokio::spawn(async move {
        // First, send session.update to configure the session
        let session_update = json!({
            "type": "session.update",
            "session": {
                "instructions": "You are a helpful assistant that answers questions about recipes. Keep your answers concise and useful.",
                "modalities": ["text", "audio"],
                "voice": "sage",
                "input_audio_format": "pcm16",
                "output_audio_format": "pcm16",
                "turn_detection": {
                    "type": "server_vad",
                    "threshold": 0.5,
                    "prefix_padding_ms": 300,
                    "silence_duration_ms": 500,
                    "create_response": true
                },
                "tools": [
                    {
                        "type": "function",
                        "name": "find_recipe",
                        "description": "Search for recipes by keywords",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "query": {
                                    "type": "string",
                                    "description": "The search query for finding recipes"
                                }
                            },
                            "required": ["query"]
                        }
                    }
                ],
                "tool_choice": "auto",
                "temperature": 0.8,
                "max_response_output_tokens": "inf"
            }
        });

        if write
            .send(Message::Text(session_update.to_string()))
            .await
            .is_err()
        {
            return;
        }

        println!("Sent session configuration");

        // Process messages from both channels
        loop {
            tokio::select! {
                Some(audio_msg) = audio_rx.recv() => {
                    if write.send(Message::Text(audio_msg)).await.is_err() {
                        break;
                    }
                }
                Some(tool_result) = tool_result_rx.recv() => {
                    if write.send(Message::Text(tool_result)).await.is_err() {
                        break;
                    }
                }
                else => break,
            }
        }
    });

    // Spawn a task to process audio data
    let audio_task = tokio::spawn(async move {
        let mut rx_guard = rx.lock().await;

        // Send audio data
        while let Some(audio_chunk) = rx_guard.recv().await {
            // Encode audio data as base64
            let audio_base64 = BASE64_STANDARD.encode(&audio_chunk);

            // Send audio data using input_audio_buffer.append
            let audio_message = json!({
                "type": "input_audio_buffer.append",
                "audio": audio_base64
            });

            if audio_tx.send(audio_message.to_string()).await.is_err() {
                break;
            }
        }

        // When VAD is enabled, we don't need to explicitly commit
        // But for completeness, we'll send the commit message followed by response.create
        let commit_message = json!({
            "type": "input_audio_buffer.commit"
        });

        let _ = audio_tx.send(commit_message.to_string()).await;

        // Create a response after committing the audio buffer
        let response_message = json!({
            "type": "response.create"
        });

        let _ = audio_tx.send(response_message.to_string()).await;
    });

    // Process incoming messages
    while let Some(msg) = read.next().await {
        match msg {
            Ok(Message::Text(text)) => {
                // Print raw message for debugging
                //println!("Received: {}", text);

                let parsed: serde_json::Value = match serde_json::from_str(&text) {
                    Ok(v) => v,
                    Err(e) => {
                        eprintln!("Failed to parse response as JSON: {}", e);
                        continue;
                    }
                };

                // Extract message type
                let msg_type = parsed["type"].as_str().unwrap_or("unknown");
                println!("Message type: {}", msg_type);

                match msg_type {
                    "session.created" => {
                        println!("Session created successfully");
                    }
                    "session.updated" => {
                        println!("Session updated successfully");
                    }
                    "input_audio_buffer.speech_started" => {
                        println!("Speech detected, listening...");
                    }
                    "input_audio_buffer.speech_stopped" => {
                        println!("Speech stopped");
                    }
                    "input_audio_buffer.committed" => {
                        println!("Audio buffer committed");
                    }
                    "response.created" => {
                        println!("Response creation started");
                    }
                    "response.text.delta" => {
                        if let Some(delta) = parsed["delta"]["text"].as_str() {
                            print!("{}", delta);
                            std::io::stdout().flush().unwrap();
                        }
                    }
                    "response.audio_transcript.delta" => {
                        println!("Audio transcript delta received");
                    }
                    "response.audio.delta" => {
                        if let Some(audio_base64) = parsed["delta"].as_str() {
                            // Decode base64 audio data
                            match BASE64_STANDARD.decode(audio_base64) {
                                Ok(audio_data) => {
                                    // Send audio data to output channel
                                    if let Err(e) = audio_out_tx.try_send(audio_data) {
                                        eprintln!("Failed to send audio data to output: {}", e);
                                    }
                                }
                                Err(e) => {
                                    eprintln!("Failed to decode audio data: {}", e);
                                }
                            }
                        }
                    }
                    "response.audio.done" => {
                        println!("\n{}", style("Audio response completed").bold());
                    }
                    "response.text.done" => {
                        println!("\n{}", style("Text response completed").bold());
                    }
                    "response.done" => {
                        println!("\n{}", style("Response fully completed").bold());
                    }
                    "response.output_item.added" => {
                        // Check if this is a function call
                        if let Some(item) = parsed["item"].as_object() {
                            if item.get("type").and_then(|t| t.as_str()) == Some("function_call") {
                                // Extract tool information
                                let function_call_id = item
                                    .get("call_id")
                                    .and_then(|id| id.as_str())
                                    .unwrap_or("unknown");
                                let function_name =
                                    item.get("name").and_then(|n| n.as_str()).unwrap_or("");
                                let arguments = item
                                    .get("arguments")
                                    .and_then(|a| a.as_str())
                                    .unwrap_or("{}");

                                println!(
                                    "Function call: {} with arguments: {}",
                                    function_name, arguments
                                );

                                if function_name == "find_recipe" {
                                    // Parse the arguments
                                    let args: serde_json::Value =
                                        serde_json::from_str(arguments).unwrap_or(json!({}));

                                    // Extract query parameter
                                    let query = args["query"].as_str().unwrap_or("");
                                    println!("Searching for recipe with query: {}", query);

                                    // Search the recipe database
                                    let results = recipe_db.search(query);

                                    // Format the results
                                    let output = if results.is_empty() {
                                        json!({
                                            "status": "not_found",
                                            "message": "No recipes found matching your query."
                                        })
                                    } else {
                                        let recipes_json = results
                                            .iter()
                                            .map(|recipe| {
                                                json!({
                                                    "name": recipe.name,
                                                    "content": recipe.content
                                                })
                                            })
                                            .collect::<Vec<_>>();

                                        json!({
                                            "status": "success",
                                            "count": results.len(),
                                            "recipes": recipes_json
                                        })
                                    };

                                    // Send the function call output
                                    let tool_result = json!({
                                        "type": "conversation.item.create",
                                        "item": {
                                            "type": "function_call_output",
                                            "call_id": function_call_id,
                                            "output": output.to_string()
                                        }
                                    });

                                    let result_json = tool_result.to_string();
                                    println!("Sending function call output: {}", result_json);
                                    let _ = tool_result_tx.send(result_json).await;

                                    // Create a response after sending the function call output
                                    let response_create = json!({
                                        "type": "response.create"
                                    });

                                    let _ = tool_result_tx.send(response_create.to_string()).await;
                                }
                            }
                        }
                    }
                    "error" => {
                        eprintln!("{}", parsed);
                    }
                    _ => {
                        // For debugging, print all message types
                        println!("Received message type: {}", msg_type);
                    }
                }
            }
            Ok(Message::Close(_)) => {
                println!("\n{}", style("Connection closed by server").bold());
                break;
            }
            Err(e) => {
                eprintln!("WebSocket error: {}", e);
                break;
            }
            _ => {}
        }
    }

    // When we reach this point, the WebSocket connection is closed
    // Wait for the writer task and audio task to complete
    let _ = tokio::join!(writer_task, audio_task);

    Ok(())
}

/// Generate a random WebSocket key as per RFC 6455
fn generate_websocket_key() -> String {
    let mut rng = rand::thread_rng();
    let mut key = [0u8; 16];
    rng.fill(&mut key);
    BASE64_STANDARD.encode(key)
}

async fn play_audio(
    device: cpal::Device,
    config: cpal::SupportedStreamConfig,
    rx: Arc<Mutex<mpsc::Receiver<Vec<u8>>>>,
) -> Result<()> {
    let err_fn = |err: cpal::StreamError| eprintln!("Audio output error: {}", err);
    
    // Create a stream to handle audio playback and handle it in a separate function
    // This approach avoids Send/Sync issues with cpal::Stream
    let _stream = setup_audio_playback(device, config, rx.clone())?;
    
    // Keep the function alive
    loop {
        tokio::time::sleep(Duration::from_secs(1)).await;
    }
}

fn setup_audio_playback(
    device: cpal::Device,
    config: cpal::SupportedStreamConfig,
    rx: Arc<Mutex<mpsc::Receiver<Vec<u8>>>>,
) -> Result<()> {
    let err_fn = |err: cpal::StreamError| eprintln!("Audio output error: {}", err);
    
    // Move receiver to a thread-safe location that can be accessed from the callback
    let audio_buffer: Arc<StdMutex<Vec<u8>>> = Arc::new(StdMutex::new(Vec::new()));
    let audio_buffer_clone = audio_buffer.clone();
    
    // Spawn a regular thread (not a tokio task) to handle the audio receiving
    std::thread::spawn(move || {
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            let mut rx_guard = rx.lock().await;
            while let Some(audio_chunk) = rx_guard.recv().await {
                if let Ok(mut buffer) = audio_buffer_clone.lock() {
                    buffer.extend(audio_chunk);
                }
            }
        });
    });
    
    let stream = match config.sample_format() {
        SampleFormat::F32 => {
            device.build_output_stream(
                &config.into(),
                move |data: &mut [f32], _: &OutputCallbackInfo| {
                    // Get audio data from the shared buffer
                    let mut buffer_lock = match audio_buffer.try_lock() {
                        Ok(guard) => guard,
                        Err(_) => return,  // Skip this callback if buffer is locked
                    };
                    
                    if buffer_lock.len() >= 2 {
                        // Process PCM16 bytes to f32 samples
                        let mut i = 0;
                        while i + 1 < buffer_lock.len() && i / 2 < data.len() {
                            // Correctly interpret as signed 16-bit PCM
                            let pcm16 = i16::from_le_bytes([buffer_lock[i], buffer_lock[i+1]]);
                            // Convert to float in range [-1.0, 1.0]
                            data[i/2] = pcm16 as f32 / 32767.0;
                            i += 2;
                        }
                        
                        // Remove processed bytes
                        if i > 0 {
                            buffer_lock.drain(0..i);
                        }
                        
                        // Zero out the rest of the buffer if not enough samples
                        if i/2 < data.len() {
                            for sample in &mut data[i/2..] {
                                *sample = 0.0;
                            }
                        }
                    } else {
                        // No data available, output silence
                        for sample in data.iter_mut() {
                            *sample = 0.0;
                        }
                    }
                },
                err_fn,
                None,
            )?
        },
        SampleFormat::I16 => {
            device.build_output_stream(
                &config.into(),
                move |data: &mut [i16], _: &OutputCallbackInfo| {
                    // Get audio data from the shared buffer
                    let mut buffer_lock = match audio_buffer.try_lock() {
                        Ok(guard) => guard,
                        Err(_) => return,  // Skip this callback if buffer is locked
                    };
                    
                    if buffer_lock.len() >= 2 {
                        // Process PCM16 bytes to i16 samples - direct conversion
                        let mut i = 0;
                        while i + 1 < buffer_lock.len() && i / 2 < data.len() {
                            // PCM16 data is already in i16 format
                            data[i/2] = i16::from_le_bytes([buffer_lock[i], buffer_lock[i+1]]);
                            i += 2;
                        }
                        
                        // Remove processed bytes
                        if i > 0 {
                            buffer_lock.drain(0..i);
                        }
                        
                        // Zero out the rest of the buffer if not enough samples
                        if i/2 < data.len() {
                            for sample in &mut data[i/2..] {
                                *sample = 0;
                            }
                        }
                    } else {
                        // No data available, output silence
                        for sample in data.iter_mut() {
                            *sample = 0;
                        }
                    }
                },
                err_fn,
                None,
            )?
        },
        SampleFormat::U16 => {
            device.build_output_stream(
                &config.into(),
                move |data: &mut [u16], _: &OutputCallbackInfo| {
                    // Get audio data from the shared buffer
                    let mut buffer_lock = match audio_buffer.try_lock() {
                        Ok(guard) => guard,
                        Err(_) => return,  // Skip this callback if buffer is locked
                    };
                    
                    if buffer_lock.len() >= 2 {
                        // Process PCM16 bytes to u16 samples (with proper offset)
                        let mut i = 0;
                        while i + 1 < buffer_lock.len() && i / 2 < data.len() {
                            // Convert signed PCM16 to unsigned u16
                            let i16_sample = i16::from_le_bytes([buffer_lock[i], buffer_lock[i+1]]);
                            // Add 32768 to shift from [-32768,32767] to [0,65535]
                            let u16_sample = (i16_sample as i32 + 32768) as u16;
                            data[i/2] = u16_sample;
                            i += 2;
                        }
                        
                        // Remove processed bytes
                        if i > 0 {
                            buffer_lock.drain(0..i);
                        }
                        
                        // Zero out the rest of the buffer if not enough samples
                        if i/2 < data.len() {
                            for sample in &mut data[i/2..] {
                                *sample = 32768;  // Middle value for u16 (silence)
                            }
                        }
                    } else {
                        // No data available, output silence
                        for sample in data.iter_mut() {
                            *sample = 32768;  // Middle value for u16 (silence)
                        }
                    }
                },
                err_fn,
                None,
            )?
        },
        _ => {
            return Err(anyhow::anyhow!(
                "Unsupported sample format: {:?}",
                config.sample_format()
            ))
        }
    };

    // Play the stream
    stream.play()?;
    
    // Keep the stream alive by storing it in a global or static variable
    // We use a Box leak to ensure it stays alive for the program duration
    Box::leak(Box::new(stream));
    
    println!("Audio output initialized and playing");
    
    Ok(())
}
