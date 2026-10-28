//! Llama-server OpenAI-compatible provider for Infinity

use anyhow::Result;
use infinity_provider_protocol::{
    CompletionError, CompletionRequest, ErrorClass, ModelEntry, ModelProvider,
    ModelStream, StreamChunk, Usage, Message, UserContent, AssistantContent,
};
use futures_util::StreamExt;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio_stream::wrappers::ReceiverStream;
use tokio::sync::mpsc;

#[derive(Debug, Clone, Serialize, Deserialize)]
struct LlamaConfig {
    base_url: String,
    api_key: Option<String>,
    default_model: String,
}

impl Default for LlamaConfig {
    fn default() -> Self {
        Self {
            base_url: "http://localhost:8080/v1".to_string(),
            api_key: None,
            default_model: "qwen2.5-1.5b-instruct".to_string(),
        }
    }
}

#[derive(Debug, Deserialize)]
struct OpenAIChatResponse {
    choices: Vec<OpenAIChoice>,
    usage: Option<OpenAIUsage>,
}

#[derive(Debug, Deserialize)]
struct OpenAIChoice {
    delta: Option<OpenAIDelta>,
    message: Option<OpenAIMessage>,
    finish_reason: Option<String>,
    index: usize,
}

#[derive(Debug, Deserialize)]
struct OpenAIDelta {
    content: Option<String>,
    role: Option<String>,
}

#[derive(Debug, Deserialize)]
struct OpenAIMessage {
    content: Option<String>,
    role: String,
}

#[derive(Debug, Deserialize)]
struct OpenAIUsage {
    prompt_tokens: u32,
    completion_tokens: u32,
    total_tokens: u32,
}

#[derive(Debug, Deserialize)]
struct OpenAIModelsResponse {
    data: Vec<OpenAIModel>,
}

#[derive(Debug, Deserialize)]
struct OpenAIModel {
    id: String,
    object: String,
    owned_by: String,
}

pub struct LlamaProvider {
    client: Client,
    config: LlamaConfig,
}

impl LlamaProvider {
    pub fn new(config: LlamaConfig) -> Self {
        Self {
            client: Client::new(),
            config,
        }
    }
}

#[async_trait::async_trait]
impl ModelProvider for LlamaProvider {
    async fn list_models(&self) -> Result<Vec<ModelEntry>, Box<dyn std::error::Error + Send + Sync>> {
        // Try to fetch from the API, fall back to default
        let resp = self
            .client
            .get(format!("{}/models", self.config.base_url))
            .send()
            .await;

        if let Ok(r) = resp {
            if r.status().is_success() {
                if let Ok(models_resp) = r.json::<OpenAIModelsResponse>().await {
                    return Ok(models_resp
                        .data
                        .into_iter()
                        .map(|m| ModelEntry {
                            model_id: m.id.clone(),
                            display_name: m.id,
                            context_window: 32768,
                            max_output_tokens: Some(4096),
                            supports_image_input: false,
                        })
                        .collect());
                }
            }
        }

        // Fallback
        Ok(vec![ModelEntry {
            model_id: self.config.default_model.clone(),
            display_name: self.config.default_model.clone(),
            context_window: 32768,
            max_output_tokens: Some(4096),
            supports_image_input: false,
        }])
    }

    async fn invoke_model(
        &self,
        model_id: &str,
        request: CompletionRequest,
    ) -> Result<ModelStream, CompletionError> {
        let mut openai_messages = vec![];

        if let Some(preamble) = request.preamble {
            openai_messages.push(serde_json::json!({
                "role": "system",
                "content": preamble
            }));
        }

        for msg in request.chat_history {
            match msg {
                Message::User { content } => {
                    for c in content {
                        if let UserContent::Text(t) = c {
                            openai_messages.push(serde_json::json!({
                                "role": "user",
                                "content": t.text
                            }));
                        }
                    }
                }
                Message::Assistant { content } => {
                    for c in content {
                        if let AssistantContent::Text(t) = c {
                            openai_messages.push(serde_json::json!({
                                "role": "assistant",
                                "content": t.text
                            }));
                        }
                    }
                }
            }
        }

        let body = serde_json::json!({
            "model": model_id,
            "messages": openai_messages,
            "stream": true,
            "max_tokens": request.max_tokens.unwrap_or(4096),
            "temperature": 0.7,
        });

        let mut req = self
            .client
            .post(format!("{}/chat/completions", self.config.base_url))
            .json(&body);

        if let Some(api_key) = &self.config.api_key {
            req = req.bearer_auth(api_key);
        }

        let resp = req.send().await
            .map_err(|e| CompletionError::provider(ErrorClass::Transient, format!("Request failed: {}", e)))?;
        
        let status = resp.status();
        if !status.is_success() {
            let err_text = resp.text().await.unwrap_or_default();
            return Err(CompletionError::provider(ErrorClass::Fatal, format!("API error {}: {}", status, err_text)));
        }

        // Use a channel to bridge the reqwest stream to our async_stream
        let (tx, rx) = mpsc::channel(100);
        
        // Spawn a task to read from the response stream
        tokio::spawn(async move {
            let mut stream = resp.bytes_stream();
            let mut buffer = String::new();

            while let Some(chunk) = stream.next().await {
                match chunk {
                    Ok(bytes) => {
                        buffer.push_str(&String::from_utf8_lossy(&bytes));
                        // Parse SSE format - split by newlines
                        let lines: Vec<&str> = buffer.lines().collect();
                        for (i, line) in lines.iter().enumerate() {
                            if line.starts_with("data: ") {
                                let data = &line[6..];
                                if data == "[DONE]" {
                                    let _ = tx.send(Ok(StreamChunk::Final(infinity_provider_protocol::FinalResponse { usage: Some(Usage::default()) }))).await;
                                    return;
                                }
                                if let Ok(parsed) = serde_json::from_str::<OpenAIChatResponse>(data) {
                                    if let Some(choice) = parsed.choices.first() {
                                        if let Some(delta) = &choice.delta {
                                            if let Some(content) = &delta.content {
                                                if !content.is_empty() {
                                                    let _ = tx.send(Ok(StreamChunk::Text(content.clone()))).await;
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        // Keep the last potentially incomplete line in buffer
                        if let Some(last) = lines.last() {
                            if !buffer.ends_with('\n') {
                                buffer = last.to_string();
                            } else {
                                buffer.clear();
                            }
                        }
                    }
                    Err(e) => {
                        let _ = tx.send(Err(CompletionError::provider(ErrorClass::Transient, format!("Stream error: {}", e)))).await;
                        return;
                    }
                }
            }
            
            let _ = tx.send(Ok(StreamChunk::Final(infinity_provider_protocol::FinalResponse { usage: Some(Usage::default()) }))).await;
        });

        let stream = ReceiverStream::new(rx);
        Ok(Box::pin(stream))
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    let mut config = LlamaConfig::default();

    // Override from env if present
    if let Ok(base_url) = std::env::var("LLAMA_BASE_URL") {
        config.base_url = base_url;
    }
    if let Ok(model) = std::env::var("LLAMA_MODEL") {
        config.default_model = model;
    }
    if let Ok(api_key) = std::env::var("LLAMA_API_KEY") {
        config.api_key = Some(api_key);
    }

    let provider = Arc::new(LlamaProvider::new(config));
    
    let (path, server) = infinity_provider_protocol::remote::serve_provider(provider)
        .expect("Failed to bind provider socket");

    // Print socket path for the daemon to discover
    println!("{}", path.display());
    
    // Run the server
    server.await;
    
    Ok(())
}