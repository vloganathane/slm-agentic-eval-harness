//! Inference RAP Tool

use infinity_agent_core::tools::rap_tool::{RapTool, RapToolDescriptor};
use rap_client::http::SimpleHttpClient;
use serde_json::json;

pub fn make_inference_tool(
    _model_endpoint: String,
    _default_model: String,
    _api_key: Option<String>,
) -> RapTool<SimpleHttpClient> {
    let descriptor = RapToolDescriptor {
        name: "run_inference".to_string(),
        description: "Run model inference on a prompt (llama-server / OpenAI-compatible)".to_string(),
        parameters: json!({
            "type": "object",
            "properties": {
                "prompt": {"type": "string"},
                "model": {"type": "string", "default": "qwen2.5-1.5b-instruct"},
                "json_mode": {"type": "boolean", "default": false},
                "temperature": {"type": "number", "default": 0.0},
                "max_tokens": {"type": "integer", "default": 2048}
            },
            "required": ["prompt"]
        }),
        display_script: None,
    };

    RapTool {
        descriptor,
        endpoint: "http://localhost:8002/v1/chat/completions".to_string(),
        http_client: SimpleHttpClient::new(),
        callback_url: None,
    }
}