//! Synthetic Generation RAP Tool

use infinity_agent_core::tools::rap_tool::{RapTool, RapToolDescriptor};
use rap_client::http::SimpleHttpClient;
use serde_json::json;

pub fn make_synthetic_tool() -> RapTool<SimpleHttpClient> {
    let descriptor = RapToolDescriptor {
        name: "generate_synthetic".to_string(),
        description: "Generate synthetic cases using web search + templates".to_string(),
        parameters: json!({
            "type": "object",
            "properties": {
                "n_propose": {"type": "integer", "default": 20},
                "merchants": {"type": "array", "items": {"type": "object"}},
                "minhash_threshold": {"type": "number", "default": 0.6},
                "use_web": {"type": "boolean", "default": true}
            },
            "required": ["n_propose"]
        }),
        display_script: None,
    };

    RapTool {
        descriptor,
        endpoint: "http://localhost:8007/invoke".to_string(),
        http_client: SimpleHttpClient::new(),
        callback_url: None,
    }
}