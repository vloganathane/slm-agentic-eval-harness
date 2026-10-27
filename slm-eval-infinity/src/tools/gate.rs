//! Promotion Gate RAP Tool

use infinity_agent_core::tools::rap_tool::{RapTool, RapToolDescriptor};
use rap_client::http::SimpleHttpClient;
use serde_json::json;

pub fn make_gate_tool() -> RapTool<SimpleHttpClient> {
    let descriptor = RapToolDescriptor {
        name: "evaluate_gate".to_string(),
        description: "Evaluate promotion gate using CI lower bounds".to_string(),
        parameters: json!({
            "type": "object",
            "properties": {
                "scores_file": {"type": "string"},
                "calibration_file": {"type": "string"},
                "pending_reviews": {"type": "integer", "default": 0},
                "thresholds": {"type": "object"}
            },
            "required": ["scores_file", "calibration_file"]
        }),
        display_script: None,
    };

    RapTool {
        descriptor,
        endpoint: "http://localhost:8006/invoke".to_string(),
        http_client: SimpleHttpClient::new(),
        callback_url: None,
    }
}