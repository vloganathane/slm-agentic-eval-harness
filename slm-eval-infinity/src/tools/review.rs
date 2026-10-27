//! Human Review Queue RAP Tool

use infinity_agent_core::tools::rap_tool::{RapTool, RapToolDescriptor};
use rap_client::http::SimpleHttpClient;
use serde_json::json;

pub fn make_review_tool() -> RapTool<SimpleHttpClient> {
    let descriptor = RapToolDescriptor {
        name: "human_review".to_string(),
        description: "Submit cases to human review queue, wait for resolution".to_string(),
        parameters: json!({
            "type": "object",
            "properties": {
                "run_id": {"type": "string"},
                "cases": {"type": "array", "items": {"type": "object"}},
                "reason": {"type": "string", "enum": ["synthetic_injection", "judge_disagreement", "low_confidence"]}
            },
            "required": ["run_id", "cases", "reason"]
        }),
        display_script: None,
    };

    RapTool {
        descriptor,
        endpoint: "http://localhost:8004/invoke".to_string(),
        http_client: SimpleHttpClient::new(),
        callback_url: None,
    }
}