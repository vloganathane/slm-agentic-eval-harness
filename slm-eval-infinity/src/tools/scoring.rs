//! Scoring RAP Tool

use infinity_agent_core::tools::rap_tool::{RapTool, RapToolDescriptor};
use rap_client::http::SimpleHttpClient;
use serde_json::json;

pub fn make_scoring_tool() -> RapTool<SimpleHttpClient> {
    let descriptor = RapToolDescriptor {
        name: "score_cases".to_string(),
        description: "Score predictions against gold using element-level micro-F1 + bootstrap CI".to_string(),
        parameters: json!({
            "type": "object",
            "properties": {
                "predictions_file": {"type": "string"},
                "gold_file": {"type": "string"},
                "bootstrap_resamples": {"type": "integer", "default": 1000}
            },
            "required": ["predictions_file", "gold_file"]
        }),
        display_script: None,
    };

    RapTool {
        descriptor,
        endpoint: "http://localhost:8005/invoke".to_string(),
        http_client: SimpleHttpClient::new(),
        callback_url: None,
    }
}