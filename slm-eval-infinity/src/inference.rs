use anyhow::Result;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashSet;
use std::time::Duration;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAIMessage {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAIRequest {
    pub model: String,
    pub messages: Vec<OpenAIMessage>,
    pub temperature: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_format: Option<Value>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct OpenAIResponse {
    pub choices: Vec<OpenAIChoice>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct OpenAIChoice {
    pub message: OpenAIMessage,
}

#[derive(Debug, Clone, Deserialize)]
pub struct OpenAIErrorResponse {
    pub error: OpenAIError,
}

#[derive(Debug, Clone, Deserialize)]
pub struct OpenAIError {
    pub message: String,
}

pub async fn infer_case(
    client: &Client,
    base_url: &str,
    model: &str,
    timeout_seconds: usize,
    max_attempts: usize,
    backoff_base: f64,
    text: &str,
    exact_fields: &[String],
    subjective_fields: &[String],
) -> Result<Value> {
    let fields: Vec<String> = exact_fields.iter()
        .chain(subjective_fields.iter())
        .cloned()
        .collect::<HashSet<_>>()
        .into_iter()
        .collect();

    let prompt = format!(
        r#"Extract the following fields from this receipt/invoice.
Return ONLY a JSON object with exactly these keys.
Use an empty string for a field that is not present in the text. Do not invent values.

Fields: {}

Text:
{}

JSON:"#,
        fields.join(", "),
        text
    );

    let request = OpenAIRequest {
        model: model.to_string(),
        messages: vec![OpenAIMessage {
            role: "user".to_string(),
            content: prompt,
        }],
        temperature: 0.0,
        response_format: Some(serde_json::json!({"type": "json_object"})),
    };

    let url = format!("{}/chat/completions", base_url.trim_end_matches('/'));
    
    let mut last_err = None;
    for attempt in 0..max_attempts {
        let response = client
            .post(&url)
            .timeout(Duration::from_secs(timeout_seconds as u64))
            .json(&request)
            .send()
            .await;

        match response {
            Ok(resp) => {
                if resp.status().is_success() {
                    let openai_resp: OpenAIResponse = resp.json().await?;
                    if let Some(choice) = openai_resp.choices.first() {
                        let content = choice.message.content.trim();
                        match serde_json::from_str::<Value>(content) {
                            Ok(obj) => {
                                // Ensure all wanted fields are present
                                let mut result = serde_json::Map::new();
                                for field in &fields {
                                    result.insert(field.clone(), obj.get(field).cloned().unwrap_or(Value::String(String::new())));
                                }
                                return Ok(Value::Object(result));
                            }
                            Err(_) => {
                                last_err = Some(anyhow::anyhow!("model returned malformed JSON"));
                            }
                        }
                    }
                } else {
                    let status = resp.status();
                    let err_resp: Result<OpenAIErrorResponse, _> = resp.json().await;
                    if let Ok(err_resp) = err_resp {
                        last_err = Some(anyhow::anyhow!("API error: {}", err_resp.error.message));
                    } else {
                        last_err = Some(anyhow::anyhow!("API error: {}", status));
                    }
                }
            }
            Err(e) => {
                last_err = Some(anyhow::anyhow!("Request failed: {}", e));
            }
        }

        // Exponential backoff
        if attempt < max_attempts - 1 {
            let delay = Duration::from_millis((backoff_base.powi(attempt as i32) * 1000.0) as u64);
            tokio::time::sleep(delay).await;
        }
    }

    Err(last_err.unwrap_or_else(|| anyhow::anyhow!("Max attempts reached")))
}

pub async fn run_inference(
    client: &Client,
    base_url: &str,
    model: &str,
    timeout_seconds: usize,
    max_attempts: usize,
    backoff_base: f64,
    cases: &[crate::models::Case],
    exact_fields: &[String],
    subjective_fields: &[String],
) -> Result<Vec<crate::models::Prediction>> {
    let mut predictions = Vec::new();
    
    for (i, case) in cases.iter().enumerate() {
        eprintln!("  [{}/{}] {}", i + 1, cases.len(), case.case_id);
        let prediction = infer_case(
            client, base_url, model, timeout_seconds, max_attempts, backoff_base,
            &case.text, exact_fields, subjective_fields
        ).await?;
        
        predictions.push(crate::models::Prediction {
            case_id: case.case_id.clone(),
            fields: match prediction {
                Value::Object(map) => map.into_iter().collect(),
                _ => std::collections::HashMap::new(),
            },
        });
    }
    
    Ok(predictions)
}