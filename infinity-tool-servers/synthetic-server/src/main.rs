//! Synthetic Generation RAP Tool Server
//! 
//! Exposes `generate_synthetic` tool via RAP protocol for Infinity agents.
//! Generates synthetic cases using web search data + templates.

use std::collections::HashMap;
use std::sync::Arc;

use anyhow::Result;
use axum::{
    extract::{Json, State},
    http::StatusCode,
    routing::{get, post},
    Router,
};
use clap::Parser;
use rap_protocol::{
    RapCallback, RapInvocation, RapToolResult, RapToolResultContent, DisplaySegment,
};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use tokio::net::TcpListener;
use tracing::{info, warn};

#[derive(Parser, Debug)]
#[command(name = "synthetic-server")]
struct Args {
    #[arg(short, long, default_value = "8007")]
    port: u16,
}

#[derive(Deserialize)]
struct SyntheticArgs {
    n_propose: usize,
    merchants: Vec<Value>,
    #[serde(default = "default_threshold")]
    minhash_threshold: f64,
    #[serde(default)]
    use_web: bool,
}

fn default_threshold() -> f64 { 0.6 }

#[derive(Serialize)]
struct SyntheticResponse {
    cases: Vec<Value>,
    source: String,
    merchant_count: usize,
}

#[derive(Clone)]
struct AppState;

async fn toolset_manifest() -> Json<Value> {
    Json(json!({
        "name": "synthetic",
        "endpoint": format!("http://localhost:{}/invoke", std::env::var("PORT").unwrap_or("8007".to_string())),
        "tools": [{
            "name": "generate_synthetic",
            "description": "Generate synthetic cases using web search + templates",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "n_propose": {"type": "integer", "default": 20, "minimum": 1, "maximum": 200},
                    "merchants": {"type": "array", "items": {"type": "object"}},
                    "minhash_threshold": {"type": "number", "default": 0.6, "minimum": 0.0, "maximum": 1.0},
                    "use_web": {"type": "boolean", "default": true}
                },
                "required": ["n_propose"]
            }
        }]
    }))
}

async fn invoke_tool(
    State(_): State<AppState>,
    Json(inv): Json<RapInvocation>,
) -> Result<StatusCode, StatusCode> {
    let args: SyntheticArgs = serde_json::from_value(inv.arguments)
        .map_err(|_| StatusCode::BAD_REQUEST)?;

    let merchants = if args.merchants.is_empty() {
        get_fallback_merchants()
    } else {
        args.merchants
    };

    let merchants_is_empty = merchants.is_empty();
    let cases = generate_cases(args.n_propose, &merchants);
    let source = if args.use_web && !merchants_is_empty { "web" } else { "templates" };

    let response = SyntheticResponse {
        cases,
        source: source.to_string(),
        merchant_count: merchants.len(),
    };

    // Send callback
    let text = format!("Generated {} synthetic cases from {}", response.cases.len(), source);
    let callback = RapCallback::ToolResult(RapToolResult {
        group_id: inv.group_id,
        id: inv.id,
        call_id: inv.call_id,
        text: Some(text.clone()),
        content: Some(vec![RapToolResultContent::Text {
            text: serde_json::to_string(&response).unwrap_or_default(),
        }]),
        display_as: Some(vec![DisplaySegment::Text(text)]),
        subscription: None,
    });

    let callback_body = serde_json::to_string(&callback).map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    let callback_url = inv.callback_url.clone();

    let http = reqwest::Client::new();
    tokio::spawn(async move {
        if let Err(e) = http.post(&callback_url)
            .header("content-type", "application/json")
            .body(callback_body)
            .send()
            .await {
            warn!("Failed to deliver callback: {}", e);
        }
    });

    Ok(StatusCode::OK)
}

fn get_fallback_merchants() -> Vec<Value> {
    vec![
        json!({"name": "WALMART", "category": "grocery", "items": ["milk", "bread", "eggs", "chicken", "rice", "detergent"], "price_range": [1.0, 50.0]}),
        json!({"name": "TARGET", "category": "retail", "items": ["chicken", "rice", "detergent", "toothpaste", "clothing", "electronics"], "price_range": [2.0, 100.0]}),
        json!({"name": "STARBUCKS", "category": "cafe", "items": ["latte", "croissant", "muffin", "americano", "frappuccino"], "price_range": [3.0, 15.0]}),
        json!({"name": "HOME DEPOT", "category": "hardware", "items": ["hammer", "nails", "screwdriver", "tape measure", "lumber"], "price_range": [5.0, 200.0]}),
        json!({"name": "CVS PHARMACY", "category": "pharmacy", "items": ["aspirin", "bandages", "vitamins", "cough syrup"], "price_range": [3.0, 30.0]}),
        json!({"name": "MCDONALDS", "category": "fast_food", "items": ["big mac", "fries", "coke", "mcchicken", "nuggets"], "price_range": [1.0, 10.0]}),
        json!({"name": "BEST BUY", "category": "electronics", "items": ["usb cable", "phone case", "headphones", "charger", "laptop"], "price_range": [10.0, 2000.0]}),
        json!({"name": "SUBWAY", "category": "restaurant", "items": ["turkey sub", "chips", "water", "cookie", "salad"], "price_range": [5.0, 20.0]}),
        json!({"name": "WHOLE FOODS", "category": "grocery", "items": ["milk", "avocado", "salmon", "quinoa", "kale"], "price_range": [3.0, 40.0]}),
        json!({"name": "COSTCO", "category": "warehouse", "items": ["rotisserie chicken", "toilet paper", "gasoline", "hot dog"], "price_range": [1.5, 200.0]}),
    ]
}

fn generate_cases(n: usize, merchants: &[Value]) -> Vec<Value> {
    use rand::seq::SliceRandom;
    use rand::Rng;
    let mut rng = rand::thread_rng();
    let mut cases = Vec::new();

    let templates = vec![
        ("receipt", r#"{{merchant}}
{{date}}
{{#items}}
{{name}}  {{price}}
{{/items}}
TOTAL: {{total}}
THANK YOU"#),
        ("invoice", r#"INVOICE
{{merchant}}
Date: {{date}}
{{#items}}
{{name}} x{{qty}} @ {{unit_price}} = {{price}}
{{/items}}
Subtotal: {{subtotal}}
Tax: {{tax}}
Total: {{total}}"#),
    ];

    for i in 0..n {
        let merchant = merchants.choose(&mut rng).unwrap();
        let merchant_name = merchant["name"].as_str().unwrap_or("UNKNOWN");
        let items_data = merchant["items"].as_array().cloned().unwrap_or_default();
        let price_range_data = merchant["price_range"].as_array().cloned().unwrap_or(vec![json!(1.0), json!(50.0)]);

        let n_items = rng.gen_range(1..=5);
        let mut items = Vec::new();
        let mut total = 0.0;

        for _ in 0..n_items {
            let item = items_data.choose(&mut rng).unwrap().as_str().unwrap_or("item");
            let price = rng.gen_range(price_range_data[0].as_f64().unwrap_or(1.0)..=price_range_data[1].as_f64().unwrap_or(50.0));
            let qty = rng.gen_range(1..=3);
            let line_total = price * qty as f64;
            total += line_total;

            items.push(json!({
                "name": item,
                "qty": qty,
                "unit_price": round2(price),
                "price": round2(line_total),
            }));
        }

        let tax = total * 0.08;
        let subtotal = total;
        let grand_total = total + tax;

        let (doc_type, template) = templates.choose(&mut rng).unwrap();
        let date = format!("2024-{:02}-{:02}", rng.gen_range(1..=12), rng.gen_range(1..=28));

        let mut text = template
            .replace("{{merchant}}", merchant_name)
            .replace("{{date}}", &date)
            .replace("{{total}}", &round2_str(grand_total))
            .replace("{{subtotal}}", &round2_str(subtotal))
            .replace("{{tax}}", &round2_str(tax));

        // Replace items loop
        let items_text = items.iter().map(|it| {
            format!("{}  {}", it["name"].as_str().unwrap_or("item"), round2_str(it["price"].as_f64().unwrap_or(0.0)))
        }).collect::<Vec<_>>().join("\n");
        
        text = text.replace("{{#items}}\n{{name}}  {{price}}\n{{/items}}", &items_text);
        text = text.replace("{{#items}}\n{{name}} x{{qty}} @ {{unit_price}} = {{price}}\n{{/items}}", 
            &items.iter().map(|it| {
                format!("{} x{} @ {} = {}", 
                    it["name"].as_str().unwrap_or("item"),
                    it["qty"].as_u64().unwrap_or(1),
                    round2_str(it["unit_price"].as_f64().unwrap_or(0.0)),
                    round2_str(it["price"].as_f64().unwrap_or(0.0)))
            }).collect::<Vec<_>>().join("\n"));

        let gold = json!({
            "merchant": merchant_name,
            "date": date,
            "total": round2_str(grand_total),
            "subtotal": round2_str(subtotal),
            "tax": round2_str(tax),
            "items": items,
        });

        cases.push(json!({
            "case_id": format!("synth_{}_{}", doc_type, i),
            "text": text,
            "gold": gold,
            "strata": {"quality": "high", "layout": doc_type},
        }));
    }

    cases
}

fn round2(x: f64) -> f64 {
    (x * 100.0).round() / 100.0
}

fn round2_str(x: f64) -> String {
    format!("{:.2}", x)
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    let args = Args::parse();
    unsafe { std::env::set_var("PORT", args.port.to_string()); }

    let state = AppState;

    let app = Router::new()
        .route("/.well-known/rap-toolset", get(toolset_manifest))
        .route("/invoke", post(invoke_tool))
        .with_state(state);

    let listener = TcpListener::bind(format!("127.0.0.1:{}", args.port)).await?;
    info!("Synthetic RAP server listening on http://127.0.0.1:{}", args.port);

    axum::serve(listener, app).await?;
    Ok(())
}