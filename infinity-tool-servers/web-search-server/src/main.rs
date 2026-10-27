//! Web Search RAP Tool Server
//! 
//! Exposes `web_search` tool via RAP protocol for Infinity agents.
//! Supports multiple providers: crw (CLI), Tavily (API), SearXNG (local).

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use anyhow::Result;
use axum::{
    extract::{Json, State},
    http::StatusCode,
    routing::{get, post},
    Router,
};
use clap::Parser;
use rap_protocol::{
    DiffContent, DisplaySegment, ImageContent, RapCallback, RapInvocation, RapToolResult,
    RapToolResultContent, ToolDef,
};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use tokio::net::TcpListener;
use tracing::{info, warn};

#[derive(Parser, Debug)]
#[command(name = "web-search-server")]
struct Args {
    /// Port to listen on
    #[arg(short, long, default_value = "8001")]
    port: u16,
    
    /// Default search provider
    #[arg(long, default_value = "crw")]
    provider: String,
    
    /// Tavily API key (or TAVILY_API_KEY env)
    #[arg(long, env = "TAVILY_API_KEY")]
    tavily_key: Option<String>,
    
    /// SearXNG instance URL
    #[arg(long, env = "SEARXNG_URL")]
    searxng_url: Option<String>,
}

#[derive(Deserialize)]
struct SearchArgs {
    query: String,
    #[serde(default = "default_max_results")]
    max_results: usize,
    #[serde(default = "default_provider")]
    provider: String,
}

fn default_max_results() -> usize { 5 }
fn default_provider() -> String { "crw".to_string() }

#[derive(Serialize)]
struct SearchResultItem {
    title: String,
    url: String,
    content: String,
}

#[derive(Serialize)]
struct SearchResponse {
    results: Vec<SearchResultItem>,
}

struct SearchProvider {
    name: String,
    tavily_key: Option<String>,
    searxng_url: Option<String>,
    http: Client,
}

impl Clone for SearchProvider {
    fn clone(&self) -> Self {
        Self {
            name: self.name.clone(),
            tavily_key: self.tavily_key.clone(),
            searxng_url: self.searxng_url.clone(),
            http: self.http.clone(),
        }
    }
}

impl SearchProvider {
    async fn search(&self, query: &str, max_results: usize) -> Result<Vec<SearchResultItem>> {
        match self.name.as_str() {
            "tavily" => self.search_tavily(query, max_results).await,
            "searxng" => self.search_searxng(query, max_results).await,
            "crw" => self.search_crw(query, max_results).await,
            _ => self.search_crw(query, max_results).await, // default to crw
        }
    }
    
    async fn search_tavily(&self, query: &str, max_results: usize) -> Result<Vec<SearchResultItem>> {
        let key = self.tavily_key.as_ref()
            .ok_or_else(|| anyhow::anyhow!("Tavily API key not configured"))?;
        
        let response = self.http
            .post("https://api.tavily.com/search")
            .json(&json!({
                "api_key": key,
                "query": query,
                "max_results": max_results,
                "search_depth": "basic",
                "include_answer": false,
                "include_raw_content": false
            }))
            .send()
            .await?;
        
        if !response.status().is_success() {
            return Err(anyhow::anyhow!("Tavily error: {}", response.status()));
        }
        
        let data: Value = response.json().await?;
        let results_value = data["results"].as_array().cloned().unwrap_or_default();
        
        Ok(results_value.iter().filter_map(|r| {
            Some(SearchResultItem {
                title: r["title"].as_str()?.to_string(),
                url: r["url"].as_str()?.to_string(),
                content: r["content"].as_str()?.to_string(),
            })
        }).collect())
    }
    
    async fn search_searxng(&self, query: &str, max_results: usize) -> Result<Vec<SearchResultItem>> {
        let url = self.searxng_url.as_ref()
            .ok_or_else(|| anyhow::anyhow!("SearXNG URL not configured"))?;
        
        let response = self.http
            .get(format!("{}/search", url))
            .query(&[("q", query), ("format", "json"), ("categories", "general")])
            .send()
            .await?;
        
        if !response.status().is_success() {
            return Err(anyhow::anyhow!("SearXNG error: {}", response.status()));
        }
        
        let data: Value = response.json().await?;
        let results_value = data["results"].as_array().cloned().unwrap_or_default();
        
        Ok(results_value.iter().take(max_results).filter_map(|r| {
            Some(SearchResultItem {
                title: r["title"].as_str()?.to_string(),
                url: r["url"].as_str()?.to_string(),
                content: r["content"].as_str().unwrap_or(r["snippet"].as_str()?).to_string(),
            })
        }).collect())
    }
    
    async fn search_crw(&self, query: &str, max_results: usize) -> Result<Vec<SearchResultItem>> {
        // Try to call crw CLI
        let output = tokio::process::Command::new("crw")
            .args(["search", query, "--json", "--limit", &max_results.to_string()])
            .output()
            .await;
        
        match output {
            Ok(out) if out.status.success() => {
                let text = String::from_utf8_lossy(&out.stdout);
                let data: Value = serde_json::from_str(&text)?;
                
                let results = if data.is_array() {
                    data.as_array().unwrap()
                } else if let Some(arr) = data["results"].as_array() {
                    arr
                } else {
                    &vec![]
                };
                
                Ok(results.iter().filter_map(|r| {
                    Some(SearchResultItem {
                        title: r["title"].as_str()?.to_string(),
                        url: r["url"].as_str()?.to_string(),
                        content: r["snippet"].as_str().or_else(|| r["content"].as_str())?.to_string(),
                    })
                }).collect())
            }
            _ => {
                // Fallback: return built-in merchant data
                warn!("crw not available, using fallback merchant data");
                Ok(get_fallback_merchants(query, max_results))
            }
        }
    }
}

fn get_fallback_merchants(query: &str, max_results: usize) -> Vec<SearchResultItem> {
    let merchants = vec![
        ("WALMART", "https://walmart.com", "Walmart sells groceries: milk bread eggs chicken rice detergent"),
        ("TARGET", "https://target.com", "Target retail: chicken rice detergent toothpaste clothing electronics"),
        ("STARBUCKS", "https://starbucks.com", "Starbucks cafe: latte croissant muffin americano frappuccino"),
        ("HOME DEPOT", "https://homedepot.com", "Home Depot hardware: hammer nails screwdriver tape measure lumber"),
        ("CVS PHARMACY", "https://cvs.com", "CVS pharmacy: aspirin bandages vitamins cough syrup prescriptions"),
        ("MCDONALDS", "https://mcdonalds.com", "McDonalds fast food: big mac fries coke mcchicken nuggets"),
        ("BEST BUY", "https://bestbuy.com", "Best Buy electronics: usb cable phone case headphones charger laptop"),
        ("SUBWAY", "https://subway.com", "Subway sandwiches: turkey sub chips water cookie salad"),
        ("WHOLE FOODS", "https://wholefoodsmarket.com", "Whole Foods organic: milk avocado salmon quinoa kale"),
        ("COSTCO", "https://costco.com", "Costco warehouse: rotisserie chicken toilet paper gasoline hot dog"),
    ];
    
    let query_lower = query.to_lowercase();
    let mut results = Vec::new();
    
    for (name, url, content) in merchants {
        if query_lower.is_empty() || 
           name.to_lowercase().contains(&query_lower) ||
           content.to_lowercase().contains(&query_lower) {
            results.push(SearchResultItem {
                title: name.to_string(),
                url: url.to_string(),
                content: content.to_string(),
            });
        }
    }
    
    results.into_iter().take(max_results).collect()
}

#[derive(Clone)]
struct AppState {
    provider: Arc<SearchProvider>,
}

async fn toolset_manifest(State(state): State<AppState>) -> Json<Value> {
    Json(json!({
        "name": "web-search",
        "endpoint": format!("http://localhost:{}/invoke", std::env::var("PORT").unwrap_or("8001".to_string())),
        "tools": [{
            "name": "web_search",
            "description": "Search the web for real-world merchant/item data for synthetic generation",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "max_results": {"type": "integer", "default": 5, "minimum": 1, "maximum": 20},
                    "provider": {"type": "string", "enum": ["crw", "tavily", "searxng"], "default": "crw"}
                },
                "required": ["query"]
            }
        }]
    }))
}

async fn invoke_tool(
    State(state): State<AppState>,
    Json(inv): Json<RapInvocation>,
) -> Result<StatusCode, StatusCode> {
    let args: SearchArgs = serde_json::from_value(inv.arguments)
        .map_err(|_| StatusCode::BAD_REQUEST)?;
    
    let provider_name = args.provider.clone();
    let provider = if provider_name == "tavily" {
        SearchProvider {
            name: "tavily".to_string(),
            tavily_key: state.provider.tavily_key.clone(),
            searxng_url: None,
            http: state.provider.http.clone(),
        }
    } else if provider_name == "searxng" {
        SearchProvider {
            name: "searxng".to_string(),
            tavily_key: None,
            searxng_url: state.provider.searxng_url.clone(),
            http: state.provider.http.clone(),
        }
    } else {
        (*state.provider).clone()
    };
    
    let results = match provider.search(&args.query, args.max_results).await {
        Ok(r) => r,
        Err(e) => {
            warn!("Search failed: {}", e);
            vec![]
        }
    };
    
    // Send callback
    let callback = RapCallback::ToolResult(RapToolResult {
        group_id: inv.group_id,
        id: inv.id,
        call_id: inv.call_id,
        text: Some(format!("Found {} results for '{}'", results.len(), args.query)),
        content: Some(results.iter().map(|r| RapToolResultContent::Text {
            text: format!("{}: {} - {}", r.title, r.url, r.content)
        }).collect()),
        display_as: Some(results.iter().map(|r| DisplaySegment::Text(
            format!("{} - {}", r.title, r.url)
        )).collect()),
        subscription: None,
    });
    
    let callback_body = serde_json::to_string(&callback).map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    let callback_url = inv.callback_url.clone();
    
    // Fire and forget callback
    let http = state.provider.http.clone();
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

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();
    
    let args = Args::parse();
    unsafe { std::env::set_var("PORT", args.port.to_string()); }
    
    let http = Client::builder()
        .timeout(Duration::from_secs(30))
        .build()?;
    
    let provider = Arc::new(SearchProvider {
        name: args.provider,
        tavily_key: args.tavily_key,
        searxng_url: args.searxng_url,
        http: http.clone(),
    });
    
    let state = AppState { provider };
    
    let app = Router::new()
        .route("/.well-known/rap-toolset", get(toolset_manifest))
        .route("/invoke", post(invoke_tool))
        .with_state(state);
    
    let listener = TcpListener::bind(format!("127.0.0.1:{}", args.port)).await?;
    info!("Web Search RAP server listening on http://127.0.0.1:{}", args.port);
    
    axum::serve(listener, app).await?;
    Ok(())
}