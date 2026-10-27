//! Web Search RAP Tool

use infinity_agent_core::tools::rap_tool::{RapTool, RapToolDescriptor};
use rap_client::http::SimpleHttpClient;
use serde_json::json;

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
    merchants.into_iter()
        .filter(|(name, _, content)| {
            query_lower.is_empty() || 
            name.to_lowercase().contains(&query_lower) ||
            content.to_lowercase().contains(&query_lower)
        })
        .take(max_results)
        .map(|(title, url, content)| SearchResultItem { title: title.to_string(), url: url.to_string(), content: content.to_string() })
        .collect()
}

#[derive(serde::Serialize)]
struct SearchResultItem {
    title: String,
    url: String,
    content: String,
}

pub fn make_web_search_tool(
    _tavily_key: Option<String>,
    _searxng_url: Option<String>,
    _default_provider: String,
) -> RapTool<SimpleHttpClient> {
    let descriptor = RapToolDescriptor {
        name: "web_search".to_string(),
        description: "Search the web for real-world merchant/item data for synthetic generation".to_string(),
        parameters: json!({
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "max_results": {"type": "integer", "default": 5},
                "provider": {"type": "string", "enum": ["crw", "tavily", "searxng"], "default": "crw"}
            },
            "required": ["query"]
        }),
        display_script: None,
    };

    RapTool {
        descriptor,
        endpoint: "http://localhost:8001/invoke".to_string(),
        http_client: SimpleHttpClient::new(),
        callback_url: None,
    }
}