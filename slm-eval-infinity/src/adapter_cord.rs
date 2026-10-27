//! CORD dataset adapter

use anyhow::Result;
use serde::Deserialize;
use serde_json::{json, Value};
use std::path::Path;
use tokio::fs;

#[derive(Deserialize)]
struct CordAnnotation {
    #[serde(rename = "valid_line")]
    valid_line: Option<bool>,
    category: String,
    group: String,
    text: String,
    #[serde(rename = "words")]
    words: Option<Vec<CordWord>>,
}

#[derive(Deserialize)]
struct CordWord {
    text: String,
    #[serde(rename = "x1")]
    x1: f64,
    #[serde(rename = "y1")]
    y1: f64,
    #[serde(rename = "x2")]
    x2: f64,
    #[serde(rename = "y2")]
    y2: f64,
}

#[derive(Deserialize)]
struct CordReceipt {
    #[serde(rename = "image_id")]
    image_id: String,
    annotations: Vec<CordAnnotation>,
}

pub async fn convert_cord_dataset(cord_dir: &Path, output_path: &Path) -> Result<()> {
    let annotations_path = cord_dir.join("annotations.json");
    let content = fs::read_to_string(&annotations_path).await?;
    let receipts: Vec<CordReceipt> = serde_json::from_str(&content)?;

    let mut cases = Vec::new();

    for receipt in receipts {
        let mut merchant = None;
        let mut date = None;
        let mut total = None;
        let mut subtotal = None;
        let mut tax = None;
        let mut items = Vec::new();

        let mut full_text = String::new();

        for ann in &receipt.annotations {
            full_text.push_str(&ann.text);
            full_text.push('\n');

            match ann.category.as_str() {
                "store_name" | "merchant" => merchant = Some(ann.text.trim().to_string()),
                "date" => date = Some(ann.text.trim().to_string()),
                "total" | "grand_total" => total = Some(ann.text.trim().to_string()),
                "subtotal" | "sub_total" => subtotal = Some(ann.text.trim().to_string()),
                "tax" | "vat" => tax = Some(ann.text.trim().to_string()),
                "menu_item" | "item" => {
                    if let Some(price) = extract_price(&ann.text) {
                        items.push(json!({
                            "name": ann.text.trim(),
                            "qty": 1,
                            "unit_price": price,
                            "price": price,
                        }));
                    }
                }
                _ => {}
            }
        }

        // Determine quality from image_id or annotations
        let quality = if receipt.annotations.iter().any(|a| a.valid_line == Some(true)) {
            "high"
        } else if receipt.annotations.len() > 10 {
            "medium"
        } else {
            "low"
        };

        let gold = json!({
            "merchant": merchant,
            "date": date,
            "total": total,
            "subtotal": subtotal,
            "tax": tax,
            "items": items,
        });

        let strata = json!({
            "quality": quality,
            "layout": "receipt",
        });

        let case = json!({
            "case_id": format!("cord_{}", receipt.image_id),
            "text": full_text.trim(),
            "gold": gold,
            "strata": strata,
            "ocr_signal": "ok",
        });

        cases.push(case);
    }

    // Write JSONL
    let mut out = String::new();
    for case in &cases {
        out.push_str(&serde_json::to_string(case)?);
        out.push('\n');
    }

    fs::write(output_path, out).await?;
    println!("Converted {} CORD receipts to {}", cases.len(), output_path.display());
    Ok(())
}

fn extract_price(text: &str) -> Option<String> {
    // Simple price extraction
    let re = regex::Regex::new(r"[\$\£\€]?\s*(\d+[.,]\d{2})").ok()?;
    re.captures(text).and_then(|c| c.get(1)).map(|m| m.as_str().replace(',', "."))
}