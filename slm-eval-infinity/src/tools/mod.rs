//! RAP Tool wrappers for slm-eval

pub mod web_search;
pub mod inference;
pub mod minhash;
pub mod review;
pub mod scoring;
pub mod gate;
pub mod synthetic;

// Re-export tool constructors
pub use web_search::make_web_search_tool;
pub use inference::make_inference_tool;
pub use minhash::make_minhash_tool;
pub use review::make_review_tool;
pub use scoring::make_scoring_tool;
pub use gate::make_gate_tool;
pub use synthetic::make_synthetic_tool;