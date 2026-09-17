//! `agent_hint_affinity` WASM guest for upstream vLLM Router.
//!
//! Implements `vllm:router-middleware@0.1.0` (HTTP envelope WIT). Copies a
//! non-empty `agent_hint.session_id` into `session_params.session_id` when no
//! usable session id is already present.

#![allow(clippy::missing_errors_doc)]

wit_bindgen::generate!({
    path: "wit",
    world: "middleware",
});

use exports::vllm::router_middleware::on_request::Guest;
use vllm::router_middleware::types::{Action, ModifyAction, Request};

struct Component;

fn body_has_session_params_session_id(value: &serde_json::Value) -> bool {
    value
        .get("session_params")
        .and_then(|sp| sp.get("session_id"))
        .and_then(|v| v.as_str())
        .map(|s| !s.trim().is_empty())
        .unwrap_or(false)
}

fn session_id_from_agent_hint(value: &serde_json::Value) -> Option<String> {
    value
        .get("agent_hint")?
        .get("session_id")?
        .as_str()
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(|s| s.to_string())
}

fn inject_agent_hint_into_session_params(bytes: &[u8]) -> Option<Vec<u8>> {
    let mut value: serde_json::Value = serde_json::from_slice(bytes).ok()?;
    if body_has_session_params_session_id(&value) {
        return None;
    }
    let session_id = session_id_from_agent_hint(&value)?;
    let obj = value.as_object_mut()?;
    let session_params = obj
        .entry("session_params")
        .or_insert_with(|| serde_json::json!({}));
    let session_params = session_params.as_object_mut()?;
    session_params.insert(
        "session_id".to_string(),
        serde_json::Value::String(session_id),
    );
    serde_json::to_vec(&value).ok()
}

impl Guest for Component {
    fn handle(req: Request) -> Action {
        match inject_agent_hint_into_session_params(&req.body) {
            Some(body) => Action::Modify(ModifyAction {
                headers_set: Vec::new(),
                headers_add: Vec::new(),
                headers_remove: Vec::new(),
                body_replace: Some(body),
            }),
            None => Action::Continue,
        }
    }
}

export!(Component);

#[cfg(test)]
mod tests {
    use super::inject_agent_hint_into_session_params;

    #[test]
    fn copies_agent_hint_session_id_when_missing() {
        let input = br#"{"agent_hint":{"session_id":"s-1"},"messages":[]}"#;
        let out = inject_agent_hint_into_session_params(input).expect("rewrite");
        let value: serde_json::Value = serde_json::from_slice(&out).unwrap();
        assert_eq!(value["session_params"]["session_id"], "s-1");
    }

    #[test]
    fn continues_when_session_params_already_set() {
        let input = br#"{"agent_hint":{"session_id":"s-1"},"session_params":{"session_id":"keep"}}"#;
        assert!(inject_agent_hint_into_session_params(input).is_none());
    }

    #[test]
    fn continues_without_agent_hint() {
        let input = br#"{"messages":[]}"#;
        assert!(inject_agent_hint_into_session_params(input).is_none());
    }
}
