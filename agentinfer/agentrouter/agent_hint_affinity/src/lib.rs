// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project

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

fn trimmed_id(value: Option<&serde_json::Value>) -> Option<String> {
    value?
        .as_str()
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(str::to_string)
}

fn session_id_from_agent_hint(value: &serde_json::Value) -> Option<String> {
    trimmed_id(value.get("agent_hint").and_then(|hint| hint.get("session_id")))
}

/// Existing `session_params.session_id`, compared after trim so `" s-1 "` and
/// `s-1` are the same affinity key.
fn existing_session_id(value: &serde_json::Value) -> Option<String> {
    trimmed_id(
        value
            .get("session_params")
            .and_then(|sp| sp.get("session_id")),
    )
}

fn inject_agent_hint_into_session_params(bytes: &[u8]) -> Option<Vec<u8>> {
    let mut value: serde_json::Value = serde_json::from_slice(bytes).ok()?;
    let desired = existing_session_id(&value).or_else(|| session_id_from_agent_hint(&value))?;
    let current_raw = value
        .pointer("/session_params/session_id")
        .and_then(|v| v.as_str());
    if current_raw == Some(desired.as_str()) {
        return None;
    }

    let obj = value.as_object_mut()?;
    let session_params = match obj.get("session_params") {
        Some(serde_json::Value::Object(_)) => obj.get_mut("session_params")?.as_object_mut()?,
        // Missing or JSON null: create an object so a usable agent_hint can be copied.
        None | Some(serde_json::Value::Null) => {
            obj.insert("session_params".to_string(), serde_json::json!({}));
            obj.get_mut("session_params")?.as_object_mut()?
        }
        Some(_) => return None,
    };
    session_params.insert(
        "session_id".to_string(),
        serde_json::Value::String(desired),
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

    #[test]
    fn trims_both_agent_hint_and_existing_session_id() {
        let from_hint = br#"{"agent_hint":{"session_id":" s-1 "}}"#;
        let out = inject_agent_hint_into_session_params(from_hint).expect("trim hint");
        let value: serde_json::Value = serde_json::from_slice(&out).unwrap();
        assert_eq!(value["session_params"]["session_id"], "s-1");

        let padded = br#"{"session_params":{"session_id":" s-1 "}}"#;
        let out = inject_agent_hint_into_session_params(padded).expect("trim existing");
        let value: serde_json::Value = serde_json::from_slice(&out).unwrap();
        assert_eq!(value["session_params"]["session_id"], "s-1");

        let already = br#"{"agent_hint":{"session_id":" other "},"session_params":{"session_id":"s-1"}}"#;
        assert!(inject_agent_hint_into_session_params(already).is_none());
    }

    #[test]
    fn copies_agent_hint_when_session_params_is_null() {
        let input = br#"{"agent_hint":{"session_id":"s-1"},"session_params":null}"#;
        let out = inject_agent_hint_into_session_params(input).expect("rewrite null");
        let value: serde_json::Value = serde_json::from_slice(&out).unwrap();
        assert_eq!(value["session_params"]["session_id"], "s-1");
    }
}
