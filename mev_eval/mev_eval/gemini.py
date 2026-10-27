"""Minimal Gemini REST client used as the MEV-Eval judge.

Configuration comes from environment variables (see README). Defaults match the
setup used for the paper: Gemini 2.5 Pro, temperature 0, thinking budget 128.
"""
from __future__ import annotations

import ast
import json
import logging
import mimetypes
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests

DEFAULT_API_BASE_URL = "https://generativelanguage.googleapis.com"
DEFAULT_MODEL_NAME = "models/gemini-2.5-pro"
# Kept verbatim from the judging runs reported in the paper.
DEFAULT_SYSTEM_INSTRUCTION = (
    "You are a careful video annotation assistant. "
    "Return strict JSON only and prefer high-confidence visual facts."
)
RETRYABLE_STATUS = {429, 500, 502, 503, 504}


class GeminiBlockedContentError(RuntimeError):
    def __init__(self, block_reason: str, response_payload: dict[str, Any]):
        self.block_reason = (block_reason or "").strip() or "UNKNOWN"
        self.response_payload = response_payload
        response_snippet = json.dumps(response_payload, ensure_ascii=False)[:2000]
        super().__init__(
            f"Gemini request blocked by safety filter. "
            f"block_reason={self.block_reason} response={response_snippet}"
        )


def normalize_model_name(value: str | None) -> str:
    cleaned = (value or "").strip()
    if not cleaned:
        return DEFAULT_MODEL_NAME
    if cleaned.startswith("models/"):
        return cleaned
    return f"models/{cleaned}"


def resolve_thinking_budget(model_name: str, raw_value: str | None) -> int | None:
    if raw_value is not None and raw_value.strip() != "":
        return int(raw_value)
    if normalize_model_name(model_name) == "models/gemini-2.5-pro":
        return 128
    return None


def env_flag(raw_value: str | None, default: bool = True) -> bool:
    if raw_value is None:
        return default
    return raw_value.strip().lower() not in {"0", "false", "no", "off"}


def extract_json_text(raw_text: str) -> str:
    """Pull the first complete JSON object or array out of a model response."""
    text = raw_text.strip()
    fenced_matches = re.findall(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    candidates = [match.strip() for match in fenced_matches if match.strip()]
    candidates.append(text)
    for candidate in candidates:
        if candidate.startswith("{") or candidate.startswith("["):
            return candidate
    for open_char, close_char in (("{", "}"), ("[", "]")):
        start = text.find(open_char)
        if start == -1:
            continue
        depth = 0
        in_string = False
        escape = False
        for index in range(start, len(text)):
            char = text[index]
            if in_string:
                if escape:
                    escape = False
                elif char == "\\":
                    escape = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
                continue
            if char == open_char:
                depth += 1
            elif char == close_char:
                depth -= 1
                if depth == 0:
                    return text[start : index + 1]
    raise ValueError("Failed to extract a JSON object from the model response.")


def parse_json_payload(raw_text: str) -> dict[str, Any]:
    json_text = extract_json_text(raw_text)
    try:
        payload = json.loads(json_text)
    except json.JSONDecodeError:
        try:
            payload = ast.literal_eval(json_text)
        except (SyntaxError, ValueError) as exc:
            snippet = raw_text.strip().replace("\n", " ")[:500]
            raise ValueError(f"Model did not return valid JSON. Snippet: {snippet}") from exc
    if isinstance(payload, list):
        if len(payload) == 1 and isinstance(payload[0], dict):
            return payload[0]
        raise ValueError(f"Expected a JSON object, received list with {len(payload)} items.")
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object, received {type(payload).__name__}.")
    return payload


def guess_mime_type(path: Path) -> str:
    mime_type, _ = mimetypes.guess_type(str(path))
    if mime_type:
        return mime_type
    return {
        ".mp4": "video/mp4",
        ".mov": "video/quicktime",
        ".mkv": "video/x-matroska",
        ".avi": "video/x-msvideo",
        ".webm": "video/webm",
    }.get(path.suffix.lower(), "application/octet-stream")


def extract_response_text(payload: dict[str, Any]) -> str:
    texts = []
    for candidate in payload.get("candidates", []):
        for part in candidate.get("content", {}).get("parts", []):
            text = part.get("text")
            if text:
                texts.append(str(text).strip())
    return "\n".join(part for part in texts if part)


def extract_finish_reason(payload: dict[str, Any]) -> str:
    candidates = payload.get("candidates", [])
    if not candidates:
        return ""
    return str(candidates[0].get("finishReason", "")).strip()


def extract_block_reason(payload: dict[str, Any]) -> str:
    prompt_feedback = payload.get("promptFeedback", {})
    if not isinstance(prompt_feedback, dict):
        return ""
    return str(prompt_feedback.get("blockReason", "")).strip()


def looks_like_detection_output(text: str) -> bool:
    lowered = text.lower()
    markers = ("box_2d", '"bbox"', '"label"', '"labels"', '"tracking_id"', '"polygon"', '"mask"', '"score"')
    return any(marker in lowered for marker in markers)


@dataclass
class UploadedFile:
    name: str
    uri: str
    mime_type: str
    state: str


class GeminiClient:
    """Upload a video to the Gemini Files API and ask for a strict-JSON judgment."""

    def __init__(self, temperature: float = 0.0) -> None:
        self.api_key = (
            os.environ.get("GEMINI_API_KEY")
            or os.environ.get("GOOGLE_API_KEY")
            or os.environ.get("GEMINI_API_TOKEN")
            or ""
        ).strip()
        if not self.api_key:
            raise RuntimeError("Missing Gemini API key. Set GEMINI_API_KEY or GOOGLE_API_KEY.")

        self.session = requests.Session()
        self.temperature = temperature
        self.model_name = normalize_model_name(os.environ.get("GEMINI_MODEL"))
        self.api_base_url = os.environ.get("GEMINI_API_BASE_URL", DEFAULT_API_BASE_URL).rstrip("/")
        self.system_instruction = os.environ.get("GEMINI_SYSTEM_INSTRUCTION", DEFAULT_SYSTEM_INSTRUCTION).strip()
        self.thinking_budget = resolve_thinking_budget(self.model_name, os.environ.get("GEMINI_THINKING_BUDGET"))
        self.http_timeout = float(os.environ.get("GEMINI_HTTP_TIMEOUT", "600"))
        self.http_retry_count = max(1, int(os.environ.get("GEMINI_HTTP_RETRY_COUNT", "32")))
        self.http_retry_base_delay = max(0.5, float(os.environ.get("GEMINI_HTTP_RETRY_BASE_DELAY", "10")))
        self.http_retry_max_delay = max(
            self.http_retry_base_delay,
            float(os.environ.get("GEMINI_HTTP_RETRY_MAX_DELAY", "120")),
        )
        self.file_poll_interval = float(os.environ.get("GEMINI_FILE_POLL_INTERVAL", "5"))
        self.file_poll_timeout = float(os.environ.get("GEMINI_FILE_POLL_TIMEOUT", "1800"))
        self.cleanup_uploaded_files = env_flag(os.environ.get("GEMINI_CLEANUP_FILES"))
        self.json_retry_count = max(1, int(os.environ.get("GEMINI_JSON_RETRY_COUNT", "3")))
        self.retry_extra_tokens = max(128, int(os.environ.get("GEMINI_RETRY_EXTRA_TOKENS", "512")))
        self.max_retry_tokens = max(1024, int(os.environ.get("GEMINI_MAX_RETRY_TOKENS", "8192")))
        self.upload_cache: dict[Path, UploadedFile] = {}

        logging.info(
            "Initialized Gemini judge model=%s thinking_budget=%s temperature=%s",
            self.model_name,
            self.thinking_budget,
            self.temperature,
        )

    # -- HTTP ---------------------------------------------------------------

    def _retry_delay(self, attempt: int, response: requests.Response | None = None) -> float:
        if response is not None:
            retry_after = response.headers.get("Retry-After", "").strip()
            if retry_after:
                try:
                    return min(float(retry_after), self.http_retry_max_delay)
                except ValueError:
                    pass
        delay = self.http_retry_base_delay * (2 ** max(attempt - 1, 0))
        return min(delay, self.http_retry_max_delay)

    def _request(
        self,
        method: str,
        url: str,
        *,
        json_payload: dict[str, Any] | None = None,
        data: Any = None,
        headers: dict[str, str] | None = None,
        expected_status: int | None = None,
        include_api_key: bool = True,
    ) -> requests.Response:
        """Send a request, retrying network errors, 429, and 5xx with exponential backoff."""
        for attempt in range(1, self.http_retry_count + 1):
            try:
                response = self.session.request(
                    method=method,
                    url=url,
                    params={"key": self.api_key} if include_api_key else None,
                    json=json_payload,
                    data=data,
                    headers=dict(headers or {}),
                    timeout=self.http_timeout,
                )
            except requests.RequestException as exc:
                if attempt == self.http_retry_count:
                    raise RuntimeError(f"{method} {url} failed after {attempt} attempts: {exc}") from exc
                delay = self._retry_delay(attempt)
                logging.warning(
                    "%s %s failed with %s on attempt %d/%d. Retrying in %.1fs.",
                    method, url, exc.__class__.__name__, attempt, self.http_retry_count, delay,
                )
                time.sleep(delay)
                continue

            ok = response.status_code == expected_status if expected_status is not None else response.ok
            if ok:
                return response
            if response.status_code in RETRYABLE_STATUS and attempt < self.http_retry_count:
                delay = self._retry_delay(attempt, response)
                logging.warning(
                    "%s %s returned retryable status %d on attempt %d/%d. Retrying in %.1fs.",
                    method, url, response.status_code, attempt, self.http_retry_count, delay,
                )
                time.sleep(delay)
                continue
            raise RuntimeError(f"{method} {url} failed with {response.status_code}: {response.text.strip()}")
        raise RuntimeError(f"{method} {url} failed without a response.")

    def _request_json(self, method: str, url: str, **kwargs: Any) -> dict[str, Any]:
        response = self._request(method, url, **kwargs)
        return response.json() if response.content else {}

    # -- Files API ----------------------------------------------------------

    def _upload_file(self, path: Path) -> UploadedFile:
        resolved = path.expanduser().resolve()
        cached = self.upload_cache.get(resolved)
        if cached is not None:
            return cached
        if not resolved.exists():
            raise FileNotFoundError(f"Video file does not exist: {resolved}")

        mime_type = guess_mime_type(resolved)
        # Read into memory so a retried upload resends the full body.
        content = resolved.read_bytes()
        start_response = self._request(
            "POST",
            f"{self.api_base_url}/upload/v1beta/files",
            json_payload={"file": {"display_name": resolved.name}},
            headers={
                "X-Goog-Upload-Protocol": "resumable",
                "X-Goog-Upload-Command": "start",
                "X-Goog-Upload-Header-Content-Length": str(len(content)),
                "X-Goog-Upload-Header-Content-Type": mime_type,
                "Content-Type": "application/json",
            },
        )
        upload_url = (start_response.headers.get("X-Goog-Upload-URL") or "").strip()
        if not upload_url:
            raise RuntimeError("Gemini Files API did not return an upload URL.")

        upload_payload = self._request_json(
            "POST",
            upload_url,
            data=content,
            headers={
                "Content-Length": str(len(content)),
                "X-Goog-Upload-Offset": "0",
                "X-Goog-Upload-Command": "upload, finalize",
            },
            include_api_key=False,
        )
        uploaded = self._wait_until_active(self._coerce_uploaded_file(upload_payload, mime_type))
        self.upload_cache[resolved] = uploaded
        return uploaded

    @staticmethod
    def _coerce_uploaded_file(payload: dict[str, Any], fallback_mime_type: str) -> UploadedFile:
        file_payload = payload["file"] if isinstance(payload.get("file"), dict) else payload
        name = str(file_payload.get("name", "")).strip()
        if not name:
            raise RuntimeError(f"Gemini file upload/get response did not contain a file name: {payload}")
        state = file_payload.get("state")
        if isinstance(state, dict):
            state = state.get("name")
        return UploadedFile(
            name=name,
            uri=str(file_payload.get("uri", "")).strip(),
            mime_type=str(file_payload.get("mimeType", "") or file_payload.get("mime_type", "")).strip()
            or fallback_mime_type,
            state=str(state or "").upper(),
        )

    def _wait_until_active(self, uploaded: UploadedFile) -> UploadedFile:
        current = uploaded
        deadline = time.time() + self.file_poll_timeout
        while True:
            if current.uri and current.state in {"", "ACTIVE"}:
                return current
            if current.state not in {"PROCESSING", "UNSPECIFIED", "STATE_UNSPECIFIED", ""}:
                raise RuntimeError(f"Gemini file {current.name} entered unexpected state {current.state}.")
            if time.time() >= deadline:
                raise TimeoutError(f"Timed out waiting for Gemini file {current.name} to become ACTIVE.")
            time.sleep(self.file_poll_interval)
            payload = self._request_json("GET", f"{self.api_base_url}/v1beta/{current.name}")
            current = self._coerce_uploaded_file(payload, current.mime_type)

    # -- Generation ---------------------------------------------------------

    def _build_generation_config(self, max_new_tokens: int, *, force_deterministic: bool = False) -> dict[str, Any]:
        config: dict[str, Any] = {
            "temperature": 0.0 if force_deterministic else self.temperature,
            "maxOutputTokens": max_new_tokens,
            "responseMimeType": "application/json",
        }
        if self.thinking_budget is not None:
            config["thinkingConfig"] = {"thinkingBudget": self.thinking_budget}
        return config

    @staticmethod
    def _build_retry_prompt(prompt: str, retry_reason: str, attempt: int) -> str:
        return (
            f"{prompt}\n\n"
            "IMPORTANT RETRY INSTRUCTIONS:\n"
            f"- The previous response was invalid because: {retry_reason}.\n"
            f"- This is retry attempt {attempt}.\n"
            "- Return exactly one complete top-level JSON object matching the requested schema.\n"
            "- Do not return a JSON array or a list of events.\n"
            "- If you previously returned one item per clip or one item per event, merge them into one global object that matches the requested schema.\n"
            "- Do not return object-detection output, boxes, coordinates, labels-only results, OCR blocks, or grounding results.\n"
            "- Never emit keys such as box_2d, bbox, label, labels, score, mask, polygon, or tracking_id unless the requested schema explicitly asks for them.\n"
            "- Do not include markdown fences, explanations, or any extra text.\n"
            "- Make sure every string is closed and the JSON is syntactically complete.\n"
            "- Keep field values concise enough to finish within the token budget.\n"
        )

    def _generate_once(
        self,
        video_paths: list[Path],
        prompt: str,
        max_new_tokens: int,
        force_deterministic: bool,
    ) -> dict[str, Any]:
        parts: list[dict[str, Any]] = []
        for path in video_paths:
            uploaded = self._upload_file(path)
            parts.append({"file_data": {"mime_type": uploaded.mime_type, "file_uri": uploaded.uri}})
        parts.append({"text": prompt})
        request_payload = {
            "systemInstruction": {"parts": [{"text": self.system_instruction}]},
            "contents": [{"role": "user", "parts": parts}],
            "generationConfig": self._build_generation_config(
                max_new_tokens, force_deterministic=force_deterministic
            ),
            "store": False,
        }
        return self._request_json(
            "POST",
            f"{self.api_base_url}/v1beta/{self.model_name}:generateContent",
            json_payload=request_payload,
        )

    def generate_json(self, video_paths: list[Path], prompt: str, max_new_tokens: int) -> dict[str, Any]:
        """Return the judge's JSON object, re-prompting (deterministically) on invalid output."""
        current_max_tokens = max_new_tokens
        retry_reason = "initial invalid JSON response"
        last_error: Exception | None = None
        last_response_text = ""
        last_finish_reason = ""

        for attempt in range(1, self.json_retry_count + 1):
            attempt_prompt = prompt if attempt == 1 else self._build_retry_prompt(prompt, retry_reason, attempt)
            response_payload = self._generate_once(
                video_paths, attempt_prompt, current_max_tokens, force_deterministic=attempt > 1
            )
            finish_reason = extract_finish_reason(response_payload)
            block_reason = extract_block_reason(response_payload)
            output_text = extract_response_text(response_payload)
            last_response_text = output_text
            last_finish_reason = finish_reason

            if "PROHIBITED_CONTENT" in (block_reason, finish_reason):
                raise GeminiBlockedContentError(block_reason or finish_reason, response_payload)

            if not output_text:
                retry_reason = f"empty text response (finishReason={finish_reason or '<empty>'})"
                last_error = RuntimeError(
                    "Gemini generateContent did not return a text payload. "
                    f"finish_reason={finish_reason or '<empty>'} "
                    f"response={json.dumps(response_payload, ensure_ascii=False)[:2000]}"
                )
            else:
                try:
                    return parse_json_payload(output_text)
                except Exception as exc:
                    logging.warning(
                        "Gemini returned invalid JSON on attempt %d/%d (finish_reason=%s, max_tokens=%d). Snippet: %s",
                        attempt,
                        self.json_retry_count,
                        finish_reason or "<empty>",
                        current_max_tokens,
                        output_text.strip().replace("\n", " ")[:500],
                    )
                    if looks_like_detection_output(output_text):
                        retry_reason = (
                            "You returned object-detection or grounding JSON with boxes/labels instead of the requested schema."
                        )
                    else:
                        retry_reason = f"{type(exc).__name__}: {exc}"
                    last_error = exc

            if finish_reason == "MAX_TOKENS":
                current_max_tokens = min(self.max_retry_tokens, current_max_tokens + self.retry_extra_tokens)

        raise RuntimeError(
            "Gemini failed to return valid JSON after retries. "
            f"finish_reason={last_finish_reason or '<empty>'} "
            f"last_error={last_error} "
            f"last_snippet={last_response_text.strip().replace(chr(10), ' ')[:500]}"
        )

    def close(self) -> None:
        """Delete uploaded videos from the Files API unless GEMINI_CLEANUP_FILES=0."""
        if self.cleanup_uploaded_files:
            for file_name in {uploaded.name for uploaded in self.upload_cache.values() if uploaded.name}:
                try:
                    self._request("DELETE", f"{self.api_base_url}/v1beta/{file_name}", expected_status=200)
                except Exception as exc:
                    logging.warning("Failed to delete Gemini file %s: %s", file_name, exc)
        self.upload_cache.clear()
