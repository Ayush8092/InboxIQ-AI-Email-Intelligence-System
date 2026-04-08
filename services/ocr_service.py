"""
OCR service using Google Cloud Vision API.
Extracts text from image attachments in emails.
Integrates into existing pipeline by appending OCR text to email body.
"""
import os
import base64
import json
import requests
from utils.secure_logger import get_secure_logger

logger = get_secure_logger(__name__)

VISION_API_KEY = os.getenv("GOOGLE_VISION_API_KEY", "")
VISION_API_URL = (
    "https://vision.googleapis.com/v1/images:annotate"
)

# Image MIME types we attempt OCR on
_IMAGE_MIME_TYPES = {
    "image/jpeg", "image/jpg", "image/png",
    "image/gif", "image/bmp", "image/webp",
    "image/tiff",
}


def _is_image(mime_type: str) -> bool:
    return mime_type.lower() in _IMAGE_MIME_TYPES


def extract_text_from_image_base64(
    image_base64: str,
    mime_type: str = "image/jpeg",
) -> str:
    """
    Send base64-encoded image to Google Cloud Vision API.
    Returns extracted OCR text or empty string.
    """
    if not VISION_API_KEY:
        logger.warning("GOOGLE_VISION_API_KEY not set — OCR disabled")
        return ""

    try:
        payload = {
            "requests": [
                {
                    "image": {
                        "content": image_base64,
                    },
                    "features": [
                        {
                            "type":       "TEXT_DETECTION",
                            "maxResults": 1,
                        }
                    ],
                }
            ]
        }

        resp = requests.post(
            f"{VISION_API_URL}?key={VISION_API_KEY}",
            json=payload,
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()

        responses = data.get("responses", [])
        if not responses:
            return ""

        full_annotation = responses[0].get("fullTextAnnotation", {})
        text            = full_annotation.get("text", "").strip()

        if text:
            logger.info(f"OCR extracted {len(text)} characters")

        return text

    except requests.exceptions.RequestException as e:
        logger.error(f"Vision API request failed: {type(e).__name__}")
        return ""
    except Exception as e:
        logger.error(f"OCR error: {type(e).__name__}: {e}")
        return ""


def extract_text_from_image_url(image_url: str) -> str:
    """
    Send image URL to Google Cloud Vision API.
    Alternative to base64 for publicly accessible images.
    """
    if not VISION_API_KEY:
        logger.warning("GOOGLE_VISION_API_KEY not set — OCR disabled")
        return ""

    try:
        payload = {
            "requests": [
                {
                    "image": {"source": {"imageUri": image_url}},
                    "features": [{"type": "TEXT_DETECTION", "maxResults": 1}],
                }
            ]
        }
        resp = requests.post(
            f"{VISION_API_URL}?key={VISION_API_KEY}",
            json=payload,
            timeout=15,
        )
        resp.raise_for_status()
        data      = resp.json()
        responses = data.get("responses", [])
        if not responses:
            return ""
        text = responses[0].get("fullTextAnnotation",{}).get("text","").strip()
        return text

    except Exception as e:
        logger.error(f"Vision URL OCR failed: {type(e).__name__}")
        return ""


def _decode_gmail_attachment(part: dict) -> tuple[str, str] | None:
    """
    Extract base64 image data from a Gmail message part.
    Returns (base64_data, mime_type) or None.
    """
    mime_type = part.get("mimeType","")
    if not _is_image(mime_type):
        return None

    body = part.get("body",{})
    data = body.get("data","")

    if data:
        # Inline attachment
        return data, mime_type

    # External attachment (needs separate API call — not supported without token)
    return None


def process_email_attachments_ocr(
    email_body: str,
    gmail_payload: dict | None = None,
) -> str:
    """
    Process all image attachments in a Gmail message payload.
    Appends extracted OCR text to the email body.

    If no Vision API key or no image attachments → returns body unchanged.

    Usage in pipeline:
        enriched_body = process_email_attachments_ocr(
            email["body"],
            gmail_payload=raw_gmail_message["payload"],
        )
        email["body"] = enriched_body
    """
    if not VISION_API_KEY:
        return email_body

    if not gmail_payload:
        return email_body

    ocr_texts = []

    def _process_parts(parts: list):
        for part in parts:
            mime_type = part.get("mimeType","")

            # Recurse into multipart
            if mime_type.startswith("multipart/"):
                sub_parts = part.get("parts",[])
                _process_parts(sub_parts)
                continue

            # Try OCR on image parts
            result = _decode_gmail_attachment(part)
            if result:
                b64_data, mime = result
                filename = part.get("filename","attachment")
                logger.info(f"Running OCR on attachment: {filename} ({mime})")
                text = extract_text_from_image_base64(b64_data, mime)
                if text:
                    ocr_texts.append(
                        f"[OCR from {filename}]\n{text}"
                    )

    # Process top-level parts
    top_parts = gmail_payload.get("parts",[])
    _process_parts(top_parts)

    if ocr_texts:
        appended = "\n\n" + "\n\n".join(ocr_texts)
        logger.info(f"Appended {len(ocr_texts)} OCR text block(s) to email body")
        return email_body + appended

    return email_body