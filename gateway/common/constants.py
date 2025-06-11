"""Common constants used in the gateway."""

DEFAULT_API_URL = "https://explorer.invariantlabs.ai"

IGNORED_HEADERS = [
    "accept-encoding",
    "host",
    "invariant-authorization",
    "x-forwarded-for",
    "x-forwarded-host",
    "x-forwarded-port",
    "x-forwarded-proto",
    "x-forwarded-server",
    "x-real-ip",
]

CLIENT_TIMEOUT = 60.0

CONTENT_TYPE_HEADER = "content-type"
CONTENT_TYPE_JSON = "application/json"
CONTENT_TYPE_EVENT_STREAM = "text/event-stream"
