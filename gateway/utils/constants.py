"""Common constants used in the gateway."""

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
INVARIANT_AUTHORIZATION_HEADER = "invariant-authorization"
