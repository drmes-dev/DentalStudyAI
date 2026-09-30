# Dentora Beta Safety Notes

## Public beta protections

- AI provider keys remain server-side in Render environment variables.
- The public frontend never contains Gemini, Groq, Qwen, Pinecone, or admin secrets.
- CORS is restricted to the configured Dentora frontend origins.
- Public chat and temporary PDF upload endpoints use conservative in-memory rate limits.
- Chat, PDF Tutor, and feedback endpoints accept either the private owner code or the invited-student beta code.
- Owner access is remembered only in the owner's browser local storage; tester access remains session-only.
- Temporary PDF uploads are capped by file size and page count.
- Voice recordings are capped, rate-limited, transcribed through Groq speech-to-text, and are not intentionally persisted by Dentora as files.
- Temporary PDF session content expires automatically and can be explicitly removed.
- Raw persistent RAG library/search endpoints require `X-Dentora-Admin-Key`.
- Persistent ingestion and deletion remain admin-protected.

## Environment variables

Required private-beta control:

```
DENTORA_OWNER_ACCESS_CODE=YOUR_PRIVATE_OWNER_CODE
DENTORA_BETA_ACCESS_CODE=YOUR_SEPARATE_STUDENT_BETA_CODE
```

Keep all three values different. The admin key is server/admin-only; the owner key is for your own trusted browser; the beta code is the only code shared with invited testers.

Optional beta controls:

```
DENTORA_ALLOWED_ORIGINS=https://drmes-dev.github.io
MAX_TEMP_PDF_MB=20
MAX_TEMP_PDF_PAGES=500
SESSION_PDF_TTL_SECONDS=14400
CHAT_RATE_LIMIT=30
CHAT_RATE_WINDOW_SECONDS=600
PDF_RATE_LIMIT=3
PDF_RATE_WINDOW_SECONDS=3600
MAX_VOICE_AUDIO_MB=12
VOICE_RATE_LIMIT=20
VOICE_RATE_WINDOW_SECONDS=3600
```

The in-memory limiter is appropriate for an early single-instance beta. Before a paid or multi-instance production release, replace it with an account-aware/distributed quota system (for example Redis-backed limiting) and add real authentication.
