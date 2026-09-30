# D1 question bank setup

Books and semantic retrieval continue to use Pinecone. D1 stores question and
paper metadata only; it does not store 1024-dimensional dummy vectors.

Create a D1 database on Workers Free. Create an API token with D1 read/write
access scoped to the account that owns the database. Store the token only in
the Render backend environment, never in frontend JavaScript or GitHub.

Set these Render environment variables together:

| Name | Value |
| --- | --- |
| `QUESTION_BANK_STORAGE` | `d1` |
| `CLOUDFLARE_ACCOUNT_ID` | Account ID, 32 hexadecimal characters |
| `CLOUDFLARE_D1_DATABASE_ID` | Database UUID |
| `CLOUDFLARE_API_TOKEN` | Scoped D1 API token |

Keep the existing Pinecone settings. On the first question-bank request, Dentora
creates its D1 tables, copies existing questions and paper manifests, and compares
every copied record's metadata with the source before marking migration complete.
Answers, verification state, source IDs, and import progress are preserved.
Pinecone records are not deleted. A failed copy raises an error and can be retried;
it is never reported as an empty or successfully migrated bank.

After successful migration, a persistent D1 marker prevents downloading the old
Pinecone question bank again after process restarts. New bank writes use D1.
Check `/health` for `question_bank_storage: "d1"`, then verify the owner catalog,
paper progress, and a test through the existing frontend.

Rollback: set `QUESTION_BANK_STORAGE=pinecone` and redeploy. The old bank remains,
but questions or answer updates made after D1 activation will remain only in D1;
export/merge those changes before a permanent rollback.

D1 Free is finite: 500 MB per database, 5 GB across ten databases, 5 million rows
read and 100,000 rows written per day. No D1 egress charges apply. Daily quota
errors are surfaced without switching writes into another storage provider.
Pinecone's separate egress allowance still applies to books/RAG retrieval.

Sources: https://developers.cloudflare.com/d1/platform/pricing/ and
https://developers.cloudflare.com/d1/platform/limits/.
