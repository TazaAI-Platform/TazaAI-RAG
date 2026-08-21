#!/usr/bin/env bash
# Verify Factiva RAG OAuth + Retrieval from your machine.
set -euo pipefail
cd "$(dirname "$0")/.."
set -a
# shellcheck disable=SC1091
source .env
set +a

echo "== AuthN (RAG) =="
AUTHN=$(curl -sS -X POST "$FACTIVA_TOKEN_URL" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  --data-urlencode "username=$FACTIVA_RAG_USERNAME" \
  --data-urlencode "client_id=$FACTIVA_RAG_CLIENT_ID" \
  --data-urlencode "password=$FACTIVA_RAG_PASSWORD" \
  --data-urlencode "connection=service-account" \
  --data-urlencode "grant_type=password" \
  --data-urlencode "scope=openid service_account_id")
ID_TOKEN=$(python3 -c "import json,sys; print(json.load(sys.stdin)['id_token'])" <<<"$AUTHN")
ATN=$(python3 -c "import json,sys; print(json.load(sys.stdin).get('access_token',''))" <<<"$AUTHN")
echo "AuthN OK (id_token len=${#ID_TOKEN})"

echo "== AuthZ (RAG) =="
AUTHZ=$(curl -sS -X POST "$FACTIVA_TOKEN_URL" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  --data-urlencode "assertion=$ID_TOKEN" \
  --data-urlencode "access_token=$ATN" \
  --data-urlencode "client_id=$FACTIVA_RAG_CLIENT_ID" \
  --data-urlencode "grant_type=urn:ietf:params:oauth:grant-type:jwt-bearer" \
  --data-urlencode "scope=openid pib")
TOKEN=$(python3 -c "import json,sys; print(json.load(sys.stdin)['access_token'])" <<<"$AUTHZ")
echo "AuthZ OK (access_token len=${#TOKEN})"

echo "== Retrieve sample =="
WORK=$(python3 -c "import uuid; print(uuid.uuid4().hex)")
USER=$(python3 -c "import hashlib,os; print(hashlib.md5(os.environ['FACTIVA_RAG_USERNAME'].encode()).hexdigest())")
curl -sS -X POST "$FACTIVA_API_BASE/content/gen-ai/retrieve" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Accept: application/vnd.dowjones.genai-content.v_1.0" \
  -H "Content-Type: application/json" \
  -d "{
    \"data\": {
      \"attributes\": {
        \"response_limit\": 3,
        \"query\": {
          \"value\": \"SoftBank Group AI investments\",
          \"search_filters\": [{\"scope\": \"Language\", \"value\": \"en\"}],
          \"date\": {\"days_range\": \"Last6Months\"}
        },
        \"metrics_data\": {
          \"user_id\": \"$USER\",
          \"work_id\": \"$WORK\",
          \"application_id\": \"taza-rag\"
        }
      },
      \"id\": \"GenAIRetrieval\",
      \"type\": \"genai-content\"
    }
  }" | python3 -m json.tool | head -80

echo
echo "Done."
