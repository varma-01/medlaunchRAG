import boto3
import json

client = boto3.client("lambda")

def pretty_print_citation(resp):
    print("\n" + "="*80)
    print(f"📖 CITATION MODE — Chapter {resp.get('chapter')}")
    print("="*80)

    # Not found?
    if resp.get("found") is False:
        print(f"❌ {resp.get('error')}")
        return

    print(f"Document: {resp['metadata'].get('document', 'N/A')}")
    print(f"Section:  {resp['metadata'].get('section', 'N/A')}")
    print(f"Chunk ID: {resp.get('chunk_id')}")

    print("\n--- EXACT TEXT ----------------------------------------------\n")
    print(resp.get("exact_text", ""))
    print("\n--------------------------------------------------------------")
    print(f"Retrieved at: {resp.get('timestamp')}")
    print("="*80 + "\n")


def pretty_print_question(resp):
    print("\n" + "="*80)
    print("💬 QUESTION MODE — RAG Answer")
    print("="*80)

    print(f"\n❓ Query: {resp.get('query')}\n")
    print("📝 Answer:\n")
    print(resp.get("answer", ""))

    print("\n🔎 Relevance & Citations:")
    for i, c in enumerate(resp.get("citations", []), start=1):
        print(f" {i}. Chapter {c['chapter']} — {c['section']} (Similarity: {c['similarity']})")

    print(f"\nConfidence: {resp.get('confidence', 'unknown').upper()}")
    print("="*80 + "\n")


def parse_lambda_response(raw_payload):
    """
    Lambda returns:
    {
       "statusCode": 200,
       "body": "{\"query\":..., ...}"
    }
    So we must load body twice: once from Lambda, once for JSON body.
    """
    outer = json.loads(raw_payload)
    inner = json.loads(outer["body"])
    return inner


# ----------------- INTERACTIVE CLI LOOP -----------------
print("\n============================================================")
print(" 🔍 NIAHO Query CLI — Powered by AWS Lambda ")
print("============================================================")
print("Type a question or citation (e.g., 'Show me QM.1')")
print("Type 'exit' or 'quit' to stop.\n")

while True:
    q = input("Ask: ").strip()
    if q.lower() in ["exit", "quit"]:
        print("\n👋 Goodbye!\n")
        break

    print("\n⏳ Sending request to Lambda...\n")

    resp = client.invoke(
        FunctionName="query_handler",
        Payload=json.dumps({"query": q})
    )

    parsed = parse_lambda_response(resp["Payload"].read())

    # Route to appropriate pretty-printer
    if parsed.get("query_type") == "citation":
        pretty_print_citation(parsed)
    else:
        pretty_print_question(parsed)
