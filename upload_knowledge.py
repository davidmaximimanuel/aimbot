import os
import json
import urllib.request
import urllib.error

# Load your Supabase credentials from environment variables
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY") # Use your service_role key

def upload_knowledge():
    if not SUPABASE_URL or not SUPABASE_KEY:
        print("❌ Error: SUPABASE_URL and SUPABASE_KEY environment variables are not set.")
        return

    # 1. Load the JSON file
    json_path = os.path.join(os.path.dirname(__file__), 'nigeria_knowledge.json')
    
    if not os.path.exists(json_path):
        print(f"❌ Error: {json_path} not found. Please save the JSON file first.")
        return

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    rows_to_insert = []

    # 2. Flatten the JSON into database rows
    for item in data.get("slang_and_pidgin", []):
        rows_to_insert.append({
            "category": "slang_and_pidgin",
            "key": item.get("phrase", ""),
            "value": item.get("meaning", ""),
            "context": item.get("context", "")
        })

    for item in data.get("cultural_norms", []):
        rows_to_insert.append({
            "category": "cultural_norms",
            "key": item.get("topic", ""),
            "value": item.get("rule", ""),
            "context": ""
        })

    for item in data.get("geography_and_transport", []):
        rows_to_insert.append({
            "category": "geography_and_transport",
            "key": item.get("location", item.get("transport", "")),
            "value": item.get("facts", item.get("apps", "")),
            "context": ""
        })

    for item in data.get("current_events_sources", []):
        rows_to_insert.append({
            "category": "current_events_sources",
            "key": item.get("name", ""),
            "value": item.get("url", ""),
            "context": item.get("category", "")
        })

    if not rows_to_insert:
        print("No data found to upload.")
        return

    # 3. Upload to Supabase using built-in urllib (No pip install needed!)
    print(f"🚀 Uploading {len(rows_to_insert)} rows to Supabase...")
    
    url = f"{SUPABASE_URL}/rest/v1/nigeria_knowledge"
    headers = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json",
        "Prefer": "return=minimal"
    }
    
    data_bytes = json.dumps(rows_to_insert).encode('utf-8')
    req = urllib.request.Request(url, data=data_bytes, headers=headers, method='POST')
    
    try:
        with urllib.request.urlopen(req) as response:
            print(f"✅ Success! Status Code: {response.status}")
            print(f"🇳🇬 Rows inserted into nigeria_knowledge table.")
    except urllib.error.HTTPError as e:
        print(f"❌ Failed with status code: {e.code}")
        print(e.read().decode('utf-8'))

if __name__ == "__main__":
    upload_knowledge()