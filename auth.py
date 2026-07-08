import os
from google_auth_oauthlib.flow import Flow

SCOPES = [
    'https://www.googleapis.com/auth/spreadsheets', 
    'https://www.googleapis.com/auth/drive'
]

def main():
    if not os.path.exists("client_secret.json"):
        print("❌ ERROR: Missing client_secret.json!")
        return

    # Use the correct 408-byte file
    flow = Flow.from_client_secrets_file("client_secret.json", scopes=SCOPES)
    flow.redirect_uri = "http://localhost:8080/"
    
    auth_url, _ = flow.authorization_url(prompt='consent')

    print("\n" + "="*80)
    print("🔗 COPY THIS LINK AND OPEN IT IN YOUR BROWSER:")
    print(auth_url)
    print("="*80 + "\n")
    print("1. Log in and click 'Allow'.")
    print("2. When the browser fails on 'localhost refused to connect', copy the ENTIRE URL from the address bar.")
    
    redirect_response = input("\n👉 Paste the full localhost URL here and hit Enter: ").strip()
    
    try:
        flow.fetch_token(authorization_response=redirect_response)
        with open("token.json", "w") as token:
            token.write(flow.credentials.to_json())
        print("\n✅ SUCCESS! token.json is updated. You are ready.")
    except Exception as e:
        print(f"\n❌ Failed. Error: {e}")

if __name__ == "__main__":
    main()
