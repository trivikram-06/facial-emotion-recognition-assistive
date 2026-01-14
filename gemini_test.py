from google import genai

# Configure client with API key
client = genai.Client(api_key="AIzaSyBvyPpLNs0FFb402AtGEGeQ6lzzgM35LPg")

response = client.models.generate_content(
    model="gemini-1.5-flash",
    contents="Generate a short, empathetic message for someone who looks happy."
)

print(response.text)
