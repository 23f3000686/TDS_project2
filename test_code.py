import requests
url = " https://unskirted-vannessa-hypersonic.ngrok-free.dev/receive_request"
payload = {
  "email": "you@example.com",
  "secret": "THE_SECRET_YOU_GAVE_IN_GOOGLE_FORM",
  "url": "https://tds-llm-analysis.s-anand.net/demo"
}
r = requests.post(url, json=payload, timeout=20)
print(r.status_code)
print(r.text)
