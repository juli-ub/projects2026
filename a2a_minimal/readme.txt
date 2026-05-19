Application connects an Agent to Google AI Studio through A2A protocol 
localhost:8000

To see the agent card on http://localhost:8000/agent-card , run "curl http://localhost:8000/agent-card" from second terminal 
(optional: both in venv)


For request.json run the following from second terminal (Windows):

# Load request.json 
$body = Get-Content -Raw -Path 'request.json'

# Send request & pretty‑print the whole response
$response = Invoke-RestMethod -Method Post -Uri 'http://localhost:8000/a2a' -ContentType 'application/json' -Body $body
$response | ConvertTo-Json -Depth 10

