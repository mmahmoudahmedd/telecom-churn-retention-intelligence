# Deployment

The app can run using the existing reports; no retraining or model download is needed. Publish only this project folder, excluding .venv, secrets, local logs and process IDs.

## Docker hosting

Build with `docker build -t telecom-churn .` and run with `docker run --rm -p 8501:8501 telecom-churn`. The Dockerfile installs dashboard dependencies only. The PORT environment variable supports hosts with assigned ports. Health check endpoint: `/_stcore/health`.

On a Docker-capable host, deploy the repository or upload the project using the provider's supported method. Build context is the project root. Set the public service port to 8501, or let the host set PORT. The dashboard has no private data or external service credentials.

## Streamlit Community Cloud

Connect the GitHub repository, choose app.py as the entry point and Python 3.13 in advanced settings. The root requirements.txt supports the complete training environment. For a smaller dashboard-only deployment, configure a separate deployment branch using deploy/requirements.txt as its root requirements.txt. This app uses committed report files. It does not regenerate training outputs on startup.

Actual publication requires an authenticated hosting account. No public URL exists until the provider finishes deployment successfully. Verify the public health endpoint and dashboard controls after deployment.
