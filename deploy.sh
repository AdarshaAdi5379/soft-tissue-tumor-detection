#!/usr/bin/env bash
set -e

echo "1) Install deps locally (optional)"
pip install -r requirements.txt

echo "2) Create Git commit and push to remote"
git add .
git commit -m "Prepare for Railway deployment" || true
git push origin main

echo "3) On Railway: create a MySQL plugin/service and copy the credentials to the Railway Dashboard env vars"
echo "   (Do this via UI: New -> Database -> MySQL)"

echo "4) In Railway dashboard, set the environment variables listed in .env.template"
echo "   MYSQL_HOST, MYSQL_PORT, MYSQL_DATABASE, MYSQL_USER, MYSQL_PASSWORD, MODEL_DIR, SECRET_KEY"

echo "5) Deploy the project via Railway UI or railway up (if CLI installed)"
echo "   railway up"
