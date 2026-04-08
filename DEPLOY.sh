# BugFixBench — Deployment Playbook
# Run these commands TOP TO BOTTOM. Takes ~15 minutes total.
# Replace YOUR_HF_USERNAME and YOUR_GITHUB_USERNAME throughout.

# ══════════════════════════════════════════════════════════════
# STEP 1 — Local test (30 seconds)
# ══════════════════════════════════════════════════════════════

pip install -r requirements.txt

# Terminal 1: start server
uvicorn main:app --host 0.0.0.0 --port 7860

# Terminal 2: run validator
python validate.py

# If all green → proceed. Fix any red issues first.


# ══════════════════════════════════════════════════════════════
# STEP 2 — Push to GitHub
# ══════════════════════════════════════════════════════════════

git init
git add .
git commit -m "feat: BugFixBench v1.0 — OpenEnv Round 1 submission"

# Create a NEW repo on github.com named "bugfixbench" first, then:
git remote add origin https://github.com/YOUR_GITHUB_USERNAME/bugfixbench.git
git branch -M main
git push -u origin main


# ══════════════════════════════════════════════════════════════
# STEP 3 — Create HF Space
# ══════════════════════════════════════════════════════════════

pip install huggingface_hub

python -c "
from huggingface_hub import HfApi
api = HfApi()
api.create_repo(
    repo_id='YOUR_HF_USERNAME/bugfixbench',
    repo_type='space',
    space_sdk='docker',
    private=False,
)
print('Space created!')
"


# ══════════════════════════════════════════════════════════════
# STEP 4 — Push to HF Spaces
# ══════════════════════════════════════════════════════════════

# Add HF remote
git remote add hf https://huggingface.co/spaces/YOUR_HF_USERNAME/bugfixbench

# Push (login if prompted: huggingface-cli login)
git push hf main


# ══════════════════════════════════════════════════════════════
# STEP 5 — Set environment variables in HF Space
# ══════════════════════════════════════════════════════════════

# Go to: https://huggingface.co/spaces/YOUR_HF_USERNAME/bugfixbench/settings
# Under "Repository secrets" add:
#   API_BASE_URL  = https://api.openai.com/v1   (or your provider)
#   MODEL_NAME    = gpt-4o-mini
#   HF_TOKEN      = hf_xxxxxxxxxxxxxxxxxxxx

# OR via CLI:
python -c "
from huggingface_hub import HfApi
api = HfApi()
# Add each secret:
api.add_space_secret('YOUR_HF_USERNAME/bugfixbench', 'API_BASE_URL', 'https://api.openai.com/v1')
api.add_space_secret('YOUR_HF_USERNAME/bugfixbench', 'MODEL_NAME', 'gpt-4o-mini')
api.add_space_secret('YOUR_HF_USERNAME/bugfixbench', 'HF_TOKEN', 'YOUR_ACTUAL_KEY')
print('Secrets set!')
"


# ══════════════════════════════════════════════════════════════
# STEP 6 — Verify HF Space is live
# ══════════════════════════════════════════════════════════════

# Wait 2-3 min for Docker build. Then:
python -c "
import httpx
url = 'https://YOUR_HF_USERNAME-bugfixbench.hf.space'
r = httpx.get(url + '/', timeout=30)
print('Status:', r.status_code)
print('Body:', r.json())
"

# Should print: Status: 200 and body with status=ok


# ══════════════════════════════════════════════════════════════
# STEP 7 — Run inference against live HF Space
# ══════════════════════════════════════════════════════════════

export API_BASE_URL=https://api.openai.com/v1
export MODEL_NAME=gpt-4o-mini
export HF_TOKEN=your_api_key_here
export ENV_URL=https://YOUR_HF_USERNAME-bugfixbench.hf.space

python inference.py


# ══════════════════════════════════════════════════════════════
# STEP 8 — Submit
# ══════════════════════════════════════════════════════════════

# Submit this URL on the OpenEnv platform before 11:59 PM IST:
# https://YOUR_HF_USERNAME-bugfixbench.hf.space


# ══════════════════════════════════════════════════════════════
# TROUBLESHOOTING
# ══════════════════════════════════════════════════════════════

# HF Space build failed?
#   → Check build logs in HF Space "Logs" tab
#   → Common fix: make sure Dockerfile EXPOSE 7860 is present

# inference.py hangs?
#   → Your API key or base URL is wrong
#   → Test: python -c "from openai import OpenAI; c=OpenAI(base_url='...', api_key='...'); print(c.models.list())"

# validate.py shows red?
#   → Fix that specific endpoint in main.py / tasks.py
#   → Re-run: python validate.py
