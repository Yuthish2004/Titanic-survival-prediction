# Deployment Guide — Push to GitHub & Deploy Live

Follow these steps exactly, in order. Total time: ~15 minutes.

---

## Part 1 — Push to GitHub

### 1. Create a new repository on GitHub
- Go to https://github.com/new
- Repository name: `titanic-survival-prediction`
- Description: "End-to-end ML project: EDA, model comparison, and a deployed Streamlit app predicting Titanic survival"
- Keep it **Public** (so recruiters can view it without logging in)
- Do **NOT** initialize with a README (you already have one) — leave "Add a README file" unchecked
- Click **Create repository**

### 2. Push your local project
Open a terminal, navigate into the unzipped `titanic-project` folder, then run:

```bash
cd titanic-project

git init
git add .
git commit -m "Initial commit: EDA, model training, and Streamlit app"
git branch -M main
git remote add origin https://github.com/<your-username>/titanic-survival-prediction.git
git push -u origin main
```

Replace `<your-username>` with your actual GitHub username.

### 3. Verify
Refresh your GitHub repo page — you should see all folders (`app/`, `src/`, `data/`, `models/`, `assets/`) and your README rendering with the EDA image at the top.

---

## Part 2 — Deploy Live (Streamlit Community Cloud — free, no credit card)

### 1. Sign up / log in
- Go to https://share.streamlit.io
- Sign in with your GitHub account (same one you pushed to)

### 2. Create a new app
- Click **"New app"**
- Repository: select `<your-username>/titanic-survival-prediction`
- Branch: `main`
- Main file path: `app/app.py`
- Click **Deploy**

### 3. Wait for build
Streamlit will install everything from `requirements.txt` and launch your app. This takes 2-5 minutes on first deploy. You'll get a live URL like:

```
https://titanic-survival-prediction-<random>.streamlit.app
```

### 4. Update your README with the live link
Edit `README.md` locally, replace the placeholder line:

```markdown
**🔗 Live demo:** _add your deployed Streamlit link here after deployment_
```

with your actual URL, then push the update:

```bash
git add README.md
git commit -m "Add live demo link"
git push
```

### 5. Add the link everywhere it counts
- Resume: next to the project title/bullet
- LinkedIn: Projects section
- GitHub profile README (if you have one)

---

## Troubleshooting

**"ModuleNotFoundError" on deploy** → Check `requirements.txt` has every package used in `app/app.py` and that the version ranges aren't too restrictive.

**App works locally but not on Streamlit Cloud** → Check file paths are relative (they already are in `app.py` via `BASE_DIR`), not hardcoded to your local machine.

**Model file too large for GitHub** → `titanic_model.pkl` here is small (~tens of KB), so this won't apply, but if you swap in a larger model later, use [Git LFS](https://git-lfs.github.com/) for files over 50MB.

**Want a custom domain or more control?** → Alternatives to Streamlit Cloud: Render.com (free tier, more setup) or Hugging Face Spaces (also free, popular for ML demos).
