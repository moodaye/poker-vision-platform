# Setting Up Your GitHub Repository

Follow these steps to create a new GitHub repository with your screen monitor project:

## Method 1: GitHub Web Interface (Recommended)

### Step 1: Create Repository on GitHub
1. Go to [GitHub.com](https://github.com) and sign in
2. Click the "+" icon in the top right corner
3. Select "New repository"
4. Fill in the details:
   - **Repository name**: `screen-monitor` (or your preferred name)
   - **Description**: `Real-time screen capture system with webhook integration`
   - **Visibility**: Choose Public or Private
   - **Initialize**: Leave unchecked (we'll push existing code)
5. Click "Create repository"

### Step 2: Download Project Files
1. In Replit, go to the file explorer
2. Download all these files to your computer:
   ```
   app.py
   screen_capture.py
   config.py
   main.py
   README.md
   requirements-export.txt (rename to requirements.txt)
   LICENSE
   .gitignore
   templates/
   static/
   webhook_receiver_demo.py
   outbound_integration_demo.py
   test_with_upload.py
   local_setup_guide.md
   ```

### Step 3: Initialize Local Repository
Open terminal/command prompt in your project folder and run:

```bash
# Initialize git repository
git init

# Add all files
git add .

# Make initial commit
git commit -m "Initial commit: Screen monitor with webhook integration"

# Add your GitHub repository as remote
git remote add origin https://github.com/YOUR_USERNAME/screen-monitor.git

# Push to GitHub
git branch -M main
git push -u origin main
```

## Method 2: GitHub CLI (Alternative)

If you have GitHub CLI installed:

```bash
# Create repository directly from terminal
gh repo create screen-monitor --public --source=. --remote=origin --push

# Or for private repository
gh repo create screen-monitor --private --source=. --remote=origin --push
```

## Method 3: Using Git Commands Only

```bash
# Clone this Replit project first
git clone <replit-git-url>
cd screen-monitor

# Create new repository on GitHub (via web interface)
# Then add the remote:
git remote set-url origin https://github.com/YOUR_USERNAME/screen-monitor.git
git push -u origin main
```

## After Repository Creation

### 1. Update Repository Settings
- Add topics: `python`, `flask`, `screen-capture`, `webhooks`, `monitoring`
- Set up branch protection if needed
- Configure GitHub Pages (optional) for documentation

### 2. Update README.md
- Replace `<your-repo-url>` with your actual repository URL
- Add your GitHub username/organization name
- Update any specific setup instructions

### 3. Set Up Issues and Discussions (Optional)
- Enable Issues for bug reports and feature requests
- Enable Discussions for community support
- Create issue templates for better bug reporting

### 4. Add GitHub Actions (Optional)
Create `.github/workflows/test.yml` for automated testing:

```yaml
name: Test Application
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: 3.9
    - name: Install dependencies
      run: pip install -r requirements.txt
    - name: Run tests
      run: python -m pytest tests/ # if you add tests
```

## Repository Structure

Your final repository should look like this:

```
screen-monitor/
├── README.md
├── LICENSE
├── .gitignore
├── requirements.txt
├── main.py
├── app.py
├── screen_capture.py
├── config.py
├── templates/
│   ├── base.html
│   └── index.html
├── static/
│   ├── css/
│   │   └── custom.css
│   └── js/
│       └── monitor.js
├── webhook_receiver_demo.py
├── outbound_integration_demo.py
├── test_with_upload.py
└── local_setup_guide.md
```

## Next Steps

1. **Clone your repository** on machines where you want to run the screen monitor
2. **Install dependencies**: `pip install -r requirements.txt`
3. **Run the application**: `python main.py`
4. **Configure webhooks** and start capturing!

## Troubleshooting

**Permission denied errors:**
```bash
git remote set-url origin https://YOUR_USERNAME:YOUR_TOKEN@github.com/YOUR_USERNAME/screen-monitor.git
```

**Large file warnings:**
The `.gitignore` file excludes image files to prevent repository bloat.

**Authentication issues:**
Use personal access tokens instead of passwords for HTTPS authentication.