# Nexus Sky Engine

To run a local development server:
```
1. Place environment variables file (ask team) in project root and load
2. Place GCP admin keys (ask team) in /keys directory
3. (Optional) Create a virtual environment (preferably using venv)
3. pip install -r requirements
4. python main.py
```

# Testing

Most files have a very light testing script below. All files should be run as a python module starting from root, eg:

```bash
python -m src.sky.test
```

# Deployment

Run the deploy script (which configures the environment variables for deploy):

```
python deploy.py staging|prod
```
Then run the gcloud command it outputs

