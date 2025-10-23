apt-get update
apt-get install -y cmake

pip install -r requirements.txt
gunicorn -k uvicorn.workers.UvicornWorker -w 2 -b 0.0.0.0:8000 main:app