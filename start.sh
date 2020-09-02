export FLASK_APP=$PWD/app/app.py
#export FLASK_ENV=development
export FLASK_ENV=production
#python app/setup.py
#python app/app.py
flask run --host=0.0.0.0 --port 5001
