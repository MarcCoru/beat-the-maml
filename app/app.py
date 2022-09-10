from flask import Flask, render_template, request
import os
from datetime import datetime
import json

app = Flask(__name__)
app.config['CORS_HEADERS'] = 'Content-Type'

JSON_RECORDS_FILE = "app/static/records.json"

@app.route('/')
def index():
    userid = request.args.get('userid')
    if userid is None:
        userid = ""
    return render_template("index.html", userid=userid)

def append_record(record):
    if os.path.exists(JSON_RECORDS_FILE):
        append_write = 'a' # append if already exists
    else:
        append_write = 'w' # make a new file if not

    with open(JSON_RECORDS_FILE, append_write) as f:
        json.dump(record, f)
        f.write(os.linesep)

def print_record(record):
    print("***")
    print(f"new entry logged in {JSON_RECORDS_FILE}")
    for k, v in record.items():
        print(k, v)
    print("***")

@app.route('/dropevent', methods = ['POST'])
def post_javascript_data():
    jsdata = request.form['data']
    data = json.loads(jsdata)
    data["timestamp"] = datetime.now().isoformat()
    print_record(data)
    append_record(data)

    return jsdata


if __name__ == "__main__":
   from waitress import serve
   serve(app, host='0.0.0.0', port=80)
