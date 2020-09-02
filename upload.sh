rsync -avz -e "ssh" \
  --exclude 'app/static/records.json' \
  --exclude 'app/static/img' \
  . ubuntu@138.246.234.218:sen12ms-human-few-shot-classifier
