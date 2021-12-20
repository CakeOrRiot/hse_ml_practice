flake8 ./scripts ./tests ./utils > logs/flake8.log

bandit -r ./scripts ./tests ./utils > logs/bandit.log