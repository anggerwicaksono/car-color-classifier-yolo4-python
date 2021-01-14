mkdir -p ~/.streamlit/
echo "[general]
email = \"anggerwicaksono@student.ittelkom-sby.ac.id\"
" > ~/.streamlit/credentials.toml
echo "[server]
headless = true
port = $PORT
enableCORS = false
" > ~/.streamlit/config.toml
