import ssl
import certifi
import nltk

# ✅ Fix: Use a lambda to correctly assign a callable context generator
ssl._create_default_https_context = lambda: ssl.create_default_context(cafile=certifi.where())

# Download corpora
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('brown')

# run python fix_nltk_ssl.py