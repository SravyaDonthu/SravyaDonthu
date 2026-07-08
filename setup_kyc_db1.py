import ssl
import os
ssl._create_default_https_context = ssl._create_unverified_context
os.environ["REQUESTS_CA_BUNDLE"] = ""
os.environ["CURL_CA_BUNDLE"] = ""

import httpx
import urllib3
urllib3.disable_warnings()

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ----------------------------
# CONFIG
# ----------------------------

BASE_URL = "https://genailab.tcs.in"
API_KEY = "sk-KC-vYapwC0EJlJIhaHjdVA"
EMBEDDING_MODEL = "azure/genailab-maas-text-embedding-3-large"
DB_PATH = "kyc_rules_db"

# ----------------------------
# REAL EMBEDDINGS (same as main app)
# ----------------------------

embeddings = OpenAIEmbeddings(
    model=EMBEDDING_MODEL,
    api_key=API_KEY,
    base_url=BASE_URL,
    http_client=httpx.Client(verify=False)
)

# ----------------------------
# KYC RULES KNOWLEDGE BASE
# ----------------------------

kyc_rules = """
KYC VALIDATION RULES:

1. Aadhaar number must contain exactly 12 numeric digits.
2. PAN number format must be: 5 uppercase letters, 4 digits, 1 uppercase letter.
   Example: ABCDE1234F
3. User-entered name must exactly match name on Aadhaar and PAN.
4. Address entered by user must match Aadhaar card address.
5. Uploaded live photo must match Aadhaar card photo.
6. PAN number must match the PAN card uploaded.
7. If any critical field mismatches -> KYC Status: REJECTED
8. If all validations pass -> KYC Status: APPROVED
"""

# ----------------------------
# CREATE DOCUMENT OBJECT
# ----------------------------

documents = [Document(page_content=kyc_rules)]

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

docs = splitter.split_documents(documents)

# ----------------------------
# CREATE FAISS INDEX
# ----------------------------

print("Creating FAISS index with real embeddings...")

vectorstore = FAISS.from_documents(docs, embeddings)
vectorstore.save_local(DB_PATH)

print(f"✅ FAISS KYC Rules database created successfully!")
print(f"Embedding dimension: {vectorstore.index.d}")
print(f"Saved to folder: {DB_PATH}")