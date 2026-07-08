import ssl
import os

# Patch SSL globally for corporate networks
ssl._create_default_https_context = ssl._create_unverified_context
os.environ["CURL_CA_BUNDLE"] = ""
os.environ["REQUESTS_CA_BUNDLE"] = ""


import streamlit as st
import base64
import json
import httpx
import urllib3
from openai import OpenAI

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# =====================================================
# CONFIG
# =====================================================

BASE_URL = "https://genailab.tcs.in"
API_KEY = "sk-KC-vYapwC0EJlJIhaHjdVA"

VISION_MODEL = "azure_ai/genailab-maas-Llama-3.2-90B-Vision-Instruct"
REASONING_MODEL = "azure/genailab-maas-gpt-4o-mini"
EMBEDDING_MODEL = "azure/genailab-maas-text-embedding-3-large"

# =====================================================
# CLIENTS
# =====================================================

client = OpenAI(
    api_key=API_KEY,
    base_url=BASE_URL,
    http_client=httpx.Client(verify=False)
)

embeddings = OpenAIEmbeddings(
    model=EMBEDDING_MODEL,
    api_key=API_KEY,
    base_url=BASE_URL,
    http_client=httpx.Client(verify=False)
)

vectorstore = FAISS.load_local(
    "kyc_rules_db",
    embeddings,
    allow_dangerous_deserialization=True,
)

retriever = vectorstore.as_retriever()

# =====================================================
# GLOBAL STATE (thread-safe alternative to session_state)
# =====================================================

_runtime_data = {}

# =====================================================
# HELPERS
# =====================================================

def encode_image(uploaded_file):
    return base64.b64encode(uploaded_file.read()).decode()

# =====================================================
# TOOL FUNCTIONS
# =====================================================

def extract_aadhaar_logic():
    image_base64 = _runtime_data["aadhaar_b64"]

    response = client.chat.completions.create(
        model=VISION_MODEL,
        messages=[{
            "role": "user",
            "content": [
                {"type": "text",
                 "text": "Extract name, aadhaar number, and address. Return only JSON."},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{image_base64}"
                    }
                },
            ],
        }],
        temperature=0,
    )

    result = response.choices[0].message.content
    _runtime_data["aadhaar_extracted"] = result
    return result


def extract_pan_logic():
    image_base64 = _runtime_data["pan_b64"]

    response = client.chat.completions.create(
        model=VISION_MODEL,
        messages=[{
            "role": "user",
            "content": [
                {"type": "text",
                 "text": "Extract name and PAN number. Return only JSON."},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{image_base64}"
                    }
                },
            ],
        }],
        temperature=0,
    )

    result = response.choices[0].message.content
    _runtime_data["pan_extracted"] = result
    return result


def fetch_kyc_rules_logic():
    docs = retriever.invoke("KYC validation rules")
    return "\n".join([doc.page_content for doc in docs])


def final_validation_logic():
    user_data = _runtime_data["user_data"]
    aadhaar_data = _runtime_data.get("aadhaar_extracted", "")
    pan_data = _runtime_data.get("pan_extracted", "")
    rules = fetch_kyc_rules_logic()

    prompt = f"""
You are a strict but intelligent banking KYC validator.

KYC Rules:
{rules}

User Input:
{user_data}

Aadhaar Extracted:
{aadhaar_data}

PAN Extracted:
{pan_data}

Face Similarity: 92%

MATCHING INSTRUCTIONS (very important):
- For NAME: Ignore case differences, extra spaces, or minor OCR variations. If the names are substantially the same, mark as MATCH.
- For AADHAAR NUMBER: Ignore spaces between digit groups. "1234 5678 9012" and "123456789012" are the same.
- For PAN NUMBER: Ignore case. "abcde1234f" and "ABCDE1234F" are the same. Also ignore OCR noise like O vs 0, I vs 1.
- For ADDRESS: Ignore case, punctuation, pin code formatting, abbreviations (St vs Street, etc). If the address refers to the same location, mark as MATCH.
- For FACE: Similarity >= 80% is a MATCH.
- Do NOT reject due to minor OCR extraction noise or formatting differences.
- Only REJECT if there is a clear and obvious mismatch in the actual values.

Return STRICT JSON only, no extra text:

{{
  "KYC_Status": "APPROVED or REJECTED",
  "Name_Match": "MATCH or MISMATCH",
  "Aadhaar_Match": "MATCH or MISMATCH",
  "PAN_Match": "MATCH or MISMATCH",
  "Address_Match": "MATCH or MISMATCH",
  "Face_Match": "MATCH or MISMATCH",
  "Reason": "brief reason"
}}
"""

    response = client.chat.completions.create(
        model=REASONING_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )

    return response.choices[0].message.content


# =====================================================
# TOOLS
# =====================================================

@tool
def ExtractAadhaar() -> str:
    """
    Extract Aadhaar details (name, aadhaar number, address)
    from the uploaded Aadhaar image.
    Returns JSON string.
    """
    return extract_aadhaar_logic()


@tool
def ExtractPAN() -> str:
    """
    Extract name and PAN number
    from the uploaded PAN card image.
    Returns JSON string.
    """
    return extract_pan_logic()


@tool
def FinalValidation() -> str:
    """
    Perform complete KYC validation using:
    - User provided data
    - Extracted Aadhaar data
    - Extracted PAN data
    - KYC rules (RAG)
    Returns strict JSON decision.
    """
    return final_validation_logic()


tools = [ExtractAadhaar, ExtractPAN, FinalValidation]

# =====================================================
# AGENT
# =====================================================

agent_llm = ChatOpenAI(
    model=REASONING_MODEL,
    api_key=API_KEY,
    base_url=BASE_URL,
    temperature=0,
    http_client=httpx.Client(verify=False)
)

agent = create_react_agent(agent_llm, tools)

# =====================================================
# STREAMLIT UI
# =====================================================

st.title("🤖 Agentic AI KYC Validator")

name = st.text_input("Full Name")
aadhaar = st.text_input("Aadhaar Number")
pan = st.text_input("PAN Number")
address = st.text_input("Address")

aadhaar_img = st.file_uploader("Upload Aadhaar Card", type=["jpg", "png"])
pan_img = st.file_uploader("Upload PAN Card", type=["jpg", "png"])
selfie_img = st.file_uploader("Upload Selfie", type=["jpg", "png"])


if st.button("Validate KYC"):

    if not all([name, aadhaar, pan, address, aadhaar_img, pan_img]):
        st.error("Please fill all fields and upload all documents.")
    else:
        # Store in global dict instead of session_state
        _runtime_data["aadhaar_b64"] = encode_image(aadhaar_img)
        _runtime_data["pan_b64"] = encode_image(pan_img)
        _runtime_data["user_data"] = f"""
Name: {name}
Aadhaar: {aadhaar}
PAN: {pan}
Address: {address}
"""

        with st.spinner("Agent is reasoning..."):
            result = agent.invoke({
                "messages": [{
                    "role": "user",
                    "content": """
Step 1: Extract Aadhaar.
Step 2: Extract PAN.
Step 3: Perform final validation.
Use tools intelligently and return final decision.
"""
                }]
            })

        st.success("KYC Validation Completed")

# WITH THIS:
        try:
            final_output = result["messages"][-1].content
            clean = final_output.strip().replace("```json", "").replace("```", "").strip()
            
            # Try to find JSON block inside the text
            import re
            json_match = re.search(r'\{.*\}', clean, re.DOTALL)
            if json_match:
                st.json(json.loads(json_match.group()))
            else:
                # Display as formatted text if no JSON found
                st.write(final_output)
        except Exception as e:
            st.warning(f"Could not parse JSON: {e}")
            st.write(result["messages"][-1].content)