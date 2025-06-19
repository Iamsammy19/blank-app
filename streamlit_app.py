import streamlit as st
import hashlib
from datetime import datetime

# Configure the app for mobile
st.set_page_config(
    page_title="Naija Cyber Shield",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Nigerian threat database
THREATS = {
    "ransomware_exts": [".encrypted", ".locked", ".crypt", ".money", ".wannacry"],
    "malware_hashes": {
        "d41d8cd98f00b204e9800998ecf8427e": "Test/Dummy Virus",
        # Add real Nigerian malware hashes here
    },
    "scam_keywords": {
        "pidgin": ["win airtime", "click link", "your account", "bvn verification", "atm card"],
        "english": ["URGENT:", "account suspended", "verify now", "congratulations winner"]
    }
}

# Mobile-friendly UI
st.title("🇳🇬 Naija Cyber Shield")
st.markdown("Protect against ransomware, malware & Nigerian scams")

tab1, tab2 = st.tabs(["📁 File Scanner", "📩 Message Checker"])

with tab1:
    uploaded_file = st.file_uploader("Upload file to scan:")
    if uploaded_file:
        # Check ransomware extensions
        if any(uploaded_file.name.lower().endswith(ext) for ext in THREATS["ransomware_exts"]):
            st.error("🚨 RANSOMWARE DETECTED!")
            st.write("Common Nigerian ransomware extensions detected (.encrypted, .locked etc.)")
        
        # Check malware hashes
        file_hash = hashlib.md5(uploaded_file.getvalue()).hexdigest()
        if file_hash in THREATS["malware_hashes"]:
            st.error(f"🦠 MALWARE FOUND: {THREATS['malware_hashes'][file_hash]}")
        else:
            st.success("✅ No known threats detected")
        
        st.code(f"File: {uploaded_file.name}\nHash: {file_hash}")

with tab2:
    user_input = st.text_area("Paste SMS/Email:")
    language = st.radio("Language", ["pidgin", "english"], horizontal=True)
    
    if user_input:
        if any(keyword in user_input.lower() for keyword in THREATS["scam_keywords"][language]):
            st.error("🛑 SCAM ALERT!")
            st.write("Common Nigerian fraud pattern detected:")
            st.write([k for k in THREATS["scam_keywords"][language] if k in user_input.lower()])
        else:
            st.success("📩 Message appears safe")

# Footer with Nigerian contacts
st.markdown("---")
st.markdown("**Report cybercrime to:**")
st.markdown("- EFCC: `08006322222`")
st.markdown("- NCC: `622` (USSD)")
