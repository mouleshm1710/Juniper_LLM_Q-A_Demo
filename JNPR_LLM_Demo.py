# =============================
# Install Libraries First (CMD)
# pip install faiss-cpu sentence-transformers streamlit requests numpy
# =============================

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import streamlit as st
import requests
import re

# =============================
# Load Sample Transcripts
# =============================

transcripts = [
    {
        "customer": "XYZ Telecom Ltd.",
        "date": "2025-01-12",
        "transcript": 
    """Customer: We're facing persistent latency issues in our Asia-Pacific backbone network, especially during peak hours.
    Juniper Rep: Have you considered implementing dynamic traffic rerouting?
    Customer: Not yet, but we’re looking for a solution with minimal integration effort.
    Juniper Rep: Our MX series routers with Segment Routing can optimize the traffic dynamically.
    Customer: What about OPEX? Manual configs are driving up operational costs.
    Juniper Rep: Our Paragon Automation suite helps reduce manual interventions by nearly 40%.
    Customer: Interesting. Is it compatible with our current OSS/BSS stack?
    Juniper Rep: Yes, we provide open APIs and REST interfaces for seamless integration.
    Customer: Do you offer real-time network visibility dashboards?
    Juniper Rep: Absolutely, Paragon Insights provides live performance monitoring and SLA tracking.
    Customer: How does your solution handle traffic surges during sporting events?
    Juniper Rep: It uses AI-driven predictive models to pre-allocate capacity during such events.
    Customer: What about redundancy and failover capabilities?
    Juniper Rep: Our solution offers sub-second failovers using redundant routing paths.
    Customer: Security compliance is important too.
    Juniper Rep: Our SRX firewalls and AI-driven threat prevention integrate easily into the same ecosystem.
    Customer: What’s the typical deployment timeline?
    Juniper Rep: Depending on complexity, 6-8 weeks including full integration and training.
    Customer: Can we arrange a PoC?
    Juniper Rep: Absolutely, we’ll prepare a tailored PoC proposal."""
    },
    {
        "customer": "Alpha Cloud Services",
        "date": "2025-02-03",
        "transcript": 
    """Customer: Security compliance is key for us. We're SOC2 and ISO27001 certified.
    Juniper Rep: Our SRX firewalls and ATP provide deep packet inspection and real-time threat detection.
    Customer: We’re exploring zero-trust strategies as well.
    Juniper Rep: Mist AI integrates behavioral monitoring and access policies supporting zero-trust models.
    Customer: How about managing East-West traffic in our data centers?
    Juniper Rep: Contrail Networking handles microsegmentation and East-West traffic control.
    Customer: We’re concerned about SD-WAN costs. Can your solution offer flexibility?
    Juniper Rep: Yes, our AI-driven SD-WAN supports bandwidth-on-demand, optimizing costs dynamically.
    Customer: Any support for multi-cloud environments?
    Juniper Rep: Our Cloud Metro platform ensures seamless multi-cloud connectivity and centralized policies.
    Customer: Downtime is a risk. How do you handle that?
    Juniper Rep: Our EVPN-VXLAN architecture supports near-zero downtime and quick failovers.
    Customer: Do you integrate with cloud providers like AWS and Azure?
    Juniper Rep: Absolutely, with certified integrations for AWS, Azure, and GCP.
    Customer: Can we analyze user experience metrics?
    Juniper Rep: Mist AI provides detailed user experience analytics, including application performance.
    Customer: How scalable is your solution?
    Juniper Rep: It's highly scalable, tested for large enterprise-grade deployments.
    Customer: What's the support model?
    Juniper Rep: 24/7 support with dedicated technical account managers.
    Customer: Can we arrange an architectural workshop?
    Juniper Rep: Certainly, we can schedule a session next week."""
    },
    {
        "customer": "Beta Financial Group",
        "date": "2025-03-10",
        "transcript": 
    """Customer: Regulatory compliance requires audit trails for all data movement.
    Juniper Rep: Our Contrail Networking supports granular flow logging, simplifying compliance.
    Customer: We're planning a multi-cloud strategy.
    Juniper Rep: Our Cloud Metro solution ensures consistent policy control across clouds.
    Customer: Failovers should be instant. Can you guarantee sub-second?
    Juniper Rep: Yes, leveraging our EVPN-VXLAN architecture provides low-latency failovers.
    Customer: We’re seeing rising internal security breaches.
    Juniper Rep: Mist AI supports user/device behavior monitoring and anomaly detection.
    Customer: Can we enforce policy-based controls across branches?
    Juniper Rep: Yes, via centralized SD-WAN policies.
    Customer: Any AI-driven insights to help resource allocation?
    Juniper Rep: Our AI engine recommends optimal bandwidth routing and resource adjustments.
    Customer: Downtime reports are needed monthly.
    Juniper Rep: Paragon Insights can auto-generate SLA and downtime reports.
    Customer: Integration with existing SIEM tools?
    Juniper Rep: Supported! We provide APIs for SIEM integration.
    Customer: How about remote employee performance?
    Juniper Rep: Our solution provides real-time monitoring of remote network access quality.
    Customer: CRM integration possible?
    Juniper Rep: Yes, APIs support CRM and customer behavior analysis.
    Customer: Can we schedule quarterly audits?
    Juniper Rep: Absolutely, we can set them up in our engagement plan.
    Customer: Any training support?
    Juniper Rep: Yes, customized training programs are available for your team."""
    }
]

# === Chunk Function ===
def chunk_transcripts(transcripts, chunk_size=5, overlap=1):
    all_chunks = []
    for entry in transcripts:
        sentences = entry["transcript"].split('\n')
        chunks = []
        start = 0
        while start < len(sentences):
            end = start + chunk_size
            chunk = "\n".join(sentences[start:end]).strip()
            if chunk:
                chunks.append(chunk)
            start += chunk_size - overlap
        for chunk in chunks:
            all_chunks.append({
                "customer": entry["customer"],
                "date": entry["date"],
                "chunk": chunk
            })
    return all_chunks

chunked_data = chunk_transcripts(transcripts)
chunk_texts = [c["chunk"] for c in chunked_data]

# === Embedding & FAISS Setup ===
model = SentenceTransformer('all-MiniLM-L6-v2')
chunk_embeddings = model.encode(chunk_texts)

dimension = chunk_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(chunk_embeddings))

id_to_chunk = {i: chunked_data[i] for i in range(len(chunked_data))}

def search_faiss(user_query):
    query_embedding = model.encode([user_query])
    D, I = index.search(np.array(query_embedding), k=3)
    retrieved_texts = [id_to_chunk[i] for i in I[0]]
    return retrieved_texts, D[0]

def query_pipeline(user_query):
    lowered_query = user_query.lower()
    if any(word in lowered_query for word in ["summary", "summarize", "highlight", "highlights", "key points", "key"]) and \
       any(cust in lowered_query for cust in ["xyz", "alpha", "beta"]):

        # Summary mode
        target_cust = None
        for key in ["xyz", "alpha", "beta"]:
            if key in lowered_query:
                target_cust = key
                break

        full_text = None
        for entry in transcripts:
            if target_cust.lower() in entry["customer"].lower():
                full_text = entry["transcript"]
                break

        if not full_text:
            return "Customer not found."

        prompt = f"Here is a meeting transcript:\n{full_text}\n\nPlease summarize the key highlights from the meeting. Avoid repeating the prompt. Answer directly."

    else:
        # Regular Q&A
        query_embedding = model.encode([user_query])
        retrieved_chunks, _ = search_faiss(user_query)

        prompt = "Here are some meeting notes:\n"
        for item in retrieved_chunks:
            prompt += f"- {item['chunk']}\n"
        prompt += f"\nAnswer the following question: {user_query}"

    # Call LLM API
    hf_api_token = "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"
    api_url = "https://api-inference.huggingface.co/models/meta-llama/Llama-3.3-70B-Instruct"
    headers = {"Authorization": f"Bearer {hf_api_token}"}
    data = {"inputs": prompt, "parameters": {"max_new_tokens": 200}}

    response = requests.post(api_url, headers=headers, json=data)
    if response.status_code != 200:
        return "❌ Error from Hugging Face API."

    result = response.json()
    output = result[0]['generated_text']
    output = output.split(user_query)[-1].strip()
    output = re.sub(r'[#\$\\]', '', output)
    output = re.sub(r'\\boxed\{.*?\}', '', output)

    return output.strip()


# =============================
# Streamlit Frontend
# =============================

st.image("juniper_logo.png", width=100)
st.title("Juniper Meeting Insights Q&A (Working Demo)")

# Step 1: Filters
customer_list = sorted(list(set([t["customer"] for t in transcripts])))
selected_customer = st.selectbox("Select Customer", ["Select"] + customer_list)

meeting_dates = [t["date"] for t in transcripts if t["customer"] == selected_customer]
selected_date = st.selectbox("Select Meeting Date", ["Select"] + meeting_dates if selected_customer != "--" else ["--"])

# Step 2: MOM Summary
if selected_customer != "--" and selected_date != "--":
    #st.subheader("📄 Meeting Summary (Minutes of the Meeting)")

    meeting = next((t for t in transcripts if t["customer"] == selected_customer and t["date"] == selected_date), None)
    if meeting:
        prompt = f"""You are an AI assistant helping summarize enterprise customer meetings.

Meeting: {meeting['customer']}  
Date: {meeting['date']}

Transcript:
{meeting['transcript']}

Please generate a professional summary (Minutes of the Meeting), capturing key points discussed, pain points, proposed solutions, and follow-ups. Keep it concise and business-friendly.
"""

        # LLM Call
        hf_api_token = "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"
        api_url = "https://api-inference.huggingface.co/models/meta-llama/Llama-3.3-70B-Instruct"
        headers = {"Authorization": f"Bearer {hf_api_token}"}
        data = {"inputs": prompt, "parameters": {"max_new_tokens": 300}}

        response = requests.post(api_url, headers=headers, json=data)

        if response.status_code == 200:
            result = response.json()
            summary = result[0]['generated_text']
            #st.markdown("✅ **Summary:**")
            summary = summary.split("business-friendly.")[-1].strip()
            st.markdown(summary, unsafe_allow_html=True)
            #st.write(summary)

            # Step 3: Q&A Mode
            st.subheader("❓ Ask Questions About This Meeting")
            user_question = st.text_input("Ask a question:")

            if user_question:
                output = query_pipeline(user_question)
                st.text(output)
                if st.button("🔙 Back to Home"):
                    st.experimental_rerun()
        else:
            st.error("⚠️ LLM failed to generate summary. Check API.")
