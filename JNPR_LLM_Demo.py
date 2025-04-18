#fic# =============================
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


# === Step 2: Prepare per-meeting FAISS cache ===
def prepare_meeting_cache(transcripts, model):
    chunked_data = chunk_transcripts(transcripts)
    meeting_cache = {}

    for c in chunked_data:
        key = (c['customer'], c['date'])
        if key not in meeting_cache:
            meeting_cache[key] = {"chunks": [], "texts": []}
        meeting_cache[key]["chunks"].append(c)
        meeting_cache[key]["texts"].append(c["chunk"])

    for key in meeting_cache:
        texts = meeting_cache[key]["texts"]
        embeddings = model.encode(texts)
        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(embeddings)
        meeting_cache[key]["embeddings"] = embeddings
        meeting_cache[key]["index"] = index

    return meeting_cache

# instantiate the model
model = SentenceTransformer('all-MiniLM-L6-v2')
# === Prepare cache only once ===
meeting_cache = prepare_meeting_cache(transcripts, model)

def search_faiss(user_query, customer, date, meeting_cache, model):
    key = (customer, date)
    if key not in meeting_cache:
        return [], []

    chunks = meeting_cache[key]["chunks"]
    index = meeting_cache[key]["index"]
    query_embedding = model.encode([user_query])
    D, I = index.search(query_embedding, k=min(3, len(chunks)))

    return [chunks[i] for i in I[0]], D[0]



# chunked_data = chunk_transcripts(transcripts)
# chunk_texts = [c["chunk"] for c in chunked_data]

# # === Embedding & FAISS Setup ===
# model = SentenceTransformer('all-MiniLM-L6-v2')
# chunk_embeddings = model.encode(chunk_texts)

# dimension = chunk_embeddings.shape[1]
# index = faiss.IndexFlatL2(dimension)
# index.add(np.array(chunk_embeddings))

# id_to_chunk = {i: chunked_data[i] for i in range(len(chunked_data))}

# def search_faiss(user_query, customer=None, date=None):
#     # Filter chunks by meeting if customer and date are given
#     if customer and date:
#         filtered_chunks = [c for c in chunked_data if c['customer'] == customer and c['date'] == date]
#         filtered_texts = [c['chunk'] for c in filtered_chunks]
#         if not filtered_texts:
#             return [], []
#         filtered_embeddings = model.encode(filtered_texts)
#         query_embedding = model.encode([user_query])
#         index_local = faiss.IndexFlatL2(filtered_embeddings.shape[1])
#         index_local.add(np.array(filtered_embeddings))
#         D, I = index_local.search(np.array(query_embedding), k=min(3, len(filtered_texts)))
#         return [filtered_chunks[i] for i in I[0]], D[0]
#     else:
#         query_embedding = model.encode([user_query])
#         D, I = index.search(np.array(query_embedding), k=3)
#         retrieved_texts = [id_to_chunk[i] for i in I[0]]
#         return retrieved_texts, D[0]

def query_pipeline(user_query):
    retrieved_chunks, _ = search_faiss(user_query, st.session_state.selected_customer, st.session_state.selected_date, meeting_cache, model)

    prompt = "Here are some meeting notes:\n"
    for item in retrieved_chunks:
        prompt += f"- {item['chunk']}\n"
    prompt += f"\nAnswer the question. Do not explain your thought process, the following is your question: {user_query}"

    hf_api_token = "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"
    api_url = "https://api-inference.huggingface.co/models/meta-llama/Llama-3.3-70B-Instruct"
    headers = {"Authorization": f"Bearer {hf_api_token}"}
    data = {"inputs": prompt, "parameters": {"max_new_tokens": 150}}

    response = requests.post(api_url, headers=headers, json=data)
    if response.status_code != 200:
        return "❌ Error from Hugging Face API."

    result = response.json()
    output = result[0]['generated_text']
    output = output.split(user_query)[-1].strip()
    output = re.sub(r'[#\\$\\\\]', '', output)
    output = re.sub(r'\\boxed\{.*?\}', '', output)

    return output.strip()

# === Streamlit App ===
if "summary_generated" not in st.session_state:
    st.session_state.summary_generated = False
if "generated_summary" not in st.session_state:
    st.session_state.generated_summary = ""
if "qa_history" not in st.session_state:
    st.session_state.qa_history = []


st.image("juniperlogo.png", width=100)
st.markdown("<h3 style='margin-top: -10px;'>Juniper Meeting Insights Q&A (Working Demo)</h3>", unsafe_allow_html=True)

# Step 1: Filters
customer_list = sorted(list(set([t["customer"] for t in transcripts])))
st.session_state.selected_customer = st.selectbox("Select Customer", ["Select"] + customer_list)

meeting_dates = [t["date"] for t in transcripts if t["customer"] == st.session_state.selected_customer]
st.session_state.selected_date = st.selectbox("Select Meeting Date", ["Select"] + meeting_dates if st.session_state.selected_customer != "Select" else ["Select"])

# Generate MOM Button
if st.button("Generate MOM..."):
    if st.session_state.selected_customer != "Select" and st.session_state.selected_date != "Select":
        meeting = next((t for t in transcripts if t["customer"] == st.session_state.selected_customer and t["date"] == st.session_state.selected_date), None)
        if meeting:
            prompt = f"""You are an AI assistant helping summarize enterprise customer meetings.

Meeting: {meeting['customer']}  
Date: {meeting['date']}

Transcript:
{meeting['transcript']}

Please generate a 200 words complete professional summary (Minutes of the Meeting), capturing key points discussed, pain points, proposed solutions, and follow-ups. Keep it complete, concise and business-friendly.
"""

            def clean_summary_text(raw_summary: str) -> str:
                content = re.sub(r'^#+\s*', '', raw_summary, flags=re.MULTILINE)
                content = re.sub(r'\*\*|__|\*|_', '', content)
                content = re.sub(r'\n{2,}', '\n\n', content)
                return content

            hf_api_token = "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"
            api_url = "https://api-inference.huggingface.co/models/meta-llama/Llama-3.3-70B-Instruct"
            headers = {"Authorization": f"Bearer {hf_api_token}"}
            data = {"inputs": prompt, "parameters": {"max_new_tokens": 300}}

            response = requests.post(api_url, headers=headers, json=data)
            if response.status_code == 200:
                result = response.json()
                summary = result[0]['generated_text']
                summary = summary.split("business-friendly.")[-1].strip()
                summary = clean_summary_text(summary)

                st.session_state.summary_generated = True
                st.session_state.generated_summary = summary
                st.session_state.qa_history = []
            else:
                st.error("⚠️ LLM failed to generate summary. Check API.")

# === Show Summary & Q&A ===
if st.session_state.summary_generated:
    #st.subheader("📄 Meeting Summary (Minutes of the Meeting)")
    st.write(st.session_state.generated_summary)

    st.subheader("❓ Ask Specific Question About This Meeting")
    user_question = st.text_input("Write a Question and hit enter:", key="user_q")

    if user_question and (
        len(st.session_state.qa_history) == 0 or
        user_question != st.session_state.qa_history[-1]['question']
    ):
        answer = query_pipeline(user_question)
        st.session_state.qa_history.append({"question": user_question, "answer": answer})
        st.rerun()

    if st.session_state.qa_history:
        st.markdown("### 🧠 Q&A History")
        for pair in st.session_state.qa_history:
            st.markdown(f"**Q:** {pair['question']}")
            st.markdown(f"**A:** {pair['answer']}")
            st.markdown("---")

    if st.button("🔙 Back to Main"):
        st.session_state.summary_generated = False
        st.session_state.generated_summary = ""
        st.session_state.qa_history = []
        st.rerun()
