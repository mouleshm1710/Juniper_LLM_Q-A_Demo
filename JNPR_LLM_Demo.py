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
from streamlit.components.v1 import html

# =============================
# 1. Load Sample Transcripts
# =============================

transcripts = [

    """Meeting with: XYZ Telecom Ltd.
    Date: Jan 12, 2025
    
    Customer: We're facing persistent latency issues in our Asia-Pacific backbone network, especially during peak hours.
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
    Juniper Rep: Absolutely, we’ll prepare a tailored PoC proposal.""",

    """Meeting with: Alpha Cloud Services
    Date: Feb 3, 2025
    
    Customer: Security compliance is key for us. We're SOC2 and ISO27001 certified.
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
    Juniper Rep: Certainly, we can schedule a session next week.""",

    """Meeting with: Beta Financial Group
    Date: Mar 10, 2025
    
    Customer: Regulatory compliance requires audit trails for all data movement.
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
]

def chunk_transcripts(transcripts, chunk_size=3, overlap=1):
    all_chunks = []
    for transcript in transcripts:
        # Split into sentences (basic split, refine if needed)
        sentences = transcript.split('\n')
        chunks = []
        start = 0
        while start < len(sentences):
            end = start + chunk_size
            chunk = "\n".join(sentences[start:end]).strip()
            if chunk:
                chunks.append(chunk)
            start += chunk_size - overlap  # Move window with overlap
        all_chunks.extend(chunks)
    return all_chunks

# Chunk original transcripts
chunked_transcripts = chunk_transcripts(transcripts, chunk_size=5, overlap=1)

# Check total chunks
print(f"Total sub-transcripts created: {len(chunked_transcripts)}")


# Generate embeddings for chunks
model = SentenceTransformer('all-MiniLM-L6-v2')
chunk_embeddings = model.encode(chunked_transcripts)

# FAISS setup
dimension = chunk_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(chunk_embeddings))

# Updated mapping
id_to_chunk = {i: chunked_transcripts[i] for i in range(len(chunked_transcripts))}


# === Search Function ===
def search_faiss(user_query):
    query_embedding = model.encode([user_query])
    D, I = index.search(np.array(query_embedding), k=3)  # top 3
    retrieved_texts = [id_to_chunk[i] for i in I[0]]
    return retrieved_texts, D[0]

# Map customer names to full transcripts (simple lowercased keys)
customer_transcript_map = {
    "xyz": transcripts[0],
    "alpha": transcripts[1],
    "beta": transcripts[2]
}


def query_pipeline(user_query):
    lowered_query = user_query.lower()
    
    # Check if it's a summary/highlights question
    if any(word in lowered_query for word in ["summary", "summarize", "highlight", "highlights", "key points",'key']) and \
       any(cust in lowered_query for cust in ["xyz", "alpha", "beta"]):

        st.info("📄 Detected summary request - passing full transcript!")

        # Extract customer
        target_cust = None
        for cust in customer_transcript_map.keys():
            if cust in lowered_query:
                target_cust = cust
                break
        
        if target_cust:
            full_text = customer_transcript_map[target_cust]

            # Prepare prompt
            prompt = f"Here is the full meeting transcript with {target_cust.title()}:\n"
            prompt += f"{full_text}\n\n"
            user_query = f"Provide a summary or key highlights. Do not explain your reasoning. Answer directly."
            prompt += user_query

            # Display in frontend
            with st.expander("📄 Full Transcript Passed to LLM:"):
                st.write(full_text)
        else:
            return "Customer name not recognized. Please mention XYZ, Alpha, or Beta."

    else:
        # Regular FAISS flow
        st.info("🔍 Generating query embedding...")
        query_embedding = model.encode([user_query])

        st.info("📂 Searching for similar conversations in Database...")
        retrieved_texts, distances = search_faiss(user_query)
        st.success("Top similar conversations retrieved!")

        with st.expander("📄 See Retrieved Conversations"):
            for i, txt in enumerate(retrieved_texts):
                st.markdown(f"**Conversation {i+1}:**")
                st.write(txt)  # Display as clean text block

        # Prepare prompt
        st.info("📝 Preparing prompt for LLM...")
        prompt = "Here are some meeting notes:\n"
        for txt in retrieved_texts:
            prompt += f"- {txt}\n"
        prompt += f"\nAnswer the question. Do not explain your thought process, the following is your question: {user_query}"

        with st.expander("📄 Final Prompt Sent to LLM"):
            st.code(prompt)

    # === LLM API Call (Common) ===
    st.info("🚀 Sending prompt to LLM...")

    hf_api_token = "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"
    api_url = "https://api-inference.huggingface.co/models/meta-llama/Llama-3.3-70B-Instruct"
    headers = {"Authorization": f"Bearer {hf_api_token}"}
    data = {"inputs": prompt, "parameters": {"max_new_tokens": 150}}

    response = requests.post(api_url, headers=headers, json=data)
    
    if response.status_code != 200:
        return "❌ Error fetching response from Hugging Face API. Check API key."
    
    result = response.json()
    output = result[0]['generated_text']
    
    # Clean final output
    output = output.split(user_query)[-1].strip()
    output = re.sub(r'[#\$\\]', '', output)  # remove #, $, \ symbols
    output = re.sub(r'\\boxed\{.*?\}', '', output)  # remove \boxed{}
    output = output.strip()
    
    # Optional: remove extra duplicated lines
    # lines = output.splitlines()
    # cleaned_lines = []
    # for line in lines:
    #     if line.strip() not in cleaned_lines:
    #         cleaned_lines.append(line.strip())
    
    #clean_output = "\n".join(cleaned_lines)
    clean_output = output
    return clean_output


# === Streamlit Frontend ===
st.image("juniper_logo.png", width=100)
st.title("Juniper Meeting Insights Q&A (Working Demo)")

# ℹ️ Hoverable Technical Info Tooltip (HTML hack)
html("""
<div style='display: flex; align-items: center;'>
    <h4 style='margin-right: 8px;'>ℹ️</h4>
    <div style='position: relative; display: inline-block;'>
        <span style='text-decoration: underline dotted; cursor: help;' title="
🔍 What this does:
- Lets you ask natural questions about customer meetings.
- Retrieves similar past conversations using vector search (FAISS).
- Sends them to an LLM to generate a relevant answer.

🛡️ Why restricted context?
- Ensures privacy and domain relevance.
- Keeps responses tightly aligned to internal transcripts.

🚀 What could enhance this?
- Add conversational memory.
- Enable follow-ups & refining queries (chatbot-style UX).

🏠 Why Streamlit?
- Fast to prototype, but if hosting in-house:
    → Consider Flask, FastAPI + internal auth proxy.

📈 How to scale?
- Horizontally: Distribute FAISS across services or use vector DBs.
- Vertically: Use GPU inference servers or batch incoming LLM requests.
        ">
            Hover here for technical notes
        </span>
    </div>
</div>
""")


from streamlit.components.v1 import html

# Suggested starter questions
suggested_questions = [
    "What solution did Juniper recommend to XYZ Telecom to reduce latency issues?",
    "How does Paragon Automation help XYZ Telecom reduce operational costs?",
    "What kind of failover capability was discussed with XYZ Telecom?",
    "What network performance issues did XYZ Telecom report?",
    "Summarize the key concerns raised by Alpha Cloud Services.",
    "How does Juniper support zero-trust models for Alpha Cloud Services?",
    "What flexibility does Juniper's SD-WAN solution offer to Alpha Cloud Services?",
    "What solutions were recommended for audit trails in Beta Financial Group's network?",
    "How does Juniper ensure sub-second failovers for Beta Financial Group?",
    "What AI-driven recommendations were provided to optimize resource allocation for Beta Financial Group?"
]

st.markdown("### 💡 Suggested Questions")
for question in suggested_questions:
    if st.button(question):
        st.session_state["user_question"] = question  # Sets the input below

# Set default state for input
if "user_question" not in st.session_state:
    st.session_state["user_question"] = ""

# Input box
user_question = st.text_input("Ask your question about customer meetings:", value=st.session_state["user_question"])
###user_question = st.text_input("Ask your question about customer meetings:")

    
if user_question:
    with st.spinner("Processing your query..."):
        output = query_pipeline(user_question)
    st.success("✅ Final Insights Generated:")
    st.text(output)

