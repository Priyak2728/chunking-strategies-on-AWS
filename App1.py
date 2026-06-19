import streamlit as st
import boto3
import os

st.title("🟥 Default Chunking (App1)")

knowledge_base_id = "8VZPNGCNGW"
region_name = os.getenv("AWS_REGION", "us-east-1")

agent = boto3.client("bedrock-agent-runtime", region_name=region_name)
runtime = boto3.client("bedrock-runtime", region_name=region_name)

def display_retrieval(results):
    st.subheader("Retrieved Chunks")
    for i, r in enumerate(results[:5]):
        text = r.get("content", {}).get("text", "")
        source = r.get("location", {}).get("s3Location", {}).get("uri", "N/A")

        st.markdown(f"**Chunk {i+1}**")
        st.write(text[:400])
        st.caption(source)
        st.divider()

user_input = st.text_input("Ask something:")

if st.button("Send") and user_input:
    res = agent.retrieve(
        knowledgeBaseId=knowledge_base_id,
        retrievalQuery={"text": user_input}
    )

    results = res.get("retrievalResults", [])
    display_retrieval(results)

    context = "\n".join([r["content"]["text"] for r in results[:5]])

    response = runtime.converse(
        modelId="anthropic.claude-3-haiku-20240307-v1:0",
        messages=[{
            "role": "user",
            "content": [{"text": f"Answer using this context:\n{context}\n\nQuestion: {user_input}"}]
        }]
    )

    answer = response["output"]["message"]["content"][0]["text"]

    st.subheader("Final Answer")
    st.write(answer)