import streamlit as st
import pandas as pd
import json
import os
from datetime import datetime
from typing import Dict, Any
import pymupdf
from langchain_openai import ChatOpenAI
from langchain.agents import initialize_agent, AgentType, Tool

st.set_page_config(page_title="🏥 PathologyAI", layout="wide")

# ============================================================================
# API KEY SETUP
# ============================================================================

try:
    OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
    st.sidebar.success("✅ API Key loaded")
except:
    st.error("❌ API Key not found in secrets. Add OPENAI_API_KEY to Streamlit Cloud secrets.")
    st.stop()

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# ============================================================================
# INITIALIZE LLM
# ============================================================================

@st.cache_resource
def get_llm():
    return ChatOpenAI(model="gpt-4o", temperature=0.2, api_key=OPENAI_API_KEY)

llm = get_llm()

# ============================================================================
# FUNCTIONS
# ============================================================================

def extract_text_from_pdf(pdf_file):
    try:
        doc = pymupdf.open(stream=pdf_file.read(), filetype="pdf")
        full_text = ""
        for page_num in range(len(doc)):
            full_text += f"\n--- Page {page_num + 1} ---\n{doc[page_num].get_text()}"
        pages = len(doc)
        doc.close()
        return {"success": True, "text": full_text, "pages": pages, "char_count": len(full_text)}
    except Exception as e:
        return {"success": False, "error": str(e)}

def remove_ocr_artifacts(text: str) -> Dict[str, Any]:
    prompt = f"""Remove OCR artifacts (UUIDs, barcodes, garbled text) from this report:

{text}

Return JSON: {{"artifacts_found": [{{"type": "...", "text": "..."}}], "cleaned_text": "..."}}"""
    
    try:
        response = llm.invoke(prompt)
        return json.loads(response.content.strip().strip('`').replace('json\n', ''))
    except:
        return {"artifacts_found": [], "cleaned_text": text}

def validate_medical_term(term: str) -> Dict[str, Any]:
    prompt = f"""Is "{term}" valid medical terminology?

Return JSON: {{"is_valid": true/false, "correct_term": "...", "confidence": 0.95}}"""
    
    try:
        response = llm.invoke(prompt)
        return json.loads(response.content.strip().strip('`'))
    except:
        return {"is_valid": False, "correct_term": term}

def correct_with_context(text: str, word: str) -> Dict[str, Any]:
    context = text[max(0, text.find(word)-100):text.find(word)+200] if word in text else text[:300]
    prompt = f"""Correct "{word}" in context: {context}

Return JSON: {{"suggestions": [{{"word": "...", "confidence": 0.95}}]}}"""
    
    try:
        response = llm.invoke(prompt)
        return json.loads(response.content.strip().strip('`'))
    except:
        return {"suggestions": []}

def extract_all_pathology_sections(text: str) -> Dict[str, Any]:
    prompt = f"""Extract all sections from this pathology report as JSON:

{text}

Return: {{"specimen": {{"section_found": true, "full_text": "..."}}, "clinical_information": {{...}}, "gross_description": {{...}}, "microscopic_description": {{...}}, "final_diagnosis": {{...}}, "immunohistochemistry": {{...}}, "interpretation": {{...}}, "comments": {{...}}, "addendum": {{...}}, "synoptic_report": {{...}}, "original_report": {{"total_sections_found": 0}}}}"""
    
    try:
        response = llm.invoke(prompt)
        result = json.loads(response.content.strip().strip('`').replace('json\n', '').replace('```', ''))
        if 'original_report' not in result:
            result['original_report'] = {}
        result['original_report']['total_sections_found'] = sum(1 for k,v in result.items() if isinstance(v, dict) and v.get('section_found'))
        return result
    except:
        return {"original_report": {"total_sections_found": 0}}

# ============================================================================
# CREATE AGENT
# ============================================================================

@st.cache_resource
def get_agent():
    
    tools = [
        Tool(
            name="RemoveArtifacts",
            func=lambda t: json.dumps(remove_ocr_artifacts(t), indent=2),
            description="Removes OCR artifacts. Input: text. Returns: JSON."
        ),
        Tool(
            name="ValidateMedicalTerm",
            func=lambda t: json.dumps(validate_medical_term(t), indent=2),
            description="Validates medical term. Input: term. Returns: JSON."
        ),
        Tool(
            name="ExtractAllSections",
            func=lambda t: json.dumps(extract_all_pathology_sections(t), indent=2),
            description="Extracts sections. Input: text. Returns: JSON."
        )
    ]
    
    return initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=False,
        handle_parsing_errors=True,
        max_iterations=10
    )

agent = get_agent()

# ============================================================================
# MAIN UI
# ============================================================================

st.title("🏥 PathologyAI - Agentic System")
st.markdown("Process TCGA Pathology Reports with LangChain Multi-Agent AI")

st.sidebar.markdown("### 🤖 LangChain Tools")
st.sidebar.markdown("- RemoveArtifacts\n- ValidateMedicalTerm\n- ExtractAllSections")

uploaded = st.file_uploader("📤 Upload TCGA Pathology Report PDF", type="pdf")

if uploaded:
    
    with st.spinner("Extracting..."):
        extract_result = extract_text_from_pdf(uploaded)
    
    if extract_result["success"]:
        raw_text = extract_result["text"]
        st.success(f"✅ Extracted {extract_result['pages']} pages ({extract_result['char_count']:,} chars)")
        
        with st.expander("📝 Preview"):
            st.code(raw_text[:500])
        
        if st.button("🚀 Process", type="primary", use_container_width=True):
            
            with st.spinner("🤖 Agent processing..."):
                task = f"""Process this report: remove artifacts, extract sections, tell me diagnosis/site/laterality/grade/stage

{raw_text}"""
                
                try:
                    agent_out = agent.invoke({"input": task})
                    st.success("✅ Done!")
                    st.info(agent_out['output'])
                except Exception as e:
                    st.error(f"Error: {e}")
                    agent_out = None
            
            # Get data
            cleaned_res = remove_ocr_artifacts(raw_text)
            cleaned_text = cleaned_res['cleaned_text']
            sections = extract_all_pathology_sections(cleaned_text)
            
            # Metrics
            col1,col2,col3,col4 = st.columns(4)
            col1.metric("Original", f"{len(raw_text):,}")
            col2.metric("Cleaned", f"{len(cleaned_text):,}")
            col3.metric("Artifacts", len(cleaned_res.get('artifacts_found',[])))
            col4.metric("Sections", sections.get('original_report',{}).get('total_sections_found',0))
            
            # Tabs
            tab1,tab2,tab3 = st.tabs(["📄 Sections","📥 Downloads","🔬 Compare"])
            
            with tab1:
                for key in ['specimen','clinical_information','gross_description','microscopic_description','final_diagnosis','immunohistochemistry','addendum']:
                    sec = sections.get(key,{})
                    if sec.get('section_found'):
                        with st.expander(f"{key.replace('_',' ').title()} ({len(sec.get('full_text','')):,} chars)"):
                            st.text(sec.get('full_text',''))
            
            with tab2:
                st.download_button("📥 Original", raw_text, "original.txt", use_container_width=True)
                st.download_button("✨ Cleaned", cleaned_text, "cleaned.txt", use_container_width=True)
                
                csv = []
                for k in ['specimen','clinical_information','gross_description','microscopic_description','final_diagnosis','immunohistochemistry']:
                    s = sections.get(k,{})
                    csv.append({'section':k, 'text':s.get('full_text','')})
                
                st.download_button("📊 CSV", pd.DataFrame(csv).to_csv(index=False), "sections.csv", use_container_width=True)
            
            with tab3:
                c1,c2 = st.columns(2)
                c1.markdown("**Original**")
                c1.text_area("O", raw_text[:1000], height=300, label_visibility="collapsed")
                c2.markdown("**Cleaned**")
                c2.text_area("C", cleaned_text[:1000], height=300, label_visibility="collapsed")
                
                st.markdown("**Artifacts:**")
                for a in cleaned_res.get('artifacts_found',[])[:5]:
                    st.text(f"• {a.get('type')}: {a.get('text','')[:50]}...")

else:
    st.info("Upload a PDF to start")
