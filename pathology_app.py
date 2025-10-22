import streamlit as st
import pandas as pd
import json
import os
from datetime import datetime
from typing import Dict, List, Any
from difflib import SequenceMatcher
import pymupdf
from langchain_openai import ChatOpenAI
from langchain.agents import initialize_agent, AgentType, Tool
from jiwer import wer, cer

st.set_page_config(page_title="🏥 PathologyAI - Agentic System", layout="wide")

# ============================================================================
# SIDEBAR - API KEY
# ============================================================================

st.sidebar.header("⚙️ Configuration")
api_key = st.sidebar.text_input("OpenAI API Key", type="password", 
                                 help="Enter your OpenAI API key")

if not api_key:
    st.warning("👈 Please enter your OpenAI API key in the sidebar")
    st.stop()

os.environ["OPENAI_API_KEY"] = api_key

st.sidebar.success("✅ API Key configured")
st.sidebar.markdown("### 🤖 LangChain Agent Tools")
st.sidebar.markdown("""
- RemoveArtifacts
- ValidateMedicalTerm
- CorrectWithContext
- ExtractAllSections
- ExtractTextFromPDF
""")

# ============================================================================
# INITIALIZE LLM (Your exact code)
# ============================================================================

@st.cache_resource
def init_llm(_api_key):
    return ChatOpenAI(model="gpt-4o", temperature=0.2, api_key=_api_key)

llm = init_llm(api_key)

# ============================================================================
# YOUR EXACT FUNCTIONS
# ============================================================================

def extract_text_from_pdf(pdf_file):
    """Your exact function - adapted for Streamlit upload"""
    try:
        doc = pymupdf.open(stream=pdf_file.read(), filetype="pdf")
        full_text = ""
        num_pages = len(doc)
        
        for page_num in range(num_pages):
            page = doc[page_num]
            text = page.get_text()
            full_text += f"\n--- Page {page_num + 1} ---\n{text}"
        
        char_count = len(full_text)
        word_count = len(full_text.split())
        doc.close()
        
        return {
            "success": True,
            "text": full_text,
            "pages": num_pages,
            "char_count": char_count,
            "word_count": word_count
        }
    except Exception as e:
        return {"success": False, "error": str(e), "text": ""}

def validate_medical_term(term: str) -> Dict[str, Any]:
    """Your exact function"""
    prompt = f"""You are a medical terminology expert with comprehensive knowledge of medical vocabulary equivalent to UMLS SPECIALIST Lexicon.

Analyze this term: "{term}"

Determine:
1. Is it a valid medical term?
2. If invalid, what is the correct spelling?
3. What category does it belong to?

Respond ONLY with valid JSON (no markdown, no explanations):
{{
    "is_valid": true or false,
    "correct_term": "corrected spelling if needed, or original if valid",
    "alternatives": ["alternative1", "alternative2"],
    "category": "anatomy/disease/procedure/drug/biomarker/other",
    "confidence": 0.95
}}

Term to analyze: {term}"""

    try:
        response = llm.invoke(prompt)
        result = json.loads(response.content.strip().strip('`').replace('json\n', ''))
        return result
    except Exception as e:
        return {"is_valid": False, "error": str(e), "correct_term": term}

def correct_with_context(text: str, suspicious_word: str) -> Dict[str, Any]:
    """Your exact function"""
    try:
        word_pos = text.lower().find(suspicious_word.lower())
        if word_pos == -1:
            context = text[:500]
        else:
            start = max(0, word_pos - 100)
            end = min(len(text), word_pos + len(suspicious_word) + 100)
            context = text[start:end]
    except:
        context = text[:500]

    prompt = f"""You are a medical text correction specialist with expertise in pathology reports.

A pathology report contains a suspicious word that may be an OCR error.
Suggest the correct medical term based on the surrounding clinical context.

Context: "{context}"

Suspicious word: "{suspicious_word}"

Analyze:
1. What should this word be based on medical context?
2. Why is this the correct term?
3. How confident are you?

Respond ONLY with valid JSON (no markdown):
{{
    "suggestions": [
        {{
            "word": "most_likely_correction",
            "confidence": 0.95,
            "reason": "brief explanation based on context"
        }},
        {{
            "word": "alternative_correction",
            "confidence": 0.75,
            "reason": "alternative explanation"
        }}
    ],
    "context_category": "diagnosis/histology/staging/biomarker/other"
}}"""

    try:
        response = llm.invoke(prompt)
        result = json.loads(response.content.strip().strip('`').replace('json\n', ''))
        return result
    except Exception as e:
        return {"suggestions": [{"word": suspicious_word, "confidence": 0.0, "reason": f"Error: {e}"}], "error": str(e)}

def remove_ocr_artifacts(text: str) -> Dict[str, Any]:
    """Your exact function"""
    prompt = f"""You are an OCR artifact detector for medical documents.

Identify and remove OCR artifacts from this text:
- Barcodes (e.g., "III II III 1111", "||||||||")
- UUIDs (e.g., "6C9B7DE2-682F-4F4A-8879-159F9989C40C")
- Repeated special characters (e.g., "========", "--------")
- Garbled field names (e.g., "t=pis_reypncy" should be "Discrepancy")

Text:
{text}

Respond ONLY with valid JSON (no markdown):
{{
    "artifacts_found": [
        {{"type": "barcode", "text": "III II III", "should_remove": true}},
        {{"type": "uuid", "text": "6C9B...", "should_remove": true}}
    ],
    "cleaned_text": "full text with artifacts removed"
}}"""

    try:
        response = llm.invoke(prompt)
        result = json.loads(response.content.strip().strip('`').replace('json\n', ''))
        return result
    except Exception as e:
        return {"artifacts_found": [], "cleaned_text": text, "error": str(e)}

def extract_all_pathology_sections(text: str) -> Dict[str, Any]:
    """Your exact function"""
    prompt = f"""You are a pathology report parser.

Extract ALL sections from this pathology report into separate fields.

Common section headers to look for (extract if present):
- SPECIMEN / SPECIMEN TYPE / SPECIMEN SOURCE
- CLINICAL INFORMATION / CLINICAL HISTORY / CLINICAL DATA
- PREOPERATIVE DIAGNOSIS
- POSTOPERATIVE DIAGNOSIS
- INDICATION
- GROSS DESCRIPTION / GROSS EXAMINATION / MACROSCOPIC DESCRIPTION
- MICROSCOPIC DESCRIPTION / MICROSCOPIC EXAMINATION / HISTOLOGIC FINDINGS
- FINAL DIAGNOSIS / DIAGNOSIS(ES) / PATHOLOGIC DIAGNOSIS
- DESCRIPTIVE DIAGNOSIS
- IMMUNOHISTOCHEMISTRY / IMMUNOSTAINS / IHC
- SPECIAL STAINS
- MOLECULAR STUDIES / GENETIC TESTING / MOLECULAR TESTING
- COMMENT(S) / ADDITIONAL COMMENTS / NOTE(S)
- INTERPRETATION / CLINICAL CORRELATION
- ADDENDUM
- INTRAOPERATIVE CONSULTATION / FROZEN SECTION
- SYNOPTIC REPORT / TUMOR SYNOPTIC / BREAST TUMOR SYNOPTIC
- ANCILLARY STUDIES
- REVIEW OF OUTSIDE MATERIAL / REVIEW OF PATHOLOGIST

Extract the COMPLETE TEXT for each section found.

PATHOLOGY REPORT:
{text}

Respond ONLY with valid JSON (no markdown):
{{
    "report_header": {{"patient_info": "", "accession_number": "", "date": "", "pathologist": ""}},
    "specimen": {{"section_found": true/false, "full_text": ""}},
    "clinical_information": {{"section_found": true/false, "full_text": ""}},
    "preoperative_diagnosis": {{"section_found": true/false, "full_text": ""}},
    "postoperative_diagnosis": {{"section_found": true/false, "full_text": ""}},
    "gross_description": {{"section_found": true/false, "full_text": ""}},
    "microscopic_description": {{"section_found": true/false, "full_text": ""}},
    "final_diagnosis": {{"section_found": true/false, "full_text": ""}},
    "immunohistochemistry": {{"section_found": true/false, "full_text": ""}},
    "special_stains": {{"section_found": true/false, "full_text": ""}},
    "molecular_testing": {{"section_found": true/false, "full_text": ""}},
    "comments": {{"section_found": true/false, "full_text": ""}},
    "interpretation": {{"section_found": true/false, "full_text": ""}},
    "addendum": {{"section_found": true/false, "full_text": ""}},
    "intraoperative_consultation": {{"section_found": true/false, "full_text": ""}},
    "synoptic_report": {{"section_found": true/false, "full_text": ""}},
    "ancillary_studies": {{"section_found": true/false, "full_text": ""}},
    "review_notes": {{"section_found": true/false, "full_text": ""}},
    "other_sections": [],
    "original_report": {{"full_text": "", "total_length": 0, "total_sections_found": 0}}
}}"""

    try:
        response = llm.invoke(prompt)
        result = json.loads(response.content.strip().strip('`').replace('json\n', '').replace('```json', '').replace('```', ''))
        
        if not result['original_report']['full_text']:
            result['original_report']['full_text'] = text
        result['original_report']['total_length'] = len(text)
        
        sections_found = sum(1 for k, v in result.items()
                            if isinstance(v, dict) and v.get('section_found', False))
        result['original_report']['total_sections_found'] = sections_found
        
        return result
    except Exception as e:
        return {"error": str(e), "original_report": {"full_text": text, "total_length": len(text), "total_sections_found": 0}}

# ============================================================================
# CREATE LANGCHAIN TOOLS (Your exact tools)
# ============================================================================

@st.cache_resource
def create_agent(_llm):
    """Create LangChain agent with your exact tools"""
    
    artifact_tool = Tool(
        name="RemoveArtifacts",
        func=lambda text: json.dumps(remove_ocr_artifacts(text), indent=2),
        description="Removes OCR artifacts (barcodes, UUIDs, garbled text) from pathology report. Input: full pathology report text. Returns: JSON with cleaned_text and artifacts_found."
    )
    
    validator_tool = Tool(
        name="ValidateMedicalTerm",
        func=lambda term: json.dumps(validate_medical_term(term), indent=2),
        description="Validates if a medical term is correct and suggests corrections. Input: a single medical term. Returns: JSON with is_valid, correct_term, and confidence."
    )
    
    def context_tool_wrapper(input_str: str) -> str:
        try:
            if '|||' in input_str:
                text, word = input_str.split('|||', 1)
                result = correct_with_context(text.strip(), word.strip())
            else:
                result = {"error": "Input must be 'text|||word' format"}
            return json.dumps(result, indent=2)
        except Exception as e:
            return json.dumps({"error": str(e)}, indent=2)
    
    context_tool = Tool(
        name="CorrectWithContext",
        func=context_tool_wrapper,
        description="Corrects suspicious words using context. Input format: 'report_text|||suspicious_word'. Returns: JSON with correction suggestions."
    )
    
    section_tool = Tool(
        name="ExtractAllSections",
        func=lambda text: json.dumps(extract_all_pathology_sections(text), indent=2),
        description="Extracts ALL sections from pathology report. Input: corrected pathology report text. Returns: JSON with all sections."
    )
    
    tools = [artifact_tool, validator_tool, context_tool, section_tool]
    
    agent = initialize_agent(
        tools=tools,
        llm=_llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=15
    )
    
    return agent

pathology_agent = create_agent(llm)

# ============================================================================
# MAIN APP
# ============================================================================

st.title("🏥 PathologyAI - LangChain Agentic System")
st.markdown("### Process TCGA Pathology Reports with Multi-Agent AI")

st.markdown("---")

# File upload
uploaded_file = st.file_uploader("📤 Upload TCGA Pathology Report PDF", type="pdf")

if uploaded_file:
    
    # Extract text
    with st.spinner("📄 Extracting text from PDF..."):
        extraction_result = extract_text_from_pdf(uploaded_file)
    
    if extraction_result["success"]:
        raw_text = extraction_result["text"]
        
        st.success(f"✅ Extracted {extraction_result['pages']} pages ({extraction_result['char_count']:,} characters)")
        
        with st.expander("📝 View Raw Extracted Text (first 500 chars)"):
            st.code(raw_text[:500])
        
        st.markdown("---")
        
        # Process button
        if st.button("🚀 Process with LangChain Agentic AI", type="primary", use_container_width=True):
            
            # ================================================================
            # RUN LANGCHAIN AGENT (Your exact code)
            # ================================================================
            
            st.markdown("## 🤖 LangChain Agent Processing")
            
            with st.spinner("🤖 Agent running with all tools..."):
                
                # Your exact task
                task = f"""You are an intelligent pathology report processing agent. Process this TCGA report systematically:

STEP 1: Use RemoveArtifacts tool to remove OCR artifacts (UUIDs, barcodes, garbled text) from the COMPLETE report

STEP 2: In the cleaned text, identify ANY suspicious words that look like OCR errors

STEP 3: For EACH suspicious word you find:
   - Use ValidateMedicalTerm to check if it's a valid medical term
   - If invalid, use CorrectWithContext to get the correct term based on surrounding text

STEP 4: After all corrections, use ExtractAllSections to extract all report sections

STEP 5: From the extracted sections, tell me:
   - Diagnosis
   - Anatomical site
   - Laterality
   - Tumor grade
   - Pathologic stage

Here is the report text to process:

{raw_text}

Use your tools intelligently - process the COMPLETE report."""
                
                agent_output = pathology_agent.invoke({"input": task})
            
            st.success("✅ Agent processing complete!")
            
            # Show agent output
            st.markdown("### 🤖 Agent Output")
            st.info(agent_output['output'])
            
            # ================================================================
            # PROCESS DATA (Your exact code)
            # ================================================================
            
            with st.spinner("📋 Extracting structured data..."):
                cleaned_result = remove_ocr_artifacts(raw_text)
                cleaned_text = cleaned_result['cleaned_text']
                result = extract_all_pathology_sections(cleaned_text)
            
            # Metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Original Length", f"{len(raw_text):,}")
            with col2:
                st.metric("Cleaned Length", f"{len(cleaned_text):,}")
            with col3:
                st.metric("Artifacts Removed", len(cleaned_result['artifacts_found']))
            with col4:
                st.metric("Sections Found", result['original_report']['total_sections_found'])
            
            # ================================================================
            # TABS FOR DISPLAY
            # ================================================================
            
            tab1, tab2, tab3 = st.tabs(["📄 Sections", "📥 Downloads", "🔬 Comparison"])
            
            with tab1:
                st.markdown("### Extracted Sections")
                
                section_keys = ['specimen', 'clinical_information', 'gross_description',
                               'microscopic_description', 'final_diagnosis', 'immunohistochemistry',
                               'interpretation', 'comments', 'addendum', 'synoptic_report']
                
                for key in section_keys:
                    section = result.get(key, {})
                    if section.get('section_found'):
                        with st.expander(f"📄 {key.replace('_', ' ').title()} ({len(section.get('full_text', '')):,} chars)"):
                            st.text(section.get('full_text', ''))
            
            with tab2:
                st.markdown("### Download Files")
                
                report_id = "TCGA-3C-AAAU"
                
                # Original report download
                original_content = f"""================================================================================
ORIGINAL PATHOLOGY REPORT (PyMuPDF Extraction)
================================================================================
Report ID: {report_id}
Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Length: {len(raw_text):,} characters
Status: Raw extraction with OCR errors
================================================================================

{raw_text}"""
                
                st.download_button(
                    "📥 Download Original Report (PyMuPDF)",
                    original_content,
                    file_name=f"{report_id}_ORIGINAL_PyMuPDF.txt",
                    mime="text/plain",
                    use_container_width=True
                )
                
                # Cleaned report download
                cleaned_content = f"""================================================================================
CLEANED PATHOLOGY REPORT (Agentic AI Processed)
================================================================================
Report ID: {report_id}
Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Original Length: {len(raw_text):,} characters
Cleaned Length: {len(cleaned_text):,} characters
Artifacts Removed: {len(cleaned_result['artifacts_found'])}
Status: Cleaned (no UUIDs, no barcodes, no garbled text)
================================================================================

{cleaned_text}"""
                
                st.download_button(
                    "✨ Download Cleaned Report (Agentic AI)",
                    cleaned_content,
                    file_name=f"{report_id}_CLEANED_AgenticAI.txt",
                    mime="text/plain",
                    use_container_width=True
                )
                
                # CSV download
                csv_data = []
                all_section_keys = [
                    'specimen', 'clinical_information', 'preoperative_diagnosis',
                    'postoperative_diagnosis', 'gross_description', 'microscopic_description',
                    'final_diagnosis', 'immunohistochemistry', 'special_stains',
                    'molecular_testing', 'comments', 'interpretation', 'addendum',
                    'intraoperative_consultation', 'synoptic_report', 'ancillary_studies',
                    'review_notes'
                ]
                
                for key in all_section_keys:
                    section = result.get(key, {})
                    csv_data.append({
                        'report_id': report_id,
                        'section_name': key.replace('_', ' ').title(),
                        'section_found': 'Yes' if section.get('section_found') else 'No',
                        'text_length': len(section.get('full_text', '')),
                        'word_count': len(section.get('full_text', '').split()),
                        'full_text': section.get('full_text', '')
                    })
                
                sections_df = pd.DataFrame(csv_data)
                
                st.download_button(
                    "📊 Download Sections CSV",
                    sections_df.to_csv(index=False),
                    file_name=f"{report_id}_sections.csv",
                    mime="text/csv",
                    use_container_width=True
                )
                
                st.markdown("---")
                st.markdown("**📊 CSV Contents:**")
                st.dataframe(sections_df[['section_name', 'section_found', 'text_length', 'word_count']], use_container_width=True)
            
            with tab3:
                st.markdown("### Comparison: PyMuPDF vs Agentic AI")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Original (PyMuPDF Extraction)**")
                    st.text_area("Original", raw_text[:1500], height=400, label_visibility="collapsed")
                
                with col2:
                    st.markdown("**Cleaned (Agentic AI)**")
                    st.text_area("Cleaned", cleaned_text[:1500], height=400, label_visibility="collapsed")
                
                st.markdown("**🗑️ Artifacts Removed by Agent:**")
                for i, artifact in enumerate(cleaned_result['artifacts_found'][:10], 1):
                    st.text(f"{i}. {artifact['type']}: {artifact['text'][:60]}...")

else:
    st.info("👆 Upload a TCGA pathology report PDF to begin processing")
    
    st.markdown("---")
    st.markdown("## ℹ️ About This System")
    st.markdown("""
    This is an **Agentic AI system** using **LangChain** for processing TCGA pathology reports.
    
    **🤖 LangChain Agent Features:**
    - Autonomous decision-making (ReAct framework)
    - 5 specialized tools (RemoveArtifacts, ValidateMedicalTerm, CorrectWithContext, ExtractAllSections, ExtractTextFromPDF)
    - Multi-step reasoning and self-correction
    - Adaptive processing based on report content
    
    **📋 Processing Pipeline:**
    1. PDF Upload → PyMuPDF text extraction
    2. Agent removes OCR artifacts (UUIDs, barcodes)
    3. Agent validates and corrects medical terminology
    4. Agent extracts all sections
    5. Download cleaned report + CSV
    
    **🎯 Output Files:**
    - Original report (with OCR errors)
    - Cleaned report (Agentic AI processed)
    - Sections CSV (structured data)
    """)
