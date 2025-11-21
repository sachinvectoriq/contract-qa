
"""
Contract Management System - Simplified with Standard Rate Card Template
Only Contract tab is functional, other tabs under development
"""

import json
import hashlib
import pytesseract
import streamlit as st
import pandas as pd
from typing import List, Optional
from datetime import datetime
from openai import AzureOpenAI
import pdfplumber
from pdf2image import convert_from_path, convert_from_bytes
from dataclasses import dataclass, asdict
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.core.credentials import AzureKeyCredential
from dotenv import load_dotenv
import os

load_dotenv()

from dataclasses import dataclass
from typing import List, Optional
from datetime import date

@dataclass
class RateTier:
    from_qty: float
    to_qty: Optional[float]
    price: float

@dataclass
class RateCard:
    category: str
    unit: str
    currency: str
    region: Optional[str]
    included_qty: Optional[float]
    tiers: List[RateTier]
    effective_from: date
    effective_to: Optional[date]

@dataclass
class InvoiceLine:
    line_ref: str
    description: str
    quantity: float
    unit_price: float
    amount: float
    category: Optional[str] = None
    confidence: float = 1.0

@dataclass
class Invoice:
    vendor: str
    invoice_number: str
    invoice_date: date
    total_amount: float
    lines: List[InvoiceLine]

@dataclass
class Discrepancy:
    invoice_number: str
    line_ref: str
    category: str
    expected_amount: float
    actual_amount: float
    delta: float
    tolerance_applied: str
    reason: str


# ==================== AZURE CLIENT ====================
@st.cache_resource
def get_azure_client():
    """Get configured Azure OpenAI client"""
    try:
        api_key = st.secrets["AZURE_OPENAI_API_KEY"]
        endpoint = st.secrets["AZURE_OPENAI_ENDPOINT"]
        deployment = st.secrets["AZURE_OPENAI_DEPLOYMENT"]
        
        if not api_key or not endpoint:
            st.error("⚠️ Azure OpenAI credentials not configured.")
            return None, None
        
        client = AzureOpenAI(
            api_key=api_key,
            api_version="2024-02-15-preview",
            azure_endpoint=endpoint
        )
        return client, deployment
    except Exception as e:
        st.error(f"Error initializing Azure client: {e}")
        return None, None

def safe_float(value, default=0.0):
    """Safely convert value to float"""
    try:
        if value is None:
            return default
        return float(value)
    except (ValueError, TypeError):
        return default


# ==================== PDF EXTRACTION ====================
def extract_pdf_text(pdf_input) -> str:
    """Extract text from PDF using pdfplumber"""
    text = ""
    try:
        if hasattr(pdf_input, 'read'):
            with pdfplumber.open(pdf_input) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
                    
                    tables = page.extract_tables()
                    for table in tables:
                        for row in table:
                            text += " | ".join(str(cell) if cell else "" for cell in row) + "\n"
        else:
            with pdfplumber.open(pdf_input) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
                    
                    tables = page.extract_tables()
                    for table in tables:
                        for row in table:
                            text += " | ".join(str(cell) if cell else "" for cell in row) + "\n"
        
        return text
    except Exception as e:
        st.error(f"Error extracting PDF text: {e}")
        return ""


def extract_text_ocr_from_file(pdf_input, dpi=300, poppler_path=None):
    """Extract text using OCR (fallback for image-based PDFs)"""
    custom_config = r"--oem 3 --psm 6"
    full_text = []

    try:
        if isinstance(pdf_input, str):
            images = convert_from_path(pdf_input, dpi=dpi, fmt='png', poppler_path=poppler_path)
        else:
            if hasattr(pdf_input, 'read'):
                pdf_bytes = pdf_input.read()
                pdf_input.seek(0)
            else:
                pdf_bytes = pdf_input

            images = convert_from_bytes(pdf_bytes, dpi=dpi, fmt='png', poppler_path=poppler_path)

        for i, image in enumerate(images):
            text = pytesseract.image_to_string(image, config=custom_config)
            full_text.append(text)

        return '\n'.join(full_text)
    
    except Exception as e:
        st.error(f"OCR Error: {e}")
        return ""
    
# def extract_text_ocr_azure():    
    
#     # Azure Document Intelligence endpoint & key
#     doc_intell_endpoint = os.getenv("doc_intell_endpoint")
#     doc_intell_key = os.getenv("doc_intell_key")
#     file_path = "HCW contract.pdf"
    
#     # Initialize client
#     client = DocumentIntelligenceClient(endpoint=doc_intell_endpoint, credential=AzureKeyCredential(doc_intell_key))
    
#     # Open the PDF
#     with open(file_path, "rb") as f:
#         # --- FIX IS HERE: Change 'document=f' to 'body=f' ---
#         poller = client.begin_analyze_document("prebuilt-read", body=f)
#         result = poller.result()
    
#     # Extract OCR text only
#     ocr_text = ""
#     for page in result.pages:
#         for line in page.lines:
#             ocr_text += line.content + "\n"
    
#     print(ocr_text)

def extract_text_ocr_azure(pdf_input, doc_intell_endpoint=None, doc_intell_key=None):
    """
    Extract text using Azure Document Intelligence OCR
    
    Args:
        pdf_input: Can be either:
                   - str: file path to PDF
                   - UploadedFile: Streamlit uploaded file object
                   - file-like object with read() method
        doc_intell_endpoint: Azure endpoint (optional, defaults to env var)
        doc_intell_key: Azure key (optional, defaults to env var)
    
    Returns:
        str: Extracted OCR text
    """
    
    # Azure Document Intelligence endpoint & key
    endpoint =st.secrets["AZURE_DI_ENDPOINT"]
    key = st.secrets["AZURE_DI_KEY"]
    if not endpoint or not key:
        raise ValueError("Azure Document Intelligence credentials not found")
    
    # Initialize client
    client = DocumentIntelligenceClient(
        endpoint=endpoint, 
        credential=AzureKeyCredential(key)
    )
    
    try:
        # Handle file path (string)
        if isinstance(pdf_input, str):
            with open(pdf_input, "rb") as f:
                poller = client.begin_analyze_document("prebuilt-read", body=f)
                result = poller.result()
        
        # Handle Streamlit UploadedFile or any file-like object
        else:
            # For Streamlit UploadedFile, it's already in memory
            # Reset pointer to beginning if it has seek method
            if hasattr(pdf_input, 'seek'):
                pdf_input.seek(0)
            
            poller = client.begin_analyze_document("prebuilt-read", body=pdf_input)
            result = poller.result()
            
            # Reset pointer again for potential reuse
            if hasattr(pdf_input, 'seek'):
                pdf_input.seek(0)
        
        # Extract OCR text only
        ocr_text = ""
        for page in result.pages:
            for line in page.lines:
                ocr_text += line.content + "\n"
        
        return ocr_text
    
    except Exception as e:
        st.error(f"Azure Document Intelligence OCR Error: {e}")
        return ""


# ==================== AZURE AI HELPERS ====================
def azure_chat_json(client, deployment: str, system: str, user: str, max_tokens: int = 1000) -> str:
    """Make Azure OpenAI chat completion call"""
    try:
        resp = client.chat.completions.create(
            model=deployment,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user}
            ],
            temperature=0.0,
            max_tokens=max_tokens
        )
        return resp.choices[0].message.content
    except Exception as e:
        st.error(f"Azure API Error: {e}")
        return "{}"


# ==================== DYNAMIC RATE CARD EXTRACTION ====================
def extract_rate_card_dynamic(contract_text: str, client, deployment) -> tuple:
    """
    Extract rate card from contract by:
    1. Inferring the table structure (column headers as they appear in contract)
    2. Extracting data based on inferred structure
    3. Making a second LLM call to find missing/non-tabular rates and return them as 'extras'
    
    Returns: (list of dicts with rate card data, list of column headers, list of extras)
    """
    system = (
        "You are a contract parsing assistant. Your task is to find and extract rate card/pricing tables from contracts.\n\n"
        "STEP 1: Identify if there is a rate card, pricing table, or payment schedule in the contract.\n"
        "STEP 2: Extract the EXACT column headers as they appear in the contract table.\n"
        "STEP 3: Extract all rows of data from that table.\n\n"
        "Return ONLY a JSON object with this structure:\n"
        "{\n"
        '  "columns": ["Column1", "Column2", "Column3"],\n'
        '  "data": [\n'
        '    {"Column1": "value1", "Column2": "value2", "Column3": "value3"},\n'
        '    {"Column1": "value4", "Column2": "value5", "Column3": "value6"}\n'
        '  ]\n'
        "}\n\n"
        "Rules:\n"
        "- Important Use EXACT column names from the contract (e.g., 'Sr. No', 'Description', 'Unit Price', 'Item #', etc.) if sl no not present include one Do not miss out any column until or unless there is no value under that column\n"
        "- Extract ALL rows from the pricing/rate card table\n"
        "- Keep all values as strings exactly as they appear\n"
        "- as a general format there will be item/{name  provided} in the column and include other columns which may be required \n"
        "- Importantly Mandatory serial no column \"sl no.\" , field related to item name as given otherwise as item should be present \n"
        "- If a cell is empty, use empty string ''\n"
        "- Return ONLY the JSON object, no additional text\n"
        "- If no rate card table is found, return: {\"columns\": [], \"data\": []}"
        "- if make sure you haven't mistakenly included items in columns other wise table will have columns>rows"
        "- ideally columns would have common things with all items such as rate description etc mostly go with the structure we provide that will contain in table format"
    )

    user = (
        "Contract Text:\n"
        f"{contract_text}\n\n"
        "Find the rate card/pricing table and extract its structure and data."
    )

    out = azure_chat_json(client, deployment, system, user, max_tokens=3000)
    
    try:
        # Extract JSON object from first LLM response
        start = out.index("{")
        end = out.rindex("}") + 1
        json_str = out[start:end]
        result = json.loads(json_str)
        
        columns = result.get('columns', [])
        data = result.get('data', [])

        # --- SECOND LLM CALL: find missing/non-tabular rates (extras) ---
        try:
            system2 = (
                "You are a contract assistant. You have the full contract text and the table that was already extracted. "
                "Your job is to find any rate/pricing statements in the contract text that are NOT represented in the provided table. "
                "certain rates/fearures would be unlimited or similar should also be included"
                "Return ONLY a JSON object with key 'extras' whose value is a list of objects like "
                '[{\"label\": \"MIS\", \"value\": \"250{with appropriate currency}\"}, {\"label\": \"inspection\", \"value\": \"200 {with appropriate currency}\"}]. '
                "If there are no missing rates, return {\"extras\": []}."
            )

            user2 = (
                f"Contract Text:\n{contract_text}\n\n"
                f"Extracted Table JSON:\n{json_str}\n\n"
                "Return missing/non-tabular rates as described."
            )

            out2 = azure_chat_json(client, deployment, system2, user2, max_tokens=1000)

            # Extract JSON from second response
            if isinstance(out2, str) and "{" in out2 and "}" in out2:
                s2 = out2.index("{")
                e2 = out2.rindex("}") + 1
                json2 = out2[s2:e2]
                parsed2 = json.loads(json2)
                extras = parsed2.get('extras', [])
            else:
                extras = []
        except Exception:
            extras = []

        return data, columns, extras
    
    except Exception as e:
        st.error(f"Failed to parse rate card extraction: {e}")
        st.code(out, language='text')
        return [], [], []

    
def extract_invoice_dynamic(invoice_text: str, client, deployment) -> tuple:
    """
    Extract invoice data by:
    1. Identifying invoice header information (invoice number, date, vendor, etc.)
    2. Inferring the line items table structure
    3. Extracting all line items with their details
    
    Returns: (dict with invoice metadata, list of dicts with line items, list of column headers)
    """
    system = (
        "You are an invoice parsing assistant. Your task is to extract invoice information including header details and line items.\n\n"
        "STEP 1: Extract invoice header information (Invoice Number, Invoice Date, Vendor/Supplier Name, Total Amount, etc.)\n"
        "STEP 2: Identify the line items table and extract EXACT column headers as they appear.\n"
        "STEP 3: Extract all rows of line item data from the table.\n\n"
        "Return ONLY a JSON object with this structure:\n"
        "{\n"
        '  "metadata": {\n'
        '    "invoice_number": "INV-123",\n'
        '    "invoice_date": "2024-01-15",\n'
        '    "vendor_name": "ABC Corp",\n'
        '    "total_amount": "1000.00",\n'
        '    "tax_amount": "100.00",\n'
        '    "subtotal": "900.00",\n'
        '    "currency": "USD",\n'
        '    "po_number": "PO-456"\n'
        '  },\n'
        '  "columns": ["Item", "Description", "Quantity", "Unit Price", "Amount"],\n'
        '  "line_items": [\n'
        '    {"Item": "1", "Description": "Service A", "Quantity": "10", "Unit Price": "50.00", "Amount": "500.00"},\n'
        '    {"Item": "2", "Description": "Service B", "Quantity": "5", "Unit Price": "80.00", "Amount": "400.00"}\n'
        '  ]\n'
        "}\n\n"
        "Rules:\n"
        "- Extract metadata fields as they appear in the invoice header\n"
        "- Use EXACT column names from the invoice table (e.g., 'Item', 'Description', 'Qty', 'Rate', 'Amount', etc.)\n"
        "- Extract ALL line item rows from the invoice\n"
        "- Keep all values as strings exactly as they appear\n"
        "- If a field is not found, use empty string ''\n"
        "- Ensure serial number or item number column is present (add 'Item' if missing)\n"
        "- Return ONLY the JSON object, no additional text\n"
        "- If no invoice data is found, return: {\"metadata\": {}, \"columns\": [], \"line_items\": []}"
    )

    user = (
        "Invoice Text:\n"
        f"{invoice_text}\n\n"
        "Extract the invoice metadata and line items table with its structure and data."
    )

    out = azure_chat_json(client, deployment, system, user, max_tokens=3000)
    
    try:
        # Extract JSON object
        start = out.index("{")
        end = out.rindex("}") + 1
        json_str = out[start:end]
        result = json.loads(json_str)
        
        metadata = result.get('metadata', {})
        columns = result.get('columns', [])
        line_items = result.get('line_items', [])
        
        return metadata, line_items, columns
    
    except Exception as e:
        st.error(f"Failed to parse invoice extraction: {e}")
        st.code(out, language='text')
        return {}, [], []
from typing import List, Optional, Tuple
import re

# Assume ChargeCategory, ReasonCode, RateCard, InvoiceLine, Invoice, Discrepancy, RateTier, etc. are already defined as in your code.

def classify_invoice_line(description: str) -> Tuple[str, float]:
    """Classify invoice line using keywords (simplified, no OpenAI call)"""
    patterns = {
        "RECURRING": [r'monthly', r'subscription', r'recurring', r'service fee'],
        "USAGE": [r'usage', r'data', r'bandwidth'],
        "ONE_TIME": [r'setup', r'installation', r'onboarding'],
        "DISCOUNT": [r'discount', r'credit'],
        "SURCHARGE_FEE": [r'surcharge', r'fee'],
        "TAX": [r'tax', r'vat', r'gst']
    }
    desc = description.lower()
    for cat, keys in patterns.items():
        for kw in keys:
            if re.search(kw, desc):
                return cat, 0.9
    return "ONE_TIME", 0.5  # default fallback

def calculate_expected_amount(quantity: float, rate_card: RateCard) -> float:
    """Calculate expected amount based on rate card tiers"""
    if not rate_card.tiers:
        return 0
    total = 0
    remaining = quantity
    if rate_card.included_qty:
        remaining = max(0, quantity - rate_card.included_qty)
    for tier in rate_card.tiers:
        if remaining <= 0:
            break
        tier_qty = remaining
        if tier.to_qty is not None:
            tier_qty = min(remaining, tier.to_qty - tier.from_qty)
        total += tier_qty * tier.price
        remaining -= tier_qty
    return total


def match_invoice_to_contract(
    invoice_line_items: list,
    invoice_line_item_columns: list,
    contracts: list,
    client,
    deployment
) -> list:
    
    discrepancies = []

    # Build a simple structure for rate cards from contracts
    rate_cards = []
    for contract in contracts:
        rate_cards.extend(contract.get('rate_cards', []))  # Assuming 'rate_cards' list of dicts

    # Iterate invoice lines
    for line in invoice_line_items:
        description = line.get('Description', '')
        quantity = float(line.get('Quantity', 0))
        amount = float(line.get('Amount', 0))
        line_ref = line.get('Line', '')

        # Basic category classification (pattern matching or default)
        # You can implement or call your basic classifier function here, simplified
        category = 'ONE_TIME'  # Default category placeholder

        # Find matching rate card for category and effective date
        # Here you can implement simplified checks if you have invoice date info else skip

        matching_rate = None
        for rate in rate_cards:
            if rate.get('category') == category:
                # Skipping effective date check for simplification
                matching_rate = rate
                break

        # Calculate expected amount (simplified)
        expected_amount = 0
        if matching_rate and 'tiers' in matching_rate:
            tiers = matching_rate['tiers']
            remaining_qty = max(0, quantity - (matching_rate.get('included_qty') or 0))
            for tier in tiers:
                if remaining_qty <= 0:
                    break
                from_qty = tier.get('from_qty', 0)
                to_qty = tier.get('to_qty')
                price = tier.get('price', 0)
                tier_qty = remaining_qty
                if to_qty is not None:
                    tier_qty = min(remaining_qty, to_qty - from_qty)
                expected_amount += tier_qty * price
                remaining_qty -= tier_qty

        # Compare and append discrepancies if delta exceeds tolerance
        delta = abs(expected_amount - amount)
        tolerance = max(0.50, amount * 0.005)  # $0.50 or 0.5%

        if delta > tolerance:
            reason = "RATE_MISMATCH" if delta >= 1 else "ROUNDING"
            discrepancies.append({
                'line_ref': line_ref,
                'category': category,
                'expected_amount': expected_amount,
                'actual_amount': amount,
                'delta': amount - expected_amount,
                'tolerance': f"${tolerance:.2f}",
                'reason': reason
            })

    return discrepancies

        




# ==================== STREAMLIT UI ====================
def main():
    st.set_page_config(
        page_title="Contract Management System", 
        page_icon="📄", 
        layout="wide"
    )
    
    st.title("📄 Contract Management System")
    st.markdown("**Dynamic Rate Card Extraction: Automatically infers table structure from each contract**")
    
    # Initialize session state
    if 'contracts' not in st.session_state:
        st.session_state.contracts = []
    
    # Main tabs
    tab_contracts, tab_invoices, tab_reconcile, tab_export = st.tabs(
        ["📄 Contracts", "📑 Invoices ", "🔍 Reconcile (Under Dev)", "📊 Export (Under Dev)"]
    )
    
    # ==================== CONTRACTS TAB ====================
    with tab_contracts:
        st.header("Contract Management")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            uploaded_contract = st.file_uploader(
                "Upload Contract PDF",
                type=['pdf'],
                key="contract_upload",
                help="Upload a contract PDF to automatically extract and infer rate card structure"
            )
            
            use_ocr = st.checkbox(
                "Use OCR (for image-based PDFs)", 
                value=False, 
                key="contract_ocr"
            )
            
            if uploaded_contract:
                if st.button("📊 Extract Rate Card", type="primary"):
                    client, deployment = get_azure_client()
                    if not client:
                        st.error("Cannot process without Azure OpenAI configuration")
                        st.stop()
                    
                    with st.spinner("Extracting contract rate card..."):
                        # Extract text
                        if use_ocr:
                            # contract_text = extract_text_ocr_from_file(uploaded_contract)
                            contract_text = extract_text_ocr_azure(uploaded_contract)
                        else:
                            contract_text = extract_pdf_text(uploaded_contract)
                        

                        if not contract_text:
                            st.error("Failed to extract text from contract")
                            st.stop()
                        
                        # Extract rate cards dynamically
                        rate_card_data, columns, extra_rates = extract_rate_card_dynamic(contract_text, client, deployment)
                        
                        if rate_card_data and columns:
                            # Store in session state (including extras found via follow-up LLM call)
                            contract_data = {
                                'id': hashlib.md5(contract_text.encode()).hexdigest()[:8],
                                'filename': uploaded_contract.name,
                                'upload_date': datetime.now(),
                                'rate_card_data': rate_card_data,
                                'rate_card_columns': columns,
                                'extra_rates': extra_rates,   # <-- added
                                'raw_text': contract_text[:1000]
                            }
                            st.session_state.contracts.append(contract_data)

                            
                            st.success(f"✅ Extracted {len(rate_card_data)} rate card items with {len(columns)} columns!")
                            st.info(f"**Inferred Columns:** {', '.join(columns)}")
                        else:
                            st.error("❌ No rate card table found in contract")
        
        with col2:
            if st.session_state.contracts:
                st.metric("Contracts Loaded", len(st.session_state.contracts))
                total_items = sum(len(c['rate_card_data']) for c in st.session_state.contracts)
                st.metric("Total Rate Card Items", total_items)
                latest_contract = st.session_state.contracts[-1]
                st.info(f"**Latest:** {latest_contract['filename']}\n**ID:** {latest_contract['id']}")
        
        # Display contracts and rate cards
        if st.session_state.contracts:
            st.markdown("---")
            st.subheader("📋 Contract Rate Cards")
            
            for idx, contract in enumerate(st.session_state.contracts):
                with st.expander(
                    f"Contract {idx+1}: {contract['filename']}", 
                    expanded=(idx==len(st.session_state.contracts)-1)
                ):
                    st.write(f"**Contract ID:** {contract['id']}")
                    st.write(f"**Upload Date:** {contract['upload_date'].strftime('%Y-%m-%d %H:%M:%S')}")
                    st.write(f"**Rate Card Items:** {len(contract['rate_card_data'])}")
                    st.write(f"**Columns:** {', '.join(contract['rate_card_columns'])}")
                    
                    # Display rate cards with dynamic columns
                    if contract.get('rate_card_data') and contract.get('rate_card_columns'):
                        rate_df = pd.DataFrame(contract['rate_card_data'])
                        
                        # Reorder columns to match the inferred order
                        rate_df = rate_df[contract['rate_card_columns']]
                        
                        st.dataframe(rate_df, use_container_width=True, hide_index=True)
                        
                        # Download option
                        csv = rate_df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="📥 Download Rate Card CSV",
                            data=csv,
                            file_name=f"rate_card_{contract['id']}.csv",
                            mime="text/csv"
                        )
                        if contract.get('extra_rates'):
                            st.markdown("**Additional Rates / Non-tabular Items:**")
                            for extra in contract['extra_rates']:
                                label = extra.get('label') or extra.get('name') or ''
                                value = extra.get('value') or extra.get('rate') or ''
                                if label or value:
                                    st.write(f"- {label} — {value}")
                    else:
                        st.info("No rate cards extracted from this contract")
                    
                    # Show raw text preview
                    with st.expander("📄 Raw Text Preview"):
                        st.code(contract.get('raw_text', ''), language='text')
        
        else:
            st.info("👆 Upload a contract PDF to get started")
    
    # ==================== INVOICES TAB (Under Development) ====================
    with tab_invoices:
        st.header("📑 Invoice Processing")
        
        # Initialize invoice session state
        if 'invoices' not in st.session_state:
            st.session_state.invoices = []
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            uploaded_invoice = st.file_uploader(
                "Upload Invoice PDF",
                type=['pdf'],
                key="invoice_upload",
                help="Upload an invoice PDF to automatically extract header information and line items"
            )
            
            use_ocr_invoice = st.checkbox(
                "Use OCR (for image-based PDFs)", 
                value=False, 
                key="invoice_ocr"
            )
            
            if uploaded_invoice:
                if st.button("📊 Extract Invoice Data", type="primary", key="extract_invoice"):
                    client, deployment = get_azure_client()
                    if not client:
                        st.error("Cannot process without Azure OpenAI configuration")
                        st.stop()
                    
                    with st.spinner("Extracting invoice data..."):
                        # Extract text
                        if use_ocr_invoice:
                            invoice_text = extract_text_ocr_azure(uploaded_invoice)
                        else:
                            invoice_text = extract_pdf_text(uploaded_invoice)
                        
                        if not invoice_text:
                            st.error("Failed to extract text from invoice")
                            st.stop()
                        
                        # Extract invoice data dynamically
                        metadata, line_items, columns = extract_invoice_dynamic(invoice_text, client, deployment)
                        
                        if line_items and columns:
                            # Store in session state
                            invoice_data = {
                                'id': hashlib.md5(invoice_text.encode()).hexdigest()[:8],
                                'filename': uploaded_invoice.name,
                                'upload_date': datetime.now(),
                                'metadata': metadata,
                                'line_items': line_items,
                                'line_item_columns': columns,
                                'raw_text': invoice_text[:1000]
                            }
                            st.session_state.invoices.append(invoice_data)
                            
                            st.success(f"✅ Extracted invoice with {len(line_items)} line items and {len(columns)} columns!")
                            
                            # Display key metadata
                            if metadata:
                                st.info(
                                    f"**Invoice #:** {metadata.get('invoice_number', 'N/A')} | "
                                    f"**Date:** {metadata.get('invoice_date', 'N/A')} | "
                                    f"**Vendor:** {metadata.get('vendor_name', 'N/A')} | "
                                    f"**Total:** {metadata.get('currency', '')} {metadata.get('total_amount', 'N/A')}"
                                )
                            st.info(f"**Line Item Columns:** {', '.join(columns)}")
                        else:
                            st.error("❌ No invoice data found in the document")
        
        with col2:
            if st.session_state.invoices:
                st.metric("Invoices Loaded", len(st.session_state.invoices))
                total_items = sum(len(inv['line_items']) for inv in st.session_state.invoices)
                st.metric("Total Line Items", total_items)
                
                # Calculate total invoice amounts
                total_amount = 0
                for inv in st.session_state.invoices:
                    amount_str = inv['metadata'].get('total_amount', '0')
                    try:
                        # Remove currency symbols and convert to float
                        amount = float(''.join(c for c in amount_str if c.isdigit() or c == '.'))
                        total_amount += amount
                    except:
                        pass
                
                st.metric("Total Invoice Amount", f"${total_amount:,.2f}")
        
        # Display invoices
        if st.session_state.invoices:
            st.markdown("---")
            st.subheader("📋 Extracted Invoices")
            
            for idx, invoice in enumerate(st.session_state.invoices):
                with st.expander(
                    f"Invoice {idx+1}: {invoice['filename']}", 
                    expanded=(idx==len(st.session_state.invoices)-1)
                ):
                    # Display metadata
                    st.write(f"**Invoice ID:** {invoice['id']}")
                    st.write(f"**Upload Date:** {invoice['upload_date'].strftime('%Y-%m-%d %H:%M:%S')}")
                    
                    if invoice.get('metadata'):
                        st.markdown("##### 📄 Invoice Details")
                        meta = invoice['metadata']
                        
                        col_a, col_b, col_c = st.columns(3)
                        with col_a:
                            st.write(f"**Invoice Number:** {meta.get('invoice_number', 'N/A')}")
                            st.write(f"**Invoice Date:** {meta.get('invoice_date', 'N/A')}")
                        with col_b:
                            st.write(f"**Vendor:** {meta.get('vendor_name', 'N/A')}")
                            st.write(f"**PO Number:** {meta.get('po_number', 'N/A')}")
                        with col_c:
                            st.write(f"**Subtotal:** {meta.get('currency', '')} {meta.get('subtotal', 'N/A')}")
                            st.write(f"**Tax:** {meta.get('currency', '')} {meta.get('tax_amount', 'N/A')}")
                            st.write(f"**Total:** {meta.get('currency', '')} {meta.get('total_amount', 'N/A')}")
                    
                    # Display line items
                    if invoice.get('line_items') and invoice.get('line_item_columns'):
                        st.markdown("##### 📊 Line Items")
                        line_items_df = pd.DataFrame(invoice['line_items'])
                        
                        # Reorder columns to match the inferred order
                        line_items_df = line_items_df[invoice['line_item_columns']]
                        
                        st.dataframe(line_items_df, use_container_width=True, hide_index=True)
                        
                        # Download option
                        csv = line_items_df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="📥 Download Line Items CSV",
                            data=csv,
                            file_name=f"invoice_items_{invoice['id']}.csv",
                            mime="text/csv",
                            key=f"download_invoice_{idx}"
                        )
                    else:
                        st.info("No line items extracted from this invoice")
                    
                    # Quick match button
                    if st.session_state.contracts:
                        if st.button(f"🔍 Quick Match to Contracts", key=f"match_{idx}"):
                            client, deployment = get_azure_client()
                            if client:
                                with st.spinner("Matching invoice to contracts..."):
                                    match_results = match_invoice_to_contract(
                                        invoice['line_items'],
                                        invoice['line_item_columns'],
                                        st.session_state.contracts,
                                        client,
                                        deployment
                                    )
                                    
                                    # Store match results
                                    invoice['match_results'] = match_results
                                    
                                    # Display results
                                    st.success(f"Found {len(match_results)} matches!")

                                    
                                    if match_results.get('matches'):
                                        matches_df = pd.DataFrame(match_results['matches'])
                                        st.dataframe(matches_df, use_container_width=True, hide_index=True)
                                    
                                    if match_results.get('unmatched'):
                                        st.warning(f"⚠️ {len(match_results['unmatched'])} unmatched items")
                    
                    # Show raw text preview
                    with st.expander("📄 Raw Text Preview"):
                        st.code(invoice.get('raw_text', ''), language='text')
        
        else:
            st.info("👆 Upload an invoice PDF to get started")
    
    # ==================== RECONCILE TAB (Under Development) ====================
    with tab_reconcile:
        st.header("🔍 Reconciliation Analysis")
        client, deployment = get_azure_client()
        
        # Show button first
        if st.button("🔄 Reconcile Invoice with Contract", type="primary", use_container_width=True):
            
            # Fetch data from state AFTER button click
            has_invoice = 'invoices' in st.session_state and st.session_state.invoices
            has_contract = 'contracts' in st.session_state and st.session_state.contracts
            
            # Handle errors after button click
            if not has_invoice or not has_contract:
                st.error("❌ Cannot perform reconciliation")
                if not has_invoice:
                    st.warning("📄 Invoice data is missing. Please upload and process an invoice first.")
                if not has_contract:
                    st.warning("📑 Contract data is missing. Please upload and process a contract first.")
            else:
                # Data is available, proceed with reconciliation
                recon_contract = st.session_state.contracts
                recon_invoice = st.session_state.invoices
                
                system = """
                You are a smart assistant that helps to reconcile invoice data with contract data.
                consider fields/items from invoice and contract and match them, 
                if it doesn't match then ignore the missing contract field keeping the invoice field. 
                Output format respectively:
                [{"item": "...",
                "invoice_value": "...", 
                "contract_value": "...",
                "difference": "Yes/no"}]
                
                
                """
                user = f"""
                ***Here is the invoice data:
                {recon_invoice}
                
                ***Here is the contract data:
                {recon_contract}

                """
                
                try:
                    with st.spinner("🔄 Reconciling invoice with contract..."):
                        out = azure_chat_json(client, deployment, system, user, max_tokens=3000)
                    
                    # Store result in session state to persist after button click
                    if out:
                    # If output is a string, extract JSON from markdown code blocks
                        if isinstance(out, str):
                            # Remove markdown code blocks and extra text
                            import re
                            # Find JSON array in the string
                            json_match = re.search(r'\[\s*{.*}\s*\]', out, re.DOTALL)
                            if json_match:
                                json_str = json_match.group(0)
                                out = json.loads(json_str)
                            else:
                                st.error("❌ Could not extract JSON from LLM response")
                                out = None
                        
                        if out:
                            st.session_state.reconciliation_result = out
                            st.success("✅ Reconciliation completed successfully!")
                        else:
                            st.error("❌ Reconciliation failed: No valid output received")
                    else:
                        st.error("❌ Reconciliation failed: No output received from LLM")
                        
                except json.JSONDecodeError as e:
                    st.error(f"❌ Error parsing JSON response: {str(e)}")
                except Exception as e:
                    st.error(f"❌ Error during reconciliation: {str(e)}")
        
        # Display the table as per the json output from llm call
        if 'reconciliation_result' in st.session_state and st.session_state.reconciliation_result:
            
            # Convert JSON output to DataFrame and display as table
            df_reconciliation = pd.DataFrame(st.session_state.reconciliation_result)
            
            st.subheader("📊 Reconciliation Results")
            st.dataframe(
                df_reconciliation,
                use_container_width=True,
                hide_index=True
            )
            
            # Optional: Add summary metrics
            if 'difference' in df_reconciliation.columns:
                total_items = len(df_reconciliation)
                differences = df_reconciliation['difference'].str.lower().eq('yes').sum()
                matches = total_items - differences
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Items", total_items)
                with col2:
                    st.metric("✅ Matches", matches)
                with col3:
                    st.metric("⚠️ Differences", differences)
    # with tab_reconcile:
    #     st.header("🔍 Reconciliation Analysis")
        
    #     # 1) Handle if invoice and contract is not present
    #     has_invoice = 'invoice' in st.session_state and st.session_state.invoice
    #     has_contract = 'contract' in st.session_state and st.session_state.contract
        
    #     if not has_invoice or not has_contract:
    #         st.warning("⚠️ Missing data for reconciliation")
    #         if not has_invoice:
    #             st.info("📄 Please upload and process an invoice first")
    #         if not has_contract:
    #             st.info("📑 Please upload and process a contract first")
    #     else:
    #         recon_contract = st.session_state.contract
    #         recon_invoice = st.session_state.invoice
            
    #         system = """
    #         You are a smart assistant that helps to reconcile invoice data with contract data.
    #         consider fields/items from invoice and contract and match them, 
    #         if it doesn't match then ignore the missing contract field keeping the invoice field. 
    #         Output format respectively:
    #         [{"item": "...",
    #         "invoice_value": "...", 
    #         "contract_value": "...",
    #         "difference": "Yes/no"}]
            
            
    #         """
    #         user = f"""
    #         ***Here is the invoice data:
    #         {json.dumps(recon_invoice, indent=2)}
            
    #         ***Here is the contract data:
    #         {json.dumps(recon_contract, indent=2)}

    #         """
            
    #         with st.spinner("🔄 Reconciling invoice with contract..."):
    #             out = azure_chat_json(client, deployment, system, user, max_tokens=3000)
            
    #         # 2) Display the table as per the json output from llm call
    #         if out:
    #             st.success("✅ Reconciliation completed!")
                
    #             # Convert JSON output to DataFrame and display as table
    #             df_reconciliation = pd.DataFrame(out)
                
    #             st.subheader("📊 Reconciliation Results")
    #             st.dataframe(
    #                 df_reconciliation,
    #                 use_container_width=True,
    #                 hide_index=True
    #             )
                
    #             # Optional: Add summary metrics
    #             if 'difference' in df_reconciliation.columns:
    #                 total_items = len(df_reconciliation)
    #                 differences = df_reconciliation['difference'].str.lower().eq('yes').sum()
    #                 matches = total_items - differences
                    
    #                 col1, col2, col3 = st.columns(3)
    #                 with col1:
    #                     st.metric("Total Items", total_items)
    #                 with col2:
    #                     st.metric("✅ Matches", matches)
    #                 with col3:
    #                     st.metric("⚠️ Differences", differences)
    #         else:
    #             st.error("❌ Failed to get reconciliation results")

        
    # ==================== EXPORT TAB (Under Development) ====================
    with tab_export:
        st.header("📊 Export Data")
        st.warning("🚧 This feature is under development")
        st.info("Coming soon: Export all processed data in various formats")
        
        # Allow exporting rate cards if available
        if st.session_state.contracts:
            st.markdown("---")
            st.subheader("Available Now: Export Rate Cards")
            
            # Export each contract separately (since they have different structures)
            for idx, contract in enumerate(st.session_state.contracts):
                if contract.get('rate_card_data'):
                    st.write(f"**Contract {idx+1}: {contract['filename']}**")
                    
                    df = pd.DataFrame(contract['rate_card_data'])
                    if contract.get('rate_card_columns'):
                        df = df[contract['rate_card_columns']]
                    
                    csv = df.to_csv(index=False).encode('utf-8')
                    json_str = df.to_json(orient='records', indent=2).encode('utf-8')
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.download_button(
                            label=f"📥 CSV - {contract['filename']}",
                            data=csv,
                            file_name=f"rate_card_{contract['id']}.csv",
                            mime="text/csv",
                            key=f"csv_{contract['id']}"
                        )
                    with col2:
                        st.download_button(
                            label=f"📥 JSON - {contract['filename']}",
                            data=json_str,
                            file_name=f"rate_card_{contract['id']}.json",
                            mime="application/json",
                            key=f"json_{contract['id']}"
                        )
                    
                    st.markdown("---")


if __name__ == "__main__":

    main()
