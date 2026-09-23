import re
import os
import uuid
import cv2
import json
import time
import base64
import numpy as np
import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv

# ==============================
# CONFIG
# ==============================
load_dotenv()

# On Streamlit Community Cloud, secrets come from st.secrets, not a .env file.
for _key in ("AZURE_OPENAI_KEY", "AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_MODEL"):
    if not os.environ.get(_key) and _key in st.secrets:
        os.environ[_key] = st.secrets[_key]

st.set_page_config(page_title="Planogram AI", layout="wide", page_icon="📊")

UPLOAD_FOLDER = "uploads"
JSON_FOLDER = "outputs"

ACTUAL_JSON = "actual_extracted.json"
PLANOGRAM_JSON = "planogram_extracted.json"
OUTPUT_JSON = "planogram_vs_actual_comparison.json"
PRICE_JSON = "price_extracted.json"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
JSON_FOLDER = os.path.join(BASE_DIR, "outputs")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(JSON_FOLDER, exist_ok=True)


client = OpenAI(
    api_key=os.environ.get("AZURE_OPENAI_KEY"),
    base_url=os.environ.get("AZURE_OPENAI_ENDPOINT")
)

# ==============================
# TOKEN USAGE / COST TRACKING
# ==============================
# Azure OpenAI rate for the deployed model, $ per 1M tokens.
# Update these if the underlying deployment model changes.
PRICE_PER_1M_INPUT = float(os.environ.get("PRICE_PER_1M_INPUT", 0.75))
PRICE_PER_1M_CACHED_INPUT = float(os.environ.get("PRICE_PER_1M_CACHED_INPUT", 0.08))
PRICE_PER_1M_OUTPUT = float(os.environ.get("PRICE_PER_1M_OUTPUT", 4.50))

USAGE_LOG_FILE = os.path.join(JSON_FOLDER, "usage_log.json")


def log_usage(step, image_label, response):
    usage = getattr(response, "usage", None)
    if usage is None:
        return None

    input_tokens = getattr(usage, "input_tokens", 0) or 0
    cached_tokens = getattr(getattr(usage, "input_tokens_details", None), "cached_tokens", 0) or 0
    output_tokens = getattr(usage, "output_tokens", 0) or 0
    billed_input_tokens = max(input_tokens - cached_tokens, 0)

    cost = (
        billed_input_tokens / 1_000_000 * PRICE_PER_1M_INPUT
        + cached_tokens / 1_000_000 * PRICE_PER_1M_CACHED_INPUT
        + output_tokens / 1_000_000 * PRICE_PER_1M_OUTPUT
    )

    entry = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "step": step,
        "image": image_label,
        "input_tokens": input_tokens,
        "cached_tokens": cached_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "cost_usd": round(cost, 6),
    }

    log = st.session_state.setdefault("usage_log", [])
    log.append(entry)

    try:
        existing = []
        if os.path.exists(USAGE_LOG_FILE):
            with open(USAGE_LOG_FILE, "r", encoding="utf-8") as f:
                existing = json.load(f)
        existing.append(entry)
        with open(USAGE_LOG_FILE, "w", encoding="utf-8") as f:
            json.dump(existing, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

    return entry

if "unique_no" not in st.session_state:
    st.session_state["unique_no"] = uuid.uuid4().hex[:8]

# ==============================
# CLEAN PRODUCT NAME
# ==============================
def clean_product_name(name: str) -> str:
    if not isinstance(name, str) or not name.strip():
        return "Unidentified Product"
    
    name = str(name).strip()
    # Remove any HTML garbage
    name = re.sub(r'<[^>]+>', '', name)
    name = re.sub(r'class="[^"]*"', '', name, flags=re.IGNORECASE)
    name = re.sub(r'div class=', '', name, flags=re.IGNORECASE)
    name = re.sub(r'\s+', ' ', name).strip()
    
    return name if len(name) > 3 else "Unidentified Product"

# ==============================
# HELPERS
# ==============================

def save_uploaded_file(uploaded_file, filename):
    # Split name and extension
    name, ext = os.path.splitext(filename)
    unique_no = st.session_state.get("unique_no")
    unique_filename = f"{name}_{unique_no}{ext}"
    path = os.path.join(UPLOAD_FOLDER, unique_filename)
    with open(path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return path

def save_uploaded_file_high_quality(uploaded_file, base_name: str):
    unique_no = st.session_state["unique_no"]
    name, ext = os.path.splitext(base_name)
    unique_filename = f"{name}_{unique_no}{ext.lower()}"
    save_path = os.path.join(UPLOAD_FOLDER, unique_filename)

    # Open with PIL and preserve maximum quality
    img = Image.open(uploaded_file)
    img = ImageOps.exif_transpose(img)   # Fix orientation

    # Optional: Resize only if extremely large (to avoid token explosion)
    max_dim = 2500  # OpenAI high-res sweet spot is around 2048-2500
    if max(img.size) > max_dim:
        ratio = max_dim / max(img.size)
        new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
        img = img.resize(new_size, Image.Resampling.LANCZOS)

    # Save with maximum quality
    if img.mode in ("RGBA", "P"):
        img = img.convert("RGB")
    
    img.save(save_path, quality=95, optimize=True, subsampling=0)
    return save_path

def save_json(result, name, folder="outputs"):

    unique_no = st.session_state.get("unique_no")
    file_name = f"{name}_{unique_no}.json"
    file_path = os.path.join(folder, file_name)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    st.session_state[name] = file_path
    return file_path


def encode_image(image_path):
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")

    if image_path.lower().endswith((".jpg", ".jpeg")):
        return f"data:image/jpeg;base64,{b64}"
    return f"data:image/png;base64,{b64}"

# ==============================
# 3-STEP LLM FUNCTIONS
# ==============================
def extract_planogram(planogram_path):
    # st.info("📋 Step 1: Extracting from Planogram...")
    
    system_prompt = """
You are a highly precise retail shelf image parser.

Your task is to extract products from a real shelf image with STRICT visual grounding.

CRITICAL RULES (MUST FOLLOW):

1. ORDERING (VERY IMPORTANT):
- Always process shelves from TOP to BOTTOM.
- Within each shelf, read products from LEFT to RIGHT.
- Do NOT skip or reorder shelves.

2. NO HALLUCINATION:
- Only extract products that are clearly visible in the image.
- Do NOT guess, infer, or complete missing text.
- If a product is blurry or unreadable, label it as:
  "Unidentified [Category]" (e.g., "Unidentified Beverage", "Unidentified Snack")

3. VISUAL GROUNDING:
- Every product must correspond to an actual visible item.
- Do NOT add products that are not present.
- Do NOT merge multiple products into one.

4. NAMING FORMAT:
- Use short, clean names: Brand + Variant/Flavor + Type
  Example: "Lay's Classic Chips", "Coca-Cola Regular Can"
- Avoid extra words, descriptions, or assumptions.

5. SHELF STRUCTURE:
- Maintain exact shelf grouping.
- Each shelf should be a separate list.
- Preserve product count per shelf as seen.

6. OUTPUT FORMAT (STRICT JSON ONLY):
{
  "shelves": [
    {
      "shelf_number": 1,
      "products": ["Product 1", "Product 2", ...]
    },
    {
      "shelf_number": 2,
      "products": [...]
    }
  ]
}

7. DO NOT:
- Do not explain anything
- Do not add comments
- Do not return anything except JSON
"""

    response = client.responses.create(
        model=os.environ["AZURE_OPENAI_MODEL"],
        input=[
            {"role": "system", "content": [{"type": "input_text", "text": system_prompt}]},
            {"role": "user", "content": [
                {"type": "input_text", "text": "Extract products shelf by shelf from this PLANOGRAM. Return ONLY JSON."},
                {"type": "input_image", "image_url": encode_image(planogram_path)}
            ]}
        ]
    )
    log_usage("Planogram Extraction", os.path.basename(planogram_path), response)

    result = response.output_text.strip()
    if result.startswith("```"):
        result = result.split("```")[1].replace("json", "").strip()

    json_path = save_json(result, "planogram_extracted")
    # st.write(f"📄 JSON saved at: {json_path}")

    data = json.loads(result)
    for shelf in data.get("shelves", []):
        shelf["products"] = [clean_product_name(p) for p in shelf.get("products", [])]
    return data

def extract_actual(actual_path):
    # st.info("📋 Step 1: Extracting from Planogram...")
    
    system_prompt = """
You are a highly precise retail shelf image parser.

Your task is to extract products from a real shelf image with STRICT visual grounding.

CRITICAL RULES (MUST FOLLOW):

1. ORDERING (VERY IMPORTANT):
- Always process shelves from TOP to BOTTOM.
- Within each shelf, read products from LEFT to RIGHT.
- Do NOT skip or reorder shelves.

2. NO HALLUCINATION:
- Only extract products that are clearly visible in the image.
- Do NOT guess, infer, or complete missing text.
- If a product is blurry or unreadable, label it as:
  "Unidentified [Category]" (e.g., "Unidentified Beverage", "Unidentified Snack")

3. VISUAL GROUNDING:
- Every product must correspond to an actual visible item.
- Do NOT add products that are not present.
- Do NOT merge multiple products into one.

4. NAMING FORMAT:
- Use short, clean names: Brand + Variant/Flavor + Type
  Example: "Lay's Classic Chips", "Coca-Cola Regular Can"
- Avoid extra words, descriptions, or assumptions.

5. SHELF STRUCTURE:
- Maintain exact shelf grouping.
- Each shelf should be a separate list.
- Preserve product count per shelf as seen.

6. OUTPUT FORMAT (STRICT JSON ONLY):
{
  "shelves": [
    {
      "shelf_number": 1,
      "products": ["Product 1", "Product 2", ...]
    },
    {
      "shelf_number": 2,
      "products": [...]
    }
  ]
}

7. DO NOT:
- Do not explain anything
- Do not add comments
- Do not return anything except JSON
"""


    response = client.responses.create(
        model=os.environ["AZURE_OPENAI_MODEL"],
        input=[
            {"role": "system", "content": [{"type": "input_text", "text": system_prompt}]},
            {"role": "user", "content": [
                {"type": "input_text", "text": "Extract products shelf by shelf from this actual image. Return ONLY JSON."},
                {"type": "input_image", "image_url": encode_image(actual_path)}
            ]}
        ]
    )
    log_usage("Actual Shelf Extraction", os.path.basename(actual_path), response)

    result = response.output_text.strip()
    if result.startswith("```"):
        result = result.split("```")[1].replace("json", "").strip()

    json_path = save_json(result, "actual_extracted")
    # st.write(f"📄 JSON saved at: {json_path}")

    data = json.loads(result)
    for shelf in data.get("shelves", []):
        shelf["products"] = [clean_product_name(p) for p in shelf.get("products", [])]
    return data

# def extract_actual(actual_path):
#     # st.info("📷 Step 2: Extracting from Actual Shelf...")

#     system_prompt = """
# You are an AI system specialized in retail shelf analysis from images.

# The image may contain blurry or small text. You must carefully extract information without guessing incorrectly.

# Tasks:
# 1. Detect all visible price tags and item name on the shelf.
# 2. Extract the price value (e.g., 4.99, 6.99).
# 3. Handle blurry text:
#    - If the price is slightly blurry but readable, extract it.
#    - If the price is NOT clearly readable, ignore it (do NOT guess).
# 4. Identify the associated product:
#    - Look directly ABOVE the price tag for the product.
#    - Extract product name from visible packaging text.
# 5. Determine:
#    - Shelf level (top = 1st, then 2nd, 3rd, etc.)
#    - Horizontal position (left, center, right)
# 6. Assign a confidence score (0–100) based on:
#    - Text clarity (sharp vs blurry)
#    - Visibility of the price tag
#    - Certainty of product association

# Confidence Guidelines:
# - 90–100: very clear price and product
# - 70–89: readable but slightly blurry
# - 30–69: partially unclear but likely correct
# - Below 30: DO NOT include the item

# Rules:s
# - Only return valid prices (must contain decimals like 4.99)
# - Do NOT hallucinate missing prices or product names
# - If unsure, skip the item entirely
# - Keep product_hint short (2–4 words max)
# - Confidence must be an integer (no % sign)

# Return strictly valid JSON:

# {
#   "prices": [
#     {
#       "price": "4.99",
#       "shelf_level": "2nd",
#       "position": "center",
#       "product_hint": "sourdough bread",
#       "confidence": 85
#     }
#   ]
# }
# """

#     response = client.responses.create(
#         model=os.environ["AZURE_OPENAI_MODEL"],
#         input=[
#             {"role": "system", "content": [{"type": "input_text", "text": system_prompt}]},
#             {"role": "user", "content": [
#                 {"type": "input_text", "text": "Extract all visible products from this ACTUAL shelf photo. Return ONLY JSON."},
#                 {"type": "input_image", "image_url": encode_image(actual_path)}
#             ]}
#         ]
#     )

#     result = response.output_text.strip()
#     if result.startswith("```"):
#         result = result.split("```")[1].replace("json", "").strip()

#     json_path = save_json(result, "actual_extracted")
#     # st.write(f"📄 JSON saved at: {json_path}")

#     data = json.loads(result)
#     for shelf in data.get("shelves", []):
#         shelf["products"] = [clean_product_name(p) for p in shelf.get("products", [])]
#     return data


def extract_sku_counts(shelf_path):
    system_prompt = """
You are a highly precise retail shelf image parser specialized in SKU facing counts.

Your task is to extract EVERY distinct SKU from a real shelf image and COUNT how many
times each SKU appears (i.e. how many facings/units are visible), with STRICT visual grounding.

CRITICAL RULES (MUST FOLLOW):

1. ORDERING:
- Process shelves from TOP to BOTTOM.
- Within each shelf, read products from LEFT to RIGHT.

2. COUNTING:
- Count every visible facing of the same SKU (identical brand + variant + size) as one unit.
- Do NOT count empty gaps or price tags.
- If a SKU repeats across the same shelf (e.g. 5 boxes of the same product side by side), count all of them.

3. NO HALLUCINATION:
- Only count products clearly visible in the image.
- Do NOT guess or invent SKUs that aren't there.
- If a product is blurry or unreadable, label it "Unidentified [Category]" and still count its visible facings.

4. NAMING FORMAT:
- Use short, clean names: Brand + Variant/Flavor + Type (e.g. "Ritz Family Size Crackers").
- The SAME SKU name must be reused consistently whenever it repeats within a shelf.

5. OUTPUT FORMAT (STRICT JSON ONLY):
{
  "shelves": [
    {
      "shelf_number": 1,
      "skus": [
        {"sku": "Product Name", "count": 3},
        {"sku": "Product Name 2", "count": 5}
      ]
    }
  ]
}

6. DO NOT:
- Do not explain anything
- Do not add comments
- Do not return anything except JSON
"""

    response = client.responses.create(
        model=os.environ["AZURE_OPENAI_MODEL"],
        input=[
            {"role": "system", "content": [{"type": "input_text", "text": system_prompt}]},
            {"role": "user", "content": [
                {"type": "input_text", "text": "Extract SKU-wise facing counts shelf by shelf from this shelf image. Return ONLY JSON."},
                {"type": "input_image", "image_url": encode_image(shelf_path)}
            ]}
        ]
    )
    log_usage("SKU Count Extraction", os.path.basename(shelf_path), response)

    result = response.output_text.strip()
    if result.startswith("```"):
        result = result.split("```")[1].replace("json", "").strip()

    save_json(result, "sku_counts")

    data = json.loads(result)
    for shelf in data.get("shelves", []):
        for item in shelf.get("skus", []):
            item["sku"] = clean_product_name(item.get("sku"))
            try:
                item["count"] = int(item.get("count", 0))
            except (TypeError, ValueError):
                item["count"] = 0
    return data


def compare_planogram_vs_actual(planogram_data, actual_data):
    # st.info("🔍 Step 3: Comparing Planogram vs Actual...")

    comparison_prompt = f"""
You are an expert retail planogram compliance analyst specializing in noisy, real-world shelf data.

Your task is to compare the expected planogram with the observed shelf data and produce a structured compliance report.

-----------------------
INPUT DATA
-----------------------
Planogram (Expected):
{json.dumps(planogram_data, indent=2)}

Actual Shelf (Observed):
{json.dumps(actual_data, indent=2)}

-----------------------
MATCHING LOGIC (CRITICAL)
-----------------------
- Perform comparison shelf-by-shelf (level-by-level)
- Match products using FUZZY LOGIC, not exact string match:
  - Ignore case differences
  - Ignore minor spelling variations
  - Ignore brand prefixes/suffixes if core product matches
  - Example:
    - "Coca Cola 500ml" ≈ "Coke 500 ml"
    - "Lays Classic Chips" ≈ "Lays Classic"
- If similarity >= ~70%, consider it a MATCH

-----------------------
POSITION HANDLING
-----------------------
- If same product exists but position differs → still MATCH
- Do NOT penalize for left/center/right differences
- Focus ONLY on product presence per shelf

-----------------------
MISMATCH RULES
-----------------------
- If expected product is missing → mark as:
  "Product A missing"
- If a different product appears instead → mark as:
  "Product A replaced by Product X"
- If extra product exists not in planogram → mark as:
  "Unexpected product X"

-----------------------
PERCENTAGE CALCULATION
-----------------------
For EACH shelf:
- matching_percentage = (matched_items / total_planogram_items) * 100
- non_matching_percentage = 100 - matching_percentage

Overall:
- Aggregate across all shelves using total counts (not average of percentages)

-----------------------
CONFIDENCE SCORING
-----------------------
- 90–100 → clear readable match
- 75–89 → minor fuzziness
- 60–74 → uncertain/partial match
- <60 → poor data quality

-----------------------
OUTPUT FORMAT (STRICT)
-----------------------
Return ONLY valid JSON (no explanation, no markdown):

{{
  "overall_compliance": {{
    "matching_percentage": "XX%",
    "non_matching_percentage": "YY%",
    "confidence": 85
  }},
  "shelves": [
    {{
      "level": "Shelf 1",
      "planogram": ["Product A", "Product B"],
      "actual": ["Product X", "Product Y"],
      "matches": ["Product A"],
      "mismatches": [
        "Product B replaced by Product X"
      ],
      "matching_percentage": "XX%",
      "non_matching_percentage": "YY%",
      "confidence": 90
    }}
  ],
  "summary": "Clear 1-2 line summary highlighting compliance level, major gaps, and missing products."
}}

-----------------------
STRICT RULES
-----------------------
- Output MUST be valid JSON
- Do NOT include explanations or extra text
- Do NOT use arrows (→), HTML, or special characters
- Use SHORT, CLEAN product names (remove noise like size if not essential)
- Do NOT hallucinate products not present in input
- Ensure every shelf in planogram is included in output
- If a shelf has no matches → matches = []
- If a shelf has no mismatches → mismatches = []

"""


    response = client.responses.create(
        model=os.environ["AZURE_OPENAI_MODEL"],
        input=[
            {"role": "system", "content": [{"type": "input_text", "text": comparison_prompt}]}
        ]
    )
    log_usage("Planogram vs Actual Comparison", "(text-only, no image)", response)

    result = response.output_text.strip()
    if result.startswith("```"):
        result = result.split("```")[1].replace("json", "").strip()

    json_path = save_json(result, "planogram_vs_actual_comparison")
    # st.write(f"📄 JSON saved at: {json_path}")

    data = json.loads(result)
    
    # Final cleaning
    for shelf in data.get("shelves", []):
        shelf["planogram"] = [clean_product_name(p) for p in shelf.get("planogram", [])]
        shelf["actual"] = [clean_product_name(p) for p in shelf.get("actual", [])]
    
    return data

class ImageDeblurrer:
    def __init__(self):
        self.psf = None
    
    def estimate_psf(self, image, kernel_size=11):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        
        f_transform = np.fft.fft2(edges)
        f_shift = np.fft.fftshift(f_transform)
        magnitude = np.log(np.abs(f_shift) + 1)
        
        self.psf = cv2.getGaussianKernel(kernel_size, kernel_size/6)
        self.psf = self.psf @ self.psf.T
        self.psf /= self.psf.sum()
        
        return self.psf
    
    def deblur(self, image, iterations=15):
        if self.psf is None:
            self.estimate_psf(image)
        
        img = image.astype(np.float32) / 255.0
        estimate = img.copy()
        
        for _ in range(iterations):
            convolved = cv2.filter2D(estimate, -1, self.psf)
            ratio = img / (convolved + 1e-5)
            estimate *= cv2.filter2D(ratio, -1, self.psf[::-1, ::-1])
        
        return np.clip(estimate * 255, 0, 255).astype(np.uint8)


def load_image_to_base64(image_path):
    with open(image_path, "rb") as img_file:
        image_base64 = base64.b64encode(img_file.read()).decode("utf-8")
    return image_base64


def create_data_url(image_base64, mime="image/jpg"):
    data_url = f"data:{mime};base64,{image_base64}"
    return data_url


SYSTEM_PROMPT_PRICE_EXTR = """
You are a retail shelf price-tag extraction assistant.

Your priority is accuracy over completeness.
Do not guess blurry digits, and do not infer prices from nearby products.

Perform the task in two stages:
1. Identify each visible shelf-edge price tag and transcribe the price exactly as seen.
2. Match each price tag to the most likely product directly above it using horizontal alignment.

Rules:
- Only extract information that is visually supported by the image.
- Never mark a result as High confidence unless:
  - all price digits are clearly readable, and
  - the tag-to-product match is visually unambiguous.
- If the last cents digits are unclear (for example 59 vs 99), mark the price as uncertain and lower confidence.
- If multiple nearby products could belong to the same tag, lower confidence.
- If the price cannot be read reliably, use "unclear" instead of guessing.
- If the product name is partially unreadable, include the readable portion and add "name unclear".
- Be especially careful not to confuse adjacent tags on the same shelf.
- Do not use product familiarity or typical store pricing to guess a price.
- Precision is more important than recall.

Return JSON:
{
 "prices":[
  {"price":"4.99","shelf_level":"2nd","position":"center","product_hint":"bread","confidence":85}
 ]
}
"""

USER_PROMPT_PRICE_EXTR = """
Extract all visible shelf price tags along with the most likely product name and shelf location.

Important:
- First read the price tags exactly.
- Then match each tag to the nearest product above it.
- If a product-to-tag match is ambiguous, lower confidence.
"""


def extract_prices_from_image(image_path):
    
    image_base64 = load_image_to_base64(image_path)
    data_url = create_data_url(image_base64, mime="image/jpg")
    
    response = client.responses.create(
        model='gpt-5.4',
        input=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT_PRICE_EXTR
            },
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": USER_PROMPT_PRICE_EXTR},
                    {"type": "input_image", "image_url": data_url},
                ],
            }
        ]
    )
    log_usage("Price Extraction", os.path.basename(image_path), response)

    result = response.output_text.strip().replace("```json", "").replace("```", "")
    data = json.loads(result)
    
    # with open("price_extracted.json", "w") as f:
    #     json.dump(data, f, indent=2)

    json_path = save_json(data, "price_extracted")

    return data


# def deblur_image(image_path, output_path, iterations=15):
#     """Apply deblurring to image and save result"""
    
#     deblurrer = ImageDeblurrer()
#     img = cv2.imread(image_path)
#     result = deblurrer.deblur(img, iterations=iterations)
#     cv2.imwrite(output_path, result)
    
#     return result

def deblur_image(image_path, output_path=None, iterations=15):
    """Apply deblurring to image and save result"""

    deblurrer = ImageDeblurrer()
    img = cv2.imread(image_path)
    result = deblurrer.deblur(img, iterations=iterations)

    unique_no = st.session_state.get("unique_no")

    # Generate unique output path if not provided
    if output_path is None:
        # unique_name = f"deblurred_{uuid.uuid4().hex[:8]}.jpg"
        unique_name = f"deblurred_{unique_no}.jpg"
        output_path = os.path.join("uploads", unique_name)

    cv2.imwrite(output_path, result)
    return result, output_path

# def extract_prices(path):
#     deblur_image(path, "uploads/deblurred_true.jpg", iterations=12)
#     data = extract_prices_from_image("uploads/deblurred_true.jpg")
#     return data

def extract_prices(path):
    _, deblurred_path = deblur_image(path, iterations=12)
    data = extract_prices_from_image(deblurred_path)
    return data


# ==============================
# MAIN APP
# ==============================

menu = st.radio(
    "",
    ["📊 Dashboard", "💰 Price Extraction", "🔢 SKU Count"],
    horizontal=True
)


# ==============================
# TOKEN USAGE / COST SIDEBAR
# ==============================
with st.sidebar:
    st.header("💵 API Usage & Cost")

    usage_log = st.session_state.get("usage_log", [])

    if usage_log:
        import pandas as pd

        usage_df = pd.DataFrame(usage_log)

        total_cost = usage_df["cost_usd"].sum()
        total_tokens = usage_df["total_tokens"].sum()

        st.metric("Total Cost (this session)", f"${total_cost:.4f}")
        st.metric("Total Tokens", f"{int(total_tokens):,}")

        st.dataframe(
            usage_df[["timestamp", "step", "image", "input_tokens", "cached_tokens", "output_tokens", "cost_usd"]],
            use_container_width=True,
            hide_index=True
        )

        st.download_button(
            "⬇️ Download Usage Log",
            data=json.dumps(usage_log, indent=2),
            file_name="usage_log.json"
        )
    else:
        st.caption("No API calls made yet this session.")


if menu == "🔢 SKU Count":

    st.header("SKU-wise Facing Count")

    if "sku_count_data" not in st.session_state:
        st.session_state.sku_count_data = None

    sku_shelf_file = st.file_uploader(
        "Upload Shelf Image for SKU Count",
        type=["jpg", "jpeg", "png"],
        key="sku_shelf"
    )

    if sku_shelf_file:
        from PIL import Image, ImageOps
        img = Image.open(sku_shelf_file)
        img = ImageOps.exif_transpose(img)
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.image(img, caption="Shelf Image", width=400)

    if st.button("Count SKUs", disabled=not sku_shelf_file):
        with st.spinner("Counting SKUs..."):
            file_path = save_uploaded_file_high_quality(sku_shelf_file, "sku_shelf.jpg")
            st.session_state.sku_count_data = extract_sku_counts(file_path)

    sku_count_data = st.session_state.sku_count_data

    if sku_count_data and "shelves" in sku_count_data:

        import pandas as pd
        import plotly.express as px

        rows = []
        for shelf in sku_count_data["shelves"]:
            for item in shelf.get("skus", []):
                rows.append({
                    "Shelf": shelf.get("shelf_number"),
                    "SKU": item.get("sku"),
                    "Count": item.get("count", 0)
                })

        df = pd.DataFrame(rows)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Shelves", df["Shelf"].nunique() if not df.empty else 0)
        with col2:
            st.metric("Unique SKUs", df["SKU"].nunique() if not df.empty else 0)
        with col3:
            st.metric("Total Facings", int(df["Count"].sum()) if not df.empty else 0)

        if not df.empty:
            st.subheader("SKU Count by Shelf")
            for shelf_no in sorted(df["Shelf"].unique()):
                subset = df[df["Shelf"] == shelf_no].sort_values("Count", ascending=False)
                st.markdown(f"### Shelf {shelf_no}")
                st.dataframe(subset[["SKU", "Count"]], use_container_width=True, hide_index=True)

            st.subheader("Total Facings per SKU")
            totals = df.groupby("SKU", as_index=False)["Count"].sum().sort_values("Count", ascending=False)
            fig = px.bar(
                totals,
                x="SKU",
                y="Count",
                text="Count",
                color="Count",
                color_continuous_scale="Blues"
            )
            fig.update_traces(textposition="outside")
            st.plotly_chart(fig, use_container_width=True)

        st.download_button(
            "⬇️ Download JSON",
            data=json.dumps(sku_count_data, indent=2),
            file_name="sku_counts.json"
        )

    else:
        st.info("Upload a shelf image and click 'Count SKUs'")

elif menu == "💰 Price Extraction":

    st.header("Price Extraction")

    if "price_data" not in st.session_state:
        st.session_state.price_data = None

    uploaded_file_price_extraction = st.file_uploader(
        "Upload Shelf Image for Price Extraction",
        type=["jpg", "png"]
    )

    if uploaded_file_price_extraction:
        from PIL import Image, ImageOps
        img = Image.open(uploaded_file_price_extraction)
        img = ImageOps.exif_transpose(img)
        col1, col2, col3 = st.columns([1, 2, 1])

        with col2:
            st.image(
                img,
                caption="Actual Shelf",
                width=400
            )
        # st.image(uploaded_file_price_extraction, caption="Shelf Image", width=300)  # set width as needed

    if  st.button("Extract Prices",  disabled=not uploaded_file_price_extraction) :
        with st.spinner("Extracting prices..."):
            file_path = save_uploaded_file_high_quality(uploaded_file_price_extraction, "actual_price.jpg")
            st.session_state.price_data = extract_prices(file_path)

    price_data = st.session_state.price_data

    if price_data and "prices" in price_data:

        import pandas as pd
        import plotly.express as px

        df = pd.DataFrame(price_data["prices"])
        df["price"] = pd.to_numeric(df["price"], errors="coerce")

        st.subheader("Price Extraction Framework")
        # ================= GROUP BY SHELF =================
        for shelf in df["shelf_level"].dropna().unique():

            subset = df[df["shelf_level"] == shelf]

            if not subset.empty:
                st.markdown(f"### {str(shelf).capitalize()} Shelf")
                st.dataframe(subset, use_container_width=True)

        col1, col2 = st.columns([1.2, 1])

        # ================= Metrics =================
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Total Prices", len(df))

        with col2:
            avg_conf = int(df["confidence"].mean()) if not df.empty else 0
            st.metric("Avg Confidence", f"{avg_conf}%")

        with col3:
            unique_products = df["product_hint"].nunique() if not df.empty else 0
            st.metric("Unique Products", unique_products)

        # ================= Chart =================
        if not df.empty:

            fig = px.scatter(
                df,
                x="shelf_level",
                y="price",
                size="confidence",
                color="position",
                hover_data=["product_hint"],
                title="Shelf Price Distribution"
            )

            st.plotly_chart(fig, use_container_width=True)

        # ================= Download =================
        st.download_button(
            "⬇️ Download JSON",
            data=json.dumps(price_data, indent=2),
            file_name="price_extracted.json"
        )

    else:
        st.info("Upload image and click 'Extract Prices'")

else:
    st.markdown("""
<div class="planogram-header">
    <span class="logo">📊 Planogram Compliance AI</span>
    <span class="badge">3-STEP ANALYSIS</span>
</div>
""", unsafe_allow_html=True)



    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div style="font-weight:600; margin-bottom:4px;">🗂 Planogram Image</div>', unsafe_allow_html=True)
        planogram_file = st.file_uploader("", type=["jpg", "jpeg", "png"], key="plan")

    with col2:
        st.markdown('<div style="font-weight:600; margin-bottom:4px;">📷 Actual Shelf Photo</div>', unsafe_allow_html=True)
        actual_file = st.file_uploader("", type=["jpg", "jpeg", "png"], key="shelf")

    if planogram_file or actual_file:
        st.markdown('<hr>', unsafe_allow_html=True)
        prev1, prev2 = st.columns(2)
        with prev1:
            if planogram_file:
                st.image(planogram_file, caption="Planogram", width=300)
        with prev2:
            # if actual_file:
            #     st.image(actual_file, caption="Actual Shelf" , width=300)

            from PIL import Image, ImageOps

            if actual_file:
                img = Image.open(actual_file)
                img = ImageOps.exif_transpose(img)  # auto-correct orientation
                st.image(img, caption="Actual Shelf", width=300)    

    run_clicked = st.button("🚀 Run 3-Step Compliance Check", disabled=not (planogram_file and actual_file), type="primary")

    if run_clicked:
        if not planogram_file or not actual_file:
            st.error("Please upload both images.")
            st.stop()

        progress = st.progress(0)
        logs_ui = st.empty()
        logs = []

        def log(msg):
            logs.append(f"› {msg}")
            logs_ui.markdown(f'<div style="background:#1a1916;color:#a8e6cf;font-family:monospace;padding:12px;border-radius:8px;white-space:pre-wrap;">{"<br>".join(logs)}</div>', unsafe_allow_html=True)

        try:
            log("Saving uploaded images...")
            plan_path = save_uploaded_file(planogram_file, "planogram.jpg")
            actual_path = save_uploaded_file(actual_file, "actual.jpg")

          

            progress.progress(20)

            log("Step 1/3: Extracting Planogram...")
            planogram_data = extract_planogram(plan_path)
            progress.progress(45)

            log("Step 2/3: Extracting Actual Shelf...")
            actual_data = extract_actual(actual_path) #same
            progress.progress(70)

            log("Step 3/3: Comparing Planogram vs Actual...")
            data = compare_planogram_vs_actual(planogram_data, actual_data)
            progress.progress(95)

            log("✅ Analysis completed successfully!")
            # st.success("✅ Compliance Analysis Complete!")
            progress.progress(100)

            st.session_state["result"] = data
            time.sleep(0.8)

        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            log(f"ERROR: {str(e)}")
            st.stop()
        finally:
            time.sleep(1)
            progress.empty()




    # ==============================
    # RESULTS DISPLAY (MERGED)
    # ==============================

    data = st.session_state.get("result", None)

    # st.markdown("---")
    # st.title("🛒 Planogram Compliance Dashboard")

    if data:

        import pandas as pd
        import plotly.express as px

        st.markdown("### Shelf-wise Product Compliance Analysis")

        # ====================== OVERALL ======================
        st.header("Overall Compliance")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Matching Products", data["overall_compliance"]["matching_percentage"])

        with col2:
            st.metric("Non-Matching Products", data["overall_compliance"]["non_matching_percentage"])

        with col3:
            st.metric("Confidence Score", f"{data['overall_compliance']['confidence']}%")

        st.progress(int(data["overall_compliance"]["matching_percentage"].replace("%", "")) / 100)

        # ====================== SHELF DATA ======================
        shelf_data = []
        for shelf in data["shelves"]:
            shelf_data.append({
                "Shelf": shelf["level"],
                "Matching %": int(shelf["matching_percentage"].replace("%", "")),
                "Non-Matching %": int(shelf["non_matching_percentage"].replace("%", "")),
                "Confidence": shelf["confidence"],
                "Matches": len(shelf["matches"]),
                "Mismatches": len(shelf["mismatches"])
            })

        df = pd.DataFrame(shelf_data)

        # ====================== SHELF METRICS ======================
        st.header("Shelf-wise Compliance")

        cols = st.columns(len(df))

        for i, row in df.iterrows():
            with cols[i]:
                st.subheader(row["Shelf"])
                st.metric("Compliance", f"{row['Matching %']}%", delta=f"{row['Matches']} matched")
                st.progress(row['Matching %'] / 100)

        # ====================== BAR CHART ======================
        st.subheader("Compliance Rate by Shelf")

        fig = px.bar(
            df,
            x="Shelf",
            y="Matching %",
            text="Matching %",
            color="Matching %",
            color_continuous_scale="RdYlGn"
        )
        fig.update_traces(texttemplate='%{text}%', textposition='outside')

        st.plotly_chart(fig, use_container_width=True)

        # ====================== DETAILS ======================
        st.header("Detailed Shelf Analysis")

        tab1, tab2, tab3 = st.tabs(["📋 All Shelves", "✅ Matches", "❌ Mismatches"])

        with tab1:
            for shelf in data["shelves"]:
                with st.expander(f"{shelf['level']} — {shelf['matching_percentage']}"):
                    col_a, col_b = st.columns(2)

                    with col_a:
                        st.write("**Planogram:**")
                        for item in shelf["planogram"]:
                            st.write(f"• {item}")

                    with col_b:
                        st.write("**Actual:**")
                        for item in shelf["actual"]:
                            st.write(f"• {item}")

                    if shelf["matches"]:
                        st.success("Matches:")
                        for m in shelf["matches"]:
                            st.write(f"✓ {m}")

                    if shelf["mismatches"]:
                        st.error("Mismatches:")
                        for mis in shelf["mismatches"]:
                            st.write(f"✗ {mis}")

        with tab2:
            for shelf in data["shelves"]:
                for match in shelf["matches"]:
                    st.success(f"{shelf['level']}: {match}")

        with tab3:
            for shelf in data["shelves"]:
                for mismatch in shelf["mismatches"]:
                    st.error(f"{shelf['level']}: {mismatch}")

        # ====================== SUMMARY ======================
        
        st.text("\n")
        st.text("📝 Executive Summary")
        st.info(data["summary"])



    # ==============================
    # DEBUG / RAW JSON VIEW
    # ==============================

    # st.markdown("---")
    # st.header("🔍 Raw JSON Outputs (All 3 Steps)")

    tab1, tab2, tab3 = st.tabs([
        "📋 Planogram Extracted",
        "📷 Actual Extracted",
        "🔍 Final Comparison"
    ])

    # -------- PLANOGRAM --------
    with tab1:
        try:
            if "unique_no" in st.session_state:
                unique_no = st.session_state["unique_no"]
                plan_file = f"outputs/planogram_extracted_{unique_no}.json"
            else:
                plan_file = "planogram_extracted.json"  # fallback

            if os.path.exists(plan_file):
                with open(plan_file, "r", encoding="utf-8") as f:
                    plan_data = json.load(f)

                st.subheader("Planogram JSON")
                st.json(plan_data)

                st.download_button(
                    "⬇️ Download Planogram JSON",
                    data=json.dumps(plan_data, indent=2),
                    file_name=os.path.basename(plan_file)
                )
            else:
                st.warning("Planogram JSON not available")
        except Exception as e:
            st.error(f"Error loading Planogram JSON: {e}")

    # -------- ACTUAL --------
    with tab2:
        try:
            if "unique_no" in st.session_state:
                unique_no = st.session_state["unique_no"]
                actual_file = f"outputs/actual_extracted_{unique_no}.json"
            else:
                actual_file = "actual_extracted.json"  # fallback

            if os.path.exists(actual_file):
                with open(actual_file, "r", encoding="utf-8") as f:
                    actual_data = json.load(f)

                st.subheader("Actual Shelf JSON")
                st.json(actual_data)

                st.download_button(
                    "⬇️ Download Actual JSON",
                    data=json.dumps(actual_data, indent=2),
                    file_name=os.path.basename(actual_file)
                )
            else:
                st.warning("Actual JSON not available")
        except Exception as e:
            st.error(f"Error loading Actual JSON: {e}")

    # -------- FINAL --------
    with tab3:
        try:
            final_data = st.session_state.get("result")

            st.subheader("Comparison JSON")
            st.json(final_data)

            st.download_button(
                "⬇️ Download Final JSON",
                data=json.dumps(final_data, indent=2),
                file_name=f"planogram_vs_actual_comparison_{st.session_state.get('unique_no','latest')}.json"
            )
        except Exception as e:
            st.warning("Final JSON not available")
            try:
                final_data = st.session_state.get("result")

                st.subheader("Comparison JSON")
                st.json(final_data)

                st.download_button(
                    "⬇️ Download Final JSON",
                    data=json.dumps(final_data, indent=2),
                    file_name="planogram_vs_actual_comparison.json"
                )
            except:
                st.warning("Final JSON not available")    
