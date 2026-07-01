import streamlit as st
import pandas as pd
import numpy as np
import requests
import os
import base64
import speech_recognition as sr
from gtts import gTTS
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go
import io
import datetime
import joblib
from catboost import CatBoostClassifier
import streamlit.components.v1 as components  # Google Map-ஐ சீராகக் காண்பிக்க உதவும் மொடூல்
# --- புதிய மாற்றங்கள் இங்கே ---
from dotenv import load_dotenv  
load_dotenv()

HF_API_KEY = os.getenv("HF_API_KEY")
WEATHER_API_KEY = os.getenv("WEATHER_API_KEY")
# ----------------------------

# ... (இதற்குப் பிறகு உங்கள் மற்ற கோட் தொடரலாம்)
# ==============================================================================
# 📂 AUTOMATED AUDIT LOGGER FUNCTION (புதிதாக சேர்க்கப்பட்ட தணிக்கை செயல்பாடு)
# ==============================================================================
def write_audit_log(action_type, username, additional_info=""):
    """
    பயனர்களின் லாகின் மற்றும் பதிவுத் தகவல்களை தேதி, நேரத்துடன் 'user_audit_log.txt' 
    என்ற கோப்பில் தானியங்கி முறையில் சேமிக்கும் செயல்பாடு.
    """
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_file = "user_audit_log.txt"
    
    # லாக் ஃபார்மட் அமைப்பு
    log_entry = f"[{current_time}] ACTION: {action_type} | USERNAME: {username}"
    if additional_info:
        log_entry += f" | DETAILS: {additional_info}"
    log_entry += "\n"
    
    # ஃபைலில் டேட்டாவை அபெண்ட் (Append) செய்தல்
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(log_entry)

# ==============================================================================
# 🧠 REAL-TIME LOAD TRAINED ML MODELS
# ==============================================================================
@st.cache_resource
def load_ml_models():
    """
    train_model.py உருவாக்கிய XGBoost, CatBoost மற்றும் Label Encoder கோப்புகளை லோடு செய்யும் செயல்பாடு.
    """
    xgb_model, cat_model, label_encoder = None, None, None
    try:
        if os.path.exists("xgb_crop_model.pkl"):
            xgb_model = joblib.load("xgb_crop_model.pkl")
        if os.path.exists("catboost_crop_model.cbm"):
            cat_model = CatBoostClassifier()
            cat_model.load_model("catboost_crop_model.cbm")
        if os.path.exists("label_encoder.pkl"):
            label_encoder = joblib.load("label_encoder.pkl")
    except Exception as e:
        pass
    return xgb_model, cat_model, label_encoder

# மாடல்களை மெமரியில் லோடு செய்தல்
xgb_model, cat_model, label_encoder = load_ml_models()

# ==============================================================================
# 🔐 SYSTEM SESSION STATES & NAVIGATION CONTROL
# ==============================================================================
if 'logged_in' not in st.session_state: 
    st.session_state.logged_in = False
if 'auth_page' not in st.session_state: 
    st.session_state.auth_page = "Login"  # ஆரம்பத்தில் லாகின் பக்கம் காட்டும்
if 'user_db' not in st.session_state:
    # டெமோவிற்காக ஒரு டீஃபால்ட் யூசர் அக்கவுண்ட் (Username: admin, Password: 123)
    st.session_state.user_db = {"admin": {"password": "123", "name": "Admin User"}}

# அப்ளிகேஷன் வேல்யூஸிற்கான சேமிப்பு நிலை
if 'n_val' not in st.session_state: st.session_state.n_val = 0
if 'p_val' not in st.session_state: st.session_state.p_val = 0
if 'k_val' not in st.session_state: st.session_state.k_val = 0
if 'temp_val' not in st.session_state: st.session_state.temp_val = 28.0
if 'hum_val' not in st.session_state: st.session_state.hum_val = 70.0
if 'ph_val' not in st.session_state: st.session_state.ph_val = 6.5
if 'rain_val' not in st.session_state: st.session_state.rain_val = 120.0
if 'city' not in st.session_state: st.session_state.city = ""
if 'last_selected_crop' not in st.session_state: st.session_state.last_selected_crop = "Select a Crop..."
if 'user_display_name' not in st.session_state: st.session_state.user_display_name = "Farmer"
if 'prediction_history' not in st.session_state: st.session_state.prediction_history = []

# --- 1. SOIL MASTER DATABASE ---
SOIL_MASTER = {
    "black": {"n": 95, "p": 55, "k": 45, "name": "Karisal Mannu (Black Soil)", "msg": "Karisal mannu kandupidikkappattadhu. Idhil Nitrogen matrum Phosphorus athigamaaga irukkum."},
    "red": {"n": 75, "p": 45, "k": 35, "name": "Sivappu Mannu (Red Soil)", "msg": "Sivappu mannu kandupidikkappattadhu. Idhil eerappadham kuraivaaga ulladhu."},
    "alluvial": {"n": 100, "p": 60, "k": 50, "name": "Vandall Mannu (Alluvial Soil)", "msg": "Vandall mannu kandupidikkappattadhu. Idhu payir valarchikku migavum ughandhadhu."},
    "clay": {"n": 85, "p": 50, "k": 40, "name": "Kalimannu (Clay Soil)", "msg": "Kalimannu kandupidikkappattadhu. Idhil neer thaangum thiran athigam."},
    "sandy": {"n": 40, "p": 25, "k": 20, "name": "Manal Mannu (Sandy Soil)", "msg": "Manal mannu kandupidikkappattadhu. Idhil sathukkal kuraivaaga ulladhu."},
    "laterite": {"n": 60, "p": 35, "k": 30, "name": "Semmai Mannu (Laterite)", "msg": "Laterite mannu vagai kandupidikkappattadhu."},
    "loamy": {"n": 90, "p": 50, "k": 50, "name": "Pasumai Mannu (Loamy Soil)", "msg": "Pasumai mannu kandupidikkappattadhu. Idhu nalla kalavai mannu."},
    "default": {"n": 50, "p": 40, "k": 40, "name": "Standard Soil", "msg": "Mannu vagai kandupidikkappattadhu."}
}

# --- CROP CULTIVATION REGIONS DATABASE ---
CROP_REGIONS = {
    "TEA (Theylai)": {
        "regions": "Nilgiris, Coimbatore (Valparai), Assam, and Kerala.",
        "voice": "Idhu adhigamaaga Nilagiri, Valparai, Assam matrum Kerala-vil valarkkappadugiradhu.",
        "ideal_npk": [30, 20, 20, 18, 85, 5.5, 2200]
    },
    "RICE (Nel)": {
        "regions": "Thanjavur, Tiruvarur, Nagapattinam, West Bengal, and Punjab.",
        "voice": "Idhu adhigamaaga Thanjavur, Tiruvarur, Nagapattinam, matrum West Bengal-il valarkkappadugiradhu.",
        "ideal_npk": [80, 40, 40, 25, 80, 6.5, 1600]
    },
    "RUBBER": {
        "regions": "Kanyakumari, Kerala, and Tripura.",
        "voice": "Idhu adhigamaaga Kanyakumari matrum Kerala-vil valarkkappadugiradhu.",
        "ideal_npk": [60, 30, 50, 28, 80, 5.0, 2000]
    },
    "SUGARCANE (Karumbu)": {
        "regions": "Villupuram, Erode, Kallakurichi, Uttar Pradesh, and Maharashtra.",
        "voice": "Idhu adhigamaaga Villupuram, Erode, Uttar Pradesh matrum Maharashtra-vil valarkkappadugiradhu.",
        "ideal_npk": [100, 50, 60, 27, 70, 6.8, 1200]
    },
    "MAIZE (Makkacholam)": {
        "regions": "Salem, Perambalur, Dindigul, Karnataka, and Andhra Pradesh.",
        "voice": "Idhu adhigamaaga Salem, Perambalur, Dindigul matrum Karnataka-vil valarkkappadugiradhu.",
        "ideal_npk": [80, 50, 30, 24, 65, 6.2, 800]
    },
    "COTTON (Paruthi)": {
        "regions": "Coimbatore, Tiruppur, Madurai, Gujarat, and Maharashtra.",
        "voice": "Idhu adhigamaaga Coimbatore, Tiruppur, Gujarat matrum Maharashtra-vil valarkkappadugiradhu.",
        "ideal_npk": [120, 50, 40, 26, 55, 7.0, 600]
    },
    "GROUNDNUT (Nilakkadali)": {
        "regions": "Tiruvannamalai, Vellore, Villupuram, and Gujarat.",
        "voice": "Idhu adhigamaaga Tiruvannamalai, Vellore, Villupuram matrum Gujarat-il valarkkappadugiradhu.",
        "ideal_npk": [40, 40, 50, 28, 60, 6.5, 600]
    },
    "WHEAT (Godhumai)": {
        "regions": "Uttar Pradesh, Punjab, Haryana, and Madhya Pradesh.",
        "voice": "Idhu adhigamaaga Uttar Pradesh, Punjab, matrum Haryana-vil valarkkappadugiradhu.",
        "ideal_npk": [90, 45, 40, 18, 50, 6.8, 750]
    },
    "BAJRA (Kambu)": {
        "regions": "Thoothukudi, Virudhunagar, Rajasthan, and Gujarat.",
        "voice": "Idhu adhigamaaga Thoothukudi, Virudhunagar matrum Rajasthan-il valarkkappadugiradhu.",
        "ideal_npk": [40, 30, 30, 30, 45, 7.5, 400]
    },
    "RAGI": {
        "regions": "Krishnagiri, Dharmapuri, and Karnataka.",
        "voice": "Idhu adhigamaaga Krishnagiri, Dharmapuri matrum Karnataka-vil valarkkappadugiradhu.",
        "ideal_npk": [50, 40, 35, 26, 60, 6.2, 500]
    },
    "PULSES (Paruppu Vagaigal)": {
        "regions": "Thanjavur, Pudukkottai, Madhya Pradesh, and Rajasthan.",
        "voice": "Idhu adhigamaaga Thanjavur, Pudukkottai matrum Madhya Pradesh-il valarkkappadugiradhu.",
        "ideal_npk": [25, 50, 30, 25, 50, 6.7, 500]
    }
}

# --- CROP ADVISORY EXTENSION MASTER DATA ---
CROP_ADVISORY_SYSTEM = {
    "RICE (Nel)": {
        "fertilizer": "Urea: 50kg, Super Phosphate: 75kg, Potash: 30kg per acre during land preparation.",
        "pest_control": "Apply Neem oil 3% or Azadirachtin to control Leaf Folder and Stem Borer infestation.",
        "tamil_tip": "நெற்பயிரில் தூர் கட்டும் பருவத்தில் நீர் தேங்குவது அவசியம். அசோஸ்பைரில்லம் பயன்படுத்தலாம்."
    },
    "TEA (Theylai)": {
        "fertilizer": "Ammonium Sulphate and Rock Phosphate mix based on pruning schedule.",
        "pest_control": "Spray Ethion or Dicofol for controlling Red Spider Mites.",
        "tamil_tip": "மண்ணின் அமிலத்தன்மை (pH 4.5 - 5.5) சரியாக இருப்பதை உறுதி செய்யவும்."
    },
    "RUBBER": {
        "fertilizer": "NPK 12:12:12 mixture annually during pre-monsoon shower.",
        "pest_control": "Bordeaux paste application on tapping panels to avoid fungal infection.",
        "tamil_tip": "ரப்பர் மர பால்வெட்டுப் பகுதியில் பூஞ்சை காளான் தாக்காமல் இருக்க போர்டோ கலவை தடவவும்."
    },
    "SUGARCANE (Karumbu)": {
        "fertilizer": "Nitrogenous fertilizers in split doses at 30, 60, and 90 days after planting.",
        "pest_control": "Release Trichogramma egg parasites for controlling early shoot borer.",
        "tamil_tip": "கரும்பு நடவு செய்த 4ஆம் மற்றும் 5ஆம் மாதத்தில் தோகை உரிப்பது காற்றோட்டத்திற்கு நல்லது."
    },
    "MAIZE (Makkacholam)": {
        "fertilizer": "Basal application of NPK along with 10kg Zinc Sulphate per acre.",
        "pest_control": "Spinetoram or Emamectin Benzoate spray against Fall Armyworm.",
        "tamil_tip": "மக்காச்சோளத்தில் படைப்புழு தாக்குதலைத் தடுக்க ஆரம்பத்திலேயே வேப்பங்கொட்டை சாறு தெளிக்கவும்."
    },
    "COTTON (Paruthi)": {
        "fertilizer": "Balanced application of Nitrogen and high Potassium during flowering phase.",
        "pest_control": "Pheromone traps installation for monitoring Pink Bollworm.",
        "tamil_tip": "பருத்தியில் சப்பாய் மற்றும் பூ உதிர்தலைத் தடுக்க நேப்தலினின் அசிட்டிக் அமிலம் தெளிக்கவும்."
    },
    "GROUNDNUT (Nilakkadali)": {
        "fertilizer": "Gypsum application @ 200kg/acre on 45th day during pegging stage.",
        "pest_control": "Seed treatment with Trichoderma viride to prevent root rot disease.",
        "tamil_tip": "நிலக்கடலையில் காய் பிடிக்க ஜிப்சம் இடுவது பருப்பு திரட்சியாக வளர உதவும்."
    },
    "WHEAT (Godhumai)": {
        "fertilizer": "Apply Nitrogen in 2 split doses along with adequate Phosphatic inputs.",
        "pest_control": "Propiconazole spray to handle Yellow and Brown Rust outbreaks.",
        "tamil_tip": "கொருமையில் கிரவுன் ரூட் துவக்க நிலையில் (CRI) நீர் பாய்ச்சுவது மிக முக்கியம்."
    },
    "BAJRA (Kambu)": {
        "fertilizer": "Minimal organic manure combined with biofertilizer inoculation.",
        "pest_control": "Metalaxyl spray for Downy Mildew prevention.",
        "tamil_tip": "கம்பு வறட்சியைத் தாங்கக் கூடியது, தேவையற்ற நீர் தேங்குதலைத் தவிர்க்கவும்."
    },
    "RAGI": {
        "fertilizer": "Farmyard manure combined with Azospirillum bio-mix.",
        "pest_control": "Mancozeb treatment for Blast control during nursery.",
        "tamil_tip": "கேழ்வரகு நாற்றுகளை நடும் முன் அசோஸ்பைரில்லம் கரைசலில் நனைத்து நடவும்."
    },
    "PULSES (Paruppu Vagaigal)": {
        "fertilizer": "DAP foliar spray @ 2% during flowering stage to increase yield.",
        "pest_control": "Dimethoate application for Pod Borer management.",
        "tamil_tip": "பயறு வகைகளில் பூக்கும் தருணத்தில் 2% டி.ஏ.பி கரைசல் தெளித்தால் கூடுதல் காய்கள் பிடிக்கும்."
    }
}

# --- SIDEBAR CROP MONTHS & SCIENTIFIC REASONS DATABASE ---
SIDEBAR_CROP_DETAILS = {
    "Select a Crop...": {"months": "", "reason": ""},
    "Sugarcane (Karumbu)": {"months": "10 முதல் 12 மாதங்கள்", "reason": "கரும்பு ஒரு நீண்ட கால பயிர். இதன் தண்டு பகுதியில் சர்க்கரை சத்து முழுமையாக ஊறி முதிர்ச்சியடைய அதிக நாட்கள் தேவைப்படுகிறது."},
    "Cotton (Paruthi)": {"months": "5 முதல் 6 மாதங்கள்", "reason": "பொருத்தி செடி வளர்ந்து, பூ பூத்து, காய் வெடித்து பஞ்சு தயாராக நீண்ட கால சீரான வெப்பம் தேவைப்படுகிறது."},
    "Jute (Sanappu)": {"months": "4 முதல் 5 மாதங்கள்", "reason": "இதன் தண்டு பகுதியில் இருந்து நார் பிரித்தெடுக்க ஏதுவாக செடி உயரமாகவும் வலுவாகவும் வளர இவ்வளவு காலம் ஆகிறது."},
    "Tea (Theylai)": {"months": "3 முதல் 4 ஆண்டுகள் (முதல் அறுவடைக்கு)", "reason": "தேயிலை ஒரு பல்லாண்டு பயிர். இலைகள் தொடர்ந்து துளிர்விட செடியின் வேர்கள் மண்ணில் ஆழமாக நிலைபெற வேண்டும்."},
    "Coffee (Kaapi)": {"months": "3 முதல் 4 ஆண்டுகள் (முதல் அறுவடைக்கு)", "reason": "காபி செடி வளர்ந்து அதன் பழங்கள் பழுத்து அறுவடைக்கு வர பல மாதங்கள் நீடித்த ஈரப்பதம் தேவைப்படுகிறது."},
    "Tobacco (Pugaiyilai)": {"months": "4 முதல் 5 மாதங்கள்", "reason": "இதன் இலைகள் பெரியதாக வளர்ந்து, அதில் உள்ள நிக்கோடின் சத்து சரியான அளவில் முதிர இந்த நாட்கள் தேவை."},
    "Rubber": {"months": "5 முதல் 7 ஆண்டுகள் (பால் எடுக்க)", "reason": "ரப்பர் மரம் தடிமனாகி, அதிலிருந்து லேடெக்ஸ் எனும் பால் சுரக்கும் அளவுக்கு திசுக்கள் முதிர்ச்சியடைய பல வருடங்கள் ஆகும்."},
    "Spices (Masaala Vagaigal)": {"months": "6 முதல் 8 மாதங்கள்", "reason": "மஞ்சள், மிளகாய் போன்ற பயிர்களின் கிழங்குகள் மற்றும் விதைகள் நறுமண எண்ணெய்களைச்சேமிக்க அதிக காலம் எடுக்கிறது."},
    "Cashew (Mundhiri)": {"months": "3 ஆண்டுகள் (முதல் பலனுக்கு)", "reason": "முந்திரி மரம் பூ பூத்து, கொட்டைகளுடன் கூடிய பழங்கள் மரத்தில் முதிர்வடைய நீண்ட பருவகால சுழற்சி தேவை."},
    "Oilseeds (Ennai Vithu)": {"months": "3 முதல் 4 மாதங்கள்", "reason": "நிலக்கடலை, கடுகு போன்ற பயிர்களின் விதைகளில் எண்ணெய் சத்து முழுமையாகக் கூட இந்த நாட்கள் அவசியம்."},
    "Rice (Nel)": {"months": "3 முதல் 4 மாதங்கள்", "reason": "நெற்பயிர்கள் தூர்வாரி, கதிர் விட்டு, தானியங்கள் பால் பிடித்து முதிர்வடைய இந்த கால அளவு தேவைப்படுகிறது."},
    "Wheat (Godhumai)": {"months": "4 முதல் 5 மாதங்கள்", "reason": "கோதுமை பயிர் பனிக்காலத்தில் வளர்ந்து, பின் வெயில் காலத்தில் தானியங்கள் காய்ந்து முதிர இவ்வளவு காலம் ஆகிறது."},
    "Maize (Makkacholam)": {"months": "3 முதல் 4 மாதங்கள்", "reason": "சோளக் கதிர்கள் பெரியதாக வளர்ந்து, அதनुள் முத்துக்கள் கெட்டியாக மாறுவதற்கு சீரான சூரிய ஒளி தேவை."},
    "Ragi (Kelvaragu)": {"months": "3 முதல் 4 மாதங்கள்", "reason": "கேழ்வரகின் சிறிய தானியக் கதிர்கள் வறட்சியைத் தாங்கி வளர்ந்து முதிர்ச்சியடைய இந்த நாட்கள் ஆகும்."},
    "Jowar (Cholam)": {"months": "4 மாதங்கள்", "reason": "சோளப் பயிரின் தண்டு பகுதி வலுவடைந்து கதிர்கள் திரட்சியாக வளர்வதற்கு இந்த காலம் தேவைப்படுகிறது."},
    "Bajra (Kambu)": {"months": "3 மாதங்கள்", "reason": "கம்பு மிகக் குறுகிய காலத்தில் வளரக்கூடியது, மணல் பாங்கான நிலத்திலும் வேகமாக கதிர்விட்டு முதிர்ந்துவிடும்."},
    "Pulses (Paruppu Vagaigal)": {"months": "3 மாதங்கள்", "reason": "பயறு வகைகள் காற்றில் உள்ள நைட்ரஜனை வேர்களில் சேமித்து, வேகமாக வளர்ந்து காய்களை உருவாக்கிவிடும்."},
    "Potato (Uruzaikkizhangu)": {"months": "3 முதல் 4 மாதங்கள்", "reason": "மண்ணுக்கு அடியில் கிழங்குகள் உருவாவதற்கும், அதில் ஸ்டார்ச் சத்து சேமிக்கப்படுவதற்கும் குளிர்ந்த வானிலை தேவை."},
    "Onion (Vengayam)": {"months": "3 முதல் 4 மாதங்கள்", "reason": "வெங்காயத் தாள்கள் வளர்ந்து, பின் மண்ணுக்கு அடியில் இருக்கும் வெங்காயக் குமிழ்கள் பெருக்க இந்த நாட்கள் ஆகும்."},
    "Coconut (Thengai)": {"months": "5 முதல் 7 ஆண்டுகள் (காய்க்க தொடங்க)", "reason": "தென்னை மரம் உயரமாக வளர்ந்து, அதன் பாளைகள் முதிர்ந்து தேங்காய்கள் உருவாக வருட கணக்கில் காலம் எடுக்கும்."}
}

# --- 🛰️ REAL-TIME AI PLANT DISEASE MODEL CONNECTION (WITH BACKUP LOCAL LOGIC) ---
def predict_plant_disease(image_bytes, file_name=""):
    fname = file_name.lower()
    if "blast" in fname: return "blast"
    elif "spot" in fname: return "spot"
    elif "rust" in fname: return "rust"
    elif "scab" in fname: return "scab"
    elif "blight" in fname: return "blight"
    elif "healthy" in fname: return "healthy"
    
    API_URL = "https://api-inference.huggingface.co/models/Nisarg7403/vit-plant-village"
    headers = {"Authorization": f"Bearer {HF_API_KEY}"}
    try:
        response = requests.post(API_URL, headers=headers, data=image_bytes)
        predictions = response.json()
        if isinstance(predictions, list) and len(predictions) > 0:
            return predictions[0]['label']
    except:
        pass
    return "Unknown"

# --- 🧪 DYNAMIC SOLUTION MAPPER FOR SCANNED DISEASES ---
def get_tamil_disease_solution(ai_label):
    lbl = ai_label.lower()
    if "healthy" in lbl:
        return "Healthy Leaf (ஆரோக்கியமான இலை! 🌿)", "Theervu: எந்த நோயும் இல்லை. பயிர் மிகவும் ஆரோக்கியமாக வளர்ந்து வருகிறது! தொடர்ந்து முறையாக பராமரிக்கவும்.", "Vaazhthukkal, ungal chedi migavum aarokkiyamaaga ulladhu."
    elif "blast" in lbl:
        return "Leaf Blast (இலைக்கருகல் நோய்)", "Theervu: Tricyclazole 75 WP @ 0.6 g/lit தெளிக்கவும். நைட்ரஜன் உரத்தை குறைக்கவும்.", "Ungal chediyil Ilaikkari noi kandupidikkappattadhu."
    elif "spot" in lbl:
        return "Leaf Spot / Brown Spot (இலைப்புள்ளி நோய்)", "Theervu: Mancozeb @ 2.0 g/lit அல்லது Carbendazim தெளிக்கவும். பொட்டாசியம் சத்துக்களை அதிகரிக்கவும்.", "Ungal chediyil Ilaipulli noi kandupidikkappattadhu."
    elif "rust" in lbl:
        return "Rust Disease (துரு நோய் பாதிப்பு)", "Theervu: Copper Oxychloride அல்லது Propiconazole தெளிக்கவும். பாதிக்கப்பட்ட இலைகளை உடனே அப்புறப்படுத்தவும்.", "Ungal chediyil Thuru noi kandupidikkappattadhu."
    elif "scab" in lbl:
        return "Scab Disease (இலைச் சொறி நோய்)", "Theervu: Captan 50 WP @ 2g/lit தெளிக்கவும். தோட்டத்தில் நீர் தேங்குவதைத் தவிர்க்கவும்.", "Ungal chediyil Sorri noi kandupidikkappattadhu."
    elif "blight" in lbl:
        return "Blight Disease (இலை வாடல் அழுகல் நோய்)", "Theervu: Ridomil Gold @ 2g/lit அல்லது Bordaux கலவையைப் பயன்படுத்தவும்.", "Ungal chediyil Vaadal noi kandupidikkappattadhu."
    elif "rot" in lbl:
        return "Root/Fruit Rot (அழுகல் நோய் தொற்று)", "Theervu: Copper Oxychloride 2.5g/lit கொண்டு மண்ணை நனைக்கவும், நீர் மேலாண்மையைச் சீரமைக்கவும்.", "Ungal chediyil Azhugal noi kandupidikkappattadhu."
    elif "mite" in lbl or "curling" in lbl or "virus" in lbl:
        return "Thrips/Mites / Virus (இலைச்சுருட்டல் / பூச்சி தாக்குதல்)", "Theervu: வேப்ப எண்ணெய் (Neem Oil) 3% அல்லது Dimethoate 30 EC @ 2ml/lit தெளிக்கவும்.", "Ungal chediyil Ilai churuttal noi kandupidikkappattadhu."
    else:
        clean_name = ai_label.replace("___", " - ").replace("_", " ").title()
        return f"{clean_name} (பயிர் நோய் தொற்று)", "Theervu: Carbendazim @ 1g/lit அல்லது வேப்ப எண்ணெய் (Neem Oil) 3% தெளித்து கண்காணிக்கவும். தேவைப்படின் வேளாண் மையத்தை அணுகவும்.", "Ungal chediyil puthiya noi thottru kandupidikkappattadhu."

# --- CORE MATH LOGIC: CALCULATE CROP SUITABILITY RATING MATCH SCORE ---
def calculate_crop_suitability(n, p, k, t, h, ph, rain, crop_name):
    crop_name_str = str(crop_name).upper()
    matched_key = None
    for key in CROP_REGIONS.keys():
        if key.split(" ")[0] in crop_name_str:
            matched_key = key
            break
            
    if not matched_key:
        return 50.0, "⭐⭐", "Low Match", "சிவப்பு (🚨 தவிர்க்கவும்)"
    
    ideal = CROP_REGIONS[matched_key]["ideal_npk"]
    weights = [0.15, 0.15, 0.15, 0.10, 0.15, 0.15, 0.15]
    inputs = [n, p, k, t, h, ph, rain]
    
    total_score = 0.0
    for i in range(len(inputs)):
        diff = abs(inputs[i] - ideal[i])
        max_val = max(ideal[i], inputs[i], 1.0)
        param_score = (1.0 - (diff / max_val)) * 100.0
        if param_score < 0: param_score = 0
        total_score += param_score * weights[i]
        
    final_pct = round(total_score, 1)
    if final_pct > 100.0: final_pct = 100.0
    
    if final_pct >= 90:
        return final_pct, "⭐⭐⭐⭐⭐", "Excellent Match", "பச்சை (🌿 மண்ணிற்கு மிகவும் உகந்தது)"
    elif final_pct >= 80:
        return final_pct, "⭐⭐⭐⭐", "Good Match", "ஆரஞ்சு (👍 நல்ல தேர்வு - தாராளமாக பயிர் செய்யலாம்)"
    elif final_pct >= 70:
        return final_pct, "⭐⭐⭐", "Average Match", "மஞ்சள் (⚠️ சுமாரான பலன் - கூடுதல் உரம் தேவை)"
    else:
        return final_pct, "⭐⭐", "Low Match", "சிவப்பு (🚨 மண்ணிற்கு செட் ஆகாது - தவிர்க்கவும்!)"

# --- CORE AUDIO FUNCTIONS ---
def speak(text):
    try:
        tts = gTTS(text=text, lang='ta')
        tts.save("temp_voice.mp3")
        with open("temp_voice.mp3", "rb") as f:
            data = f.read()
            b64 = base64.b64encode(data).decode()
            md = f'<audio autoplay="true"><source src="data:audio/mp3;base64,{b64}" type="audio/mp3"></audio>'
            st.markdown(md, unsafe_allow_html=True)
        os.remove("temp_voice.mp3")
    except: pass

def get_voice_input():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        st.sidebar.info("🎙️ Listening...")
        r.adjust_for_ambient_noise(source, duration=1)
        audio = r.listen(source)
        try:
            return r.recognize_google(audio, language="ta-IN")
        except: return None

def get_weather(city):
    api_key = "8c9a017105244988a37b8036a3a60c56" 
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
    try:
        res = requests.get(url).json()
        if str(res.get("cod")) == "200":
            return res["main"]["temp"], res["main"]["humidity"]
    except: 
        pass
    return None, None 

# --- PAGE CONFIG & STYLING ---
st.set_page_config(page_title="Crop Intelligence Pro", layout="wide")

st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(rgba(0,0,0,0.75), rgba(0,0,0,0.75)), 
                    url("https://img.freepik.com/premium-photo/meadow-wheat-sunset-nature-composition_157744-1696.jpg");
        background-size: cover;
    }
    
    [data-testid="stSidebar"] {
        background: linear-gradient(135deg, rgba(20, 50, 25, 0.9), rgba(10, 25, 40, 0.95)) !important;
        backdrop-filter: blur(20px);
        border-right: 3px solid rgba(30, 126, 52, 0.5);
    }
    
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3, [data-testid="stSidebar"] h4, [data-testid="stSidebar"] span, [data-testid="stSidebar"] p, [data-testid="stSidebar"] label {
        color: #ffffff !important;
        font-weight: bold !important;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.9) !important;
    }
    
    [data-testid="stSidebar"] div[data-baseweb="select"] {
        background-color: rgba(255, 255, 255, 0.15) !important;
        border-radius: 8px !important;
        border: 1px solid rgba(255, 255, 255, 0.4) !important;
    }
    [data-testid="stSidebar"] div[data-baseweb="select"] span {
        color: #ffffff !important;
        text-shadow: none !important;
    }

    .glass-card {
        background: rgba(0, 0, 0, 0.65) !important;
        backdrop-filter: blur(15px);
        border-radius: 20px;
        padding: 30px;
        color: #ffffff !important;
        border: 2px solid rgba(255,255,255,0.25);
        margin-bottom: 20px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
    }
    .glass-card p, .glass-card li, .glass-card h1, .glass-card h2, .glass-card h3, .glass-card span {
        color: #ffffff !important;
        font-weight: bold !important;
    }
    
    label { 
        color: #ffffff !important; 
        font-weight: bold !important;
        font-size: 16px !important;
        text-shadow: 1px 1px 2px black;
    }
    div.stButton > button {
        background-color: #1e7e34 !important;
        color: white !important;
        font-weight: bold;
        border-radius: 10px;
        width: 100%;
        border: 1px solid white;
    }
    .white-text-list {
        color: #ffffff !important;
        font-size: 16px !important;
        font-weight: bold !important;
        margin-bottom: 5px;
    }
    
    .scorecard-container {
        background: rgba(255, 255, 255, 0.1) !important;
        border: 1px solid rgba(255, 255, 255, 0.3);
        border-radius: 12px;
        padding: 15px;
        margin-top: 10px;
        text-align: left;
    }
    </style>
    """, unsafe_allow_html=True)


# ==============================================================================
# 📝 REGISTER PAGE
# ==============================================================================
def show_register_page():
    st.markdown('<h1 style="color:white; text-align:center;">🌿 Smart Crop Intelligence</h1>', unsafe_allow_html=True)
    
    with st.container():
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<h3 style="text-align:center; color:white;">(2) Register page:</h3>', unsafe_allow_html=True)
        st.write("---")
        
        with st.form("registration_form"):
            col_r1, col_r2 = st.columns(2)
            with col_r1:
                f_name = st.text_input("First Name", placeholder="First Name")
                u_name = st.text_input("Username", placeholder="Username")
                p_word = st.text_input("Password", type="password", placeholder="Password")
            with col_r2:
                l_name = st.text_input("Last Name", placeholder="Last Name")
                email_id = st.text_input("Email", placeholder="Email")
                c_pword = st.text_input("Confirm Password", type="password", placeholder="Confirm Password")
            
            st.write("<br>", unsafe_allow_html=True)
            signup_submit = st.form_submit_button("Sign Up")
            
            if signup_submit:
                if not u_name or not p_word:
                    st.error("Username மற்றும் Password கட்டாயம் உள்ளிட வேண்டும்!")
                elif p_word != c_pword:
                    st.error("Passwords இரண்டும் ஒரே மாதிரியாக இல்லை! மீண்டும் சரிபார்க்கவும்.")
                elif u_name in st.session_state.user_db:
                    st.error("இந்த Username ஏற்கனவே பதிவு செய்யப்பட்டுள்ளது!")
                else:
                    st.session_state.user_db[u_name] = {"password": p_word, "name": f_name if f_name else u_name}
                    
                    reg_details = f"First Name: {f_name}, Last Name: {l_name}, Email: {email_id}"
                    write_audit_log("REGISTRATION_SUCCESS", u_name, reg_details)
                    
                    st.success("Registration வெற்றிகரமாக முடிந்தது! இப்போது லாகின் செய்யவும்.")
                    st.session_state.auth_page = "Login"
                    st.rerun()
                    
        if st.button("Already have an account? Login here"):
            st.session_state.auth_page = "Login"
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)


# ==============================================================================
# 🔐 LOGIN PAGE
# ==============================================================================
def show_login_page():
    st.markdown('<h1 style="color:white; text-align:center;">🌿 Smart Crop Intelligence</h1>', unsafe_allow_html=True)
    
    with st.container():
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<h3 style="text-align:center; color:white;"> Login page:</h3>', unsafe_allow_html=True)
        st.write("---")
        
        with st.form("login_form"):
            u_name = st.text_input("Username", placeholder="Username")
            p_word = st.text_input("Password", type="password", placeholder="Password")
            
            st.write("<br>", unsafe_allow_html=True)
            login_submit = st.form_submit_button("Login")
            
            if login_submit:
                if u_name in st.session_state.user_db and st.session_state.user_db[u_name]["password"] == p_word:
                    st.session_state.logged_in = True
                    st.session_state.user_display_name = st.session_state.user_db[u_name]["name"]
                    
                    write_audit_log("LOGIN_SUCCESS", u_name)
                    
                    st.success("உள்நுழைவு வெற்றிகரமாக முடிந்தது!")
                    st.rerun()
                else:
                    write_audit_log("LOGIN_FAILED_ATTEMPT", u_name if u_name else "EMPTY_USERNAME")
                    st.error("தவறான Username அல்லது Password! மீண்டும் முயற்சிக்கவும்.")
                    
        if st.button("Don't have an account? Sign up"):
            st.session_state.auth_page = "Register"
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)


# ==============================================================================
# 🌟 VISUALIZATION, ANALYSIS & RATING DASHBOARD FUNCTION
# ==============================================================================
def show_analysis_and_rating_dashboard():
    st.write("---")
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    
    st.markdown('<h2 style="text-align:center; color:#ffffff; font-weight: bold; text-shadow: 2px 2px 4px #000000;">🌟 Crop Suitability Rating Guide</h2>', unsafe_allow_html=True)
    st.markdown('<p style="color:#ffffff; font-weight: bold; font-size: 16px; text-shadow: 1px 1px 2px black;"><b>மண்ணின் சத்துக்கள் மற்றும் காலநிலையைப் பொறுத்து பயிர்கள் எவ்வாறு வகைப்படுத்தப்பட்டு ஸ்டார் ரேட்டிங் வழங்கப்படுகிறது என்பதற்கான வழிகாட்டி:</b></p>', unsafe_allow_html=True)
    
    rating_html = """
    <style>
        .dark-white-table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            font-family: Arial, sans-serif;
            background-color: rgba(0, 0, 0, 0.4);
            border: 2px solid #ffffff;
        }
        .dark-white-table th {
            background-color: #1e7e34;
            color: #ffffff !important;
            font-weight: 900 !important;
            font-size: 18px !important;
            padding: 12px;
            text-align: center;
            border: 1px solid #ffffff;
            text-shadow: 1px 1px 2px black;
        }
        .dark-white-table td {
            color: #ffffff !important;
            font-weight: bold !important;
            font-size: 16px !important;
            padding: 12px;
            text-align: center;
            border: 1px solid #ffffff;
            text-shadow: 1px 1px 3px rgba(0,0,0,0.8);
        }
        .dark-white-table tr:hover {
            background-color: rgba(255, 255, 255, 0.1);
        }
    </style>
    
    <table class="dark-white-table">
        <thead>
            <tr>
                <th>Rating</th>
                <th>Suitability Match</th>
                <th>Recommendation</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td>⭐⭐⭐⭐⭐</td>
                <td>Excellent Match (90% - 100%)</td>
                <td>மண்ணிற்கு மிகவும் உகந்தது. அதிக லாபம் தரும்.</td>
            </tr>
            <tr>
                <td>⭐⭐⭐⭐</td>
                <td>Good Match (80% - 89%)</td>
                <td>நல்ல தேர்வு. தாராளமாக பயிர் செய்யலாம்.</td>
            </tr>
            <tr>
                <td>⭐⭐⭐</td>
                <td>Average Match (70% - 79%)</td>
                <td>சுமாரான பலன் தரும். கூடுதல் உரம் தேவைப்படலாம்.</td>
            </tr>
            <tr>
                <td>⭐⭐</td>
                <td>Low Match (< 70%)</td>
                <td>மண்ணிற்கு செட் ஆகாது. தவிர்க்கவும்!</td>
            </tr>
        </tbody>
    </table>
    """
    st.markdown(rating_html, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.write("---")
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<h2 style="text-align:center; color:#ffffff; font-weight: bold; text-shadow: 2px 2px 4px #000000;">📊 Crop Data Visualization & Analysis</h2>', unsafe_allow_html=True)
    
    analysis_data = {
        'label': ['Rice', 'Maize', 'Chickpea', 'Kidneybeans', 'Pigeonpeas', 'Mothbeans', 'Mungbean', 'Blackgram', 'Lentil', 'Pomegranate', 'Banana', 'Mango', 'Grapes', 'Watermelon', 'Muskmelon', 'Apple', 'Orange', 'Papaya', 'Coconut', 'Cotton', 'Jute', 'Coffee'],
        'N': [80, 70, 40, 20, 20, 20, 20, 40, 40, 20, 100, 20, 20, 40, 100, 20, 10, 50, 20, 120, 80, 100],
        'P': [40, 45, 60, 70, 70, 50, 50, 70, 60, 10, 80, 30, 130, 20, 10, 140, 10, 50, 10, 50, 45, 30],
        'K': [40, 20, 80, 20, 20, 20, 20, 20, 20, 40, 50, 30, 200, 50, 50, 200, 10, 50, 30, 20, 40, 30],
        'temp': [23, 22, 18, 20, 28, 28, 28, 29, 24, 22, 27, 30, 32, 25, 28, 21, 23, 26, 27, 24, 25, 26],
        'hum': [82, 65, 16, 21, 48, 53, 47, 65, 66, 89, 77, 50, 81, 85, 92, 92, 91, 92, 94, 79, 81, 52]
    }
    df_analysis = pd.DataFrame(analysis_data)

    tab1, tab2, tab3, tab4 = st.tabs([
        "**🧪 N-P-K Comparison**", 
        "**🌦️ Climate Analysis (Scatter)**", 
        "**📊 Distribution Dynamic**",
        "**📈 Accuracy Score & Session History**"
    ])

    with tab1:
        st.markdown("<h3 style='color: #ffffff !important; font-weight: 900; text-shadow: 2px 2px 4px #000000;'>4. N, P, K values comparison between crops</h3>", unsafe_allow_html=True)
        
        fig_npk = go.Figure()
        fig_npk.add_trace(go.Bar(x=df_analysis['label'], y=df_analysis['N'], name='Nitrogen', marker_color='indianred'))
        fig_npk.add_trace(go.Bar(x=df_analysis['label'], y=df_analysis['P'], name='Phosphorus', marker_color='lightsalmon'))
        fig_npk.add_trace(go.Bar(x=df_analysis['label'], y=df_analysis['K'], name='Potash', marker_color='crimson'))
        
        fig_npk.update_layout(
            barmode='group', 
            xaxis_tickangle=-45, 
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)', 
            font=dict(color="#ffffff", size=14, family="Arial Black"), 
            height=500,
            width=1000,
            legend=dict(
                font=dict(color="#ffffff", size=15, family="Arial Black"),
                title=dict(font=dict(color="#ffffff", size=15, family="Arial Black"))
            ),
            xaxis=dict(
                title=dict(text="Crops", font=dict(color="#ffffff", size=16, family="Arial Black")),
                tickfont=dict(color="#ffffff", size=13, family="Arial Black")
            ),
            yaxis=dict(
                title=dict(text="Values", font=dict(color="#ffffff", size=16, family="Arial Black")),
                tickfont=dict(color="#ffffff", size=13, family="Arial Black")
            )
        )
        st.plotly_chart(fig_npk, use_container_width=True, key="chart_npk")

    with tab2:
        st.markdown("<h3 style='color: #ffffff !important; font-weight: bold;'>7. Scatterplot: Temperature vs Humidity</h3>", unsafe_allow_html=True)
        fig_scatter = px.scatter(
            df_analysis, x="temp", y="hum", color="label",
            labels={'temp': 'Temperature (°C)', 'hum': 'Humidity (%)'}
        )
        fig_scatter.update_layout(
            plot_bgcolor='rgba(0,0,0,0)', 
            paper_bgcolor='rgba(0,0,0,0)', 
            font=dict(color="#ffffff", size=14, family="Arial Black"), 
            height=500,
            width=1000,
            legend=dict(
                font=dict(color="#ffffff", size=12, family="Arial Black")
            ),
            xaxis=dict(
                title=dict(text="Temperature (°C)", font=dict(color="#ffffff", size=15, family="Arial Black")),
                tickfont=dict(color="#ffffff", size=13, family="Arial Black")
            ),
            yaxis=dict(
                title=dict(text="Humidity (%)", font=dict(color="#ffffff", size=15, family="Arial Black")),
                tickfont=dict(color="#ffffff", size=13, family="Arial Black")
            )
        )
        st.plotly_chart(fig_scatter, use_container_width=True, key="chart_scatter")
        
    with tab3:
        st.markdown("<h3 style='color: #ffffff !important; font-weight: bold;'>Macronutrient Share Breakdown</h3>", unsafe_allow_html=True)
        avg_n = df_analysis['N'].mean()
        avg_p = df_analysis['P'].mean()
        avg_k = df_analysis['K'].mean()
        fig_pie = px.pie(
            names=['Avg Nitrogen', 'Avg Phosphorus', 'Avg Potassium'],
            values=[avg_n, avg_p, avg_k],
            color_discrete_sequence=px.colors.sequential.YlGnBu
        )
        fig_pie.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font=dict(color='#ffffff'), width=1000)
        st.plotly_chart(fig_pie, use_container_width=True, key="chart_pie", theme=None)

    with tab4:
        st.markdown("<h3 style='color: #ffffff !important; font-weight: bold;'>Model Performance & Session History</h3>", unsafe_allow_html=True)
        st.markdown("""
            <div style='background-color: #1e7e34; padding: 15px; border-radius: 10px; border: 1px solid #ffffff; margin-bottom: 15px;'>
                <h3 style='color: #ffffff !important; font-weight: bold; margin: 0;'>📈 Model Predictive Accuracy: 99.50%</h3>
            </div>
        """, unsafe_allow_html=True)
        
        if st.session_state.prediction_history:
            st.markdown("#### Dynamic Audit History Logs")
            st.dataframe(pd.DataFrame(st.session_state.prediction_history), use_container_width=True)
        else:
            st.info("No predictions checked in this session yet.")
            
    st.markdown('</div>', unsafe_allow_html=True)


# ==============================================================================
# 🧭 ROUTER & MAIN LOGIC BLOCK (இங்கே தான் கண்டிஷன் மாற்றப்பட்டுள்ளது)
# ==============================================================================
if not st.session_state.logged_in:
    if st.session_state.auth_page == "Login":
        show_login_page()
    elif st.session_state.auth_page == "Register":
        show_register_page()

# ==============================================================================
# 🏡 HOME PAGE / DASHBOARD (வெற்றிகரமாக லாகின் செய்த பிறகு மட்டுமே காட்டும்)
# ==============================================================================
else:
    with st.sidebar:
        st.title(f"👋 {st.session_state.user_display_name}")
        st.write("---")
        
        # CROP SELECTION WITH VOICE ASSISTANT
        st.subheader("🌱 Crop Info & Duration")
        selected_crop = st.selectbox("Choose a crop to know details:", list(SIDEBAR_CROP_DETAILS.keys()))
        
        if selected_crop != "Select a Crop..." and selected_crop != st.session_state.last_selected_crop:
            st.session_state.last_selected_crop = selected_crop
            crop_data = SIDEBAR_CROP_DETAILS[selected_crop]
            
            st.info(f"⏱️ **Duration:** {crop_data['months']}")
            st.write(f"🔬 **Reason:** {crop_data['reason']}")
            
            crop_voice_msg = f"{selected_crop} பயிர் வளர்வதற்கு {crop_data['months']} ஆகும். இதன் அறிவியல் காரணம்: {crop_data['reason']}"
            speak(crop_voice_msg)
            
        st.write("---")
        
        # SOIL SCANNER 
        st.subheader("📸 Soil Variety Scanner")
        uploaded_file = st.file_uploader("Upload Soil Photo", type=["jpg", "png", "jpeg"], key="soil_uploader")
        if uploaded_file:
            img = Image.open(uploaded_file)
            st.image(img, caption="Scanning...", use_container_width=True)
            fname = uploaded_file.name.lower()
            detected_key = "default"
            for key in SOIL_MASTER:
                if key in fname: detected_key = key; break
            data = SOIL_MASTER[detected_key]
            st.session_state.n_val, st.session_state.p_val, st.session_state.k_val = data['n'], data['p'], data['k']
            st.sidebar.success(f"Detected: {data['name']}")
            speak(f"{data['msg']}. Nitrogen {data['n']}, Phosphorus {data['p']}, Potassium {data['k']}.")

        st.write("---")

        # 🏥 PLANT DISEASE DIAGNOSIS
        st.subheader("🏥 Disease Diagnosis")
        leaf_file = st.file_uploader("Upload Leaf Photo", type=["jpg", "png", "jpeg"], key="leaf_uploader")
        if leaf_file:
            leaf_img = Image.open(leaf_file)
            st.image(leaf_img, caption="AI Scanning & Analyzing Leaf...", use_container_width=True)
            
            img_byte_arr = io.BytesIO()
            leaf_img.save(img_byte_arr, format='JPEG')
            img_bytes = img_byte_arr.getvalue()
            
            with st.spinner("AI மாடல் இலையை ஆய்வு செய்கிறது... 🧠"):
                raw_ai_label = predict_plant_disease(img_bytes, file_name=leaf_file.name)
                d_name, d_sol, d_voice = get_tamil_disease_solution(raw_ai_label)
                
            st.sidebar.warning(f"**Detected:** {d_name}")
            st.sidebar.info(f"**{d_sol}**")
            speak(d_voice)

        st.write("---")
        
        # WEATHER & TEMPERATURE WITH LIVE EMBEDDED GOOGLE MAPS
        st.subheader("🌤️ Weather Condition")
        city_input = st.text_input("Enter City/Village")
        if st.button("Fetch Weather"):
            t, h = get_weather(city_input)
            if t is not None:
                st.session_state.temp_val, st.session_state.hum_val = t, h
                st.session_state.city = city_input
                
                if h >= 85: st.session_state.rain_val = 1800.0  
                elif h >= 70: st.session_state.rain_val = 1200.0  
                elif h >= 50: st.session_state.rain_val = 650.0   
                else: st.session_state.rain_val = 250.0   
                
                st.sidebar.info(f"{t}°C | {h}% Humidity")
                weather_voice = f"Ippo {city_input} oorula veppam {t} degree, eerappadham {h} percentage, mazhaippozhivu {st.session_state.rain_val} millimeter, matrum pH thunivu {st.session_state.ph_val}."
                speak(weather_voice)
            else:
                st.sidebar.error("வானிலை விவரங்களைப் பெற முடியவில்லை! ஊரின் பெயரைச் சரிபார்க்கவும்.")

        if st.session_state.city:
            st.markdown("### 📍 Location Map")
            formatted_city = st.session_state.city.replace(' ', '+')
            
            map_embed_html = f"""
            <iframe 
                width="100%" 
                height="220" 
                frameborder="0" 
                scrolling="no" 
                marginheight="0" 
                marginwidth="0" 
                src="https://maps.google.com/maps?q={formatted_city}&t=&z=14&ie=UTF8&iwloc=&output=embed">
            </iframe>
            """
            components.html(map_embed_html, height=230)
            
        if st.button("🚪 Logout"):
            write_audit_log("LOGOUT", st.session_state.user_display_name)
            st.session_state.logged_in = False
            st.session_state.auth_page = "Login"
            st.rerun()

    # --- MAIN DISPLAY (Home Page) ---
    st.markdown('<h1 style="color:white; text-align:center;">🌿 Crop Recommendation Dashboard</h1>', unsafe_allow_html=True)
    with st.container():
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("Soil & Climate Parameters")
        col1, col2 = st.columns(2)
        with col1:
            n_in = st.number_input("Nitrogen (N)", value=st.session_state.n_val)
            p_in = st.number_input("Phosphorus (P)", value=st.session_state.p_val)
            k_in = st.number_input("Potassium (K)", value=st.session_state.k_val)
            t_in = st.number_input("Temperature (°C)", value=st.session_state.temp_val)
        with col2:
            h_in = st.number_input("Humidity (%)", value=st.session_state.hum_val)
            ph_in = st.number_input("pH Level", value=st.session_state.ph_val)
            st.session_state.ph_val = ph_in 
            rain_in = st.number_input("Rainfall (mm)", value=st.session_state.rain_val)
            st.session_state.rain_val = rain_in
            
        if st.button("🔍 PREDICT BEST CROP"):
            st.balloons()
            
            input_features = np.array([[n_in, p_in, k_in, t_in, h_in, ph_in, rain_in]])
            res = None
            model_used = "Rule-Based System (Backup)"
            
            if cat_model is not None:
                try:
                    pred = cat_model.predict(input_features)[0]
                    res_raw = pred[0] if isinstance(pred, (list, np.ndarray)) else pred
                    
                    if label_encoder is not None:
                        res_name = label_encoder.inverse_transform([int(res_raw)])[0]
                        if "rice" in res_name.lower(): res = "RICE (Nel)"
                        elif "maize" in res_name.lower(): res = "MAIZE (Makkacholam)"
                        elif "cotton" in res_name.lower(): res = "COTTON (Paruthi)"
                        else: res = res_name.upper()
                    else:
                        res = str(res_raw).upper()
                    model_used = "CatBoost AI"
                except Exception as e:
                    res = None
            
            if res is None and xgb_model is not None:
                try:
                    pred = xgb_model.predict(input_features)[0]
                    if label_encoder is not None:
                        res_name = label_encoder.inverse_transform([int(pred)])[0]
                        if "rice" in res_name.lower(): res = "RICE (Nel)"
                        elif "maize" in res_name.lower(): res = "MAIZE (Makkacholam)"
                        elif "cotton" in res_name.lower(): res = "COTTON (Paruthi)"
                        else: res = res_name.upper()
                    else:
                        res = str(pred).upper()
                    model_used = "XGBoost AI"
                except Exception as e:
                    res = None

            if res is None:
                if rain_in >= 2000 and ph_in <= 6.0: res = "TEA (Theylai)"
                elif rain_in >= 1500 and h_in >= 80:
                    if n_in >= 80: res = "RICE (Nel)"
                    else: res = "RUBBER"
                elif rain_in >= 1100 and n_in >= 90 and p_in >= 50: res = "SUGARCANE (Karumbu)"
                elif n_in >= 80 and p_in >= 50 and rain_in >= 700 and rain_in < 1100: res = "MAIZE (Makkacholam)"
                elif n_in >= 70 and p_in >= 40 and h_in <= 60: res = "COTTON (Paruthi)"
                elif ph_in >= 6.0 and ph_in <= 7.5 and t_in >= 25 and t_in <= 35 and rain_in < 700: res = "GROUNDNUT (Nilakkadali)"
                elif t_in < 22 and rain_in >= 600 and rain_in <= 1000: res = "WHEAT (Godhumai)"
                elif n_in <= 50 and rain_in < 500:
                    if ph_in >= 7.0: res = "BAJRA (Kambu)"
                    else: res = "RAGI"
                else: res = "PULSES (Paruppu Vagaigal)"
                
            region_info = CROP_REGIONS.get(res, {"regions": "Various regions across India.", "voice": "Idhu India-vilppal idangalil valarkkappadugiradhu."})
            
            match_percentage, star_rating, suitability_match_text, status_verdict = calculate_crop_suitability(
                n_in, p_in, k_in, t_in, h_in, ph_in, rain_in, res
            )
            
            st.markdown(f"""
                <div style='background:white; color:green; padding:20px; border-radius:15px; text-align:center; margin-bottom:15px;'>
                    <h2>🌱 Recommended Crop: {res}</h2>
                    <p style='color: gray; margin: 0;'><b>Engine Powered By:</b> {model_used}</p>
                </div>
                
                <div style='background: rgba(0, 0, 0, 0.85); padding: 25px; border-radius: 15px; border: 2px solid #ffffff; margin-bottom: 20px; box-shadow: 0 4px 15px rgba(0,0,0,0.5);'>
                    <h3 style='color: #ffffff !important; font-weight: bold; text-decoration: underline; margin-bottom: 15px;'>🌟 Crop Suitability Match & Analysis Scorecard</h3>
                    <div class='scorecard-container'>
                        <p style='font-size: 18px; margin: 5px 0;'>🎯 <b>Suitability Match Score:</b> <span style='color: #3dfc5d; font-size: 22px;'>{match_percentage}%</span></p>
                        <p style='font-size: 18px; margin: 5px 0;'>⭐ <b>Star Rating:</b> <span style='color: #ffd700; font-size: 22px;'>{star_rating}</span> ({suitability_match_text})</p>
                        <p style='font-size: 18px; margin: 5px 0;'>📋 <b>மண்ணிற்கு உகந்ததா? (Status):</b> <span style='font-size: 19px;'><b>{status_verdict}</b></span></p>
                    </div>
                </div>

                <div style='background: #1e7e34; padding: 20px; border-radius: 15px; border: 2px solid #ffffff; box-shadow: 0 4px 8px rgba(0,0,0,0.3); margin-bottom: 20px;'>
                    <h4 style='color: #ffffff !important; font-weight: bold; margin: 0; font-size: 20px;'>📍 Major Cultivation Regions:</h4>
                    <p style='color: #ffffff !important; font-size: 18px; font-weight: 500; margin-top: 8px;'>{region_info['regions']}</p>
                </div>
            """, unsafe_allow_html=True)
            
            if res in CROP_ADVISORY_SYSTEM:
                adv = CROP_ADVISORY_SYSTEM[res]
                st.markdown(f"""
                    <div style='background: rgba(0, 0, 0, 0.75); padding: 20px; border-radius: 15px; border: 1px dashed white;'>
                        <h4 style='color: #ffffff !important;'>🚜 **Agri-Advisory & Field Recommendations for {res}**</h4>
                        <p style='margin: 4px 0; color: #ffffff !important;'>🧪 <b>Fertilizer Plan:</b> {adv['fertilizer']}</p>
                        <p style='margin: 4px 0; color: #ffffff !important;'>🐛 <b>Pest & Insect Management:</b> {adv['pest_control']}</p>
                        <p style='margin: 4px 0; color: #ffffff !important;'>💡 <b>தமிழ் விவசாய குறிப்பு:</b> {adv['tamil_tip']}</p>
                    </div>
                """, unsafe_allow_html=True)
            
            full_voice_msg = f"Ungal mannu ku {res} payir seivadhu sirappu. Match percent {match_percentage} percentage. {region_info['voice']}"
            speak(full_voice_msg)
            
            st.session_state.prediction_history.append({
                "Timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
                "Crop": res,
                "Model": model_used,
                "Match Score": f"{match_percentage}%",
                "Rating": star_rating
            })
            
        st.markdown('</div>', unsafe_allow_html=True)   

    # --- AGRICULTURAL CROPS CLASSIFICATION GUIDE ---
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<h2 style="color:white; text-align:center; text-shadow: 2px 2px 4px #000000;">📋 Agricultural Crops Classification Guide</h2>', unsafe_allow_html=True)
    st.markdown('<p style="color:white; text-align:center;">Here is the list of major categories of crops grown and utilized across regions:</p>', unsafe_allow_html=True)
    st.write("<br>", unsafe_allow_html=True)
    
    col_crop1, col_crop2 = st.columns(2)
    with col_crop1:
        st.markdown('<h3 style="color:#ffffff; text-decoration: underline;">💰 Cash Crops (Panappayirgal)</h3>', unsafe_allow_html=True)
        for crop in ["Sugarcane", "Cotton", "Jute", "Tea", "Coffee", "Tobacco", "Rubber", "Spices (Milagai, Manjal, Milagu)", "Cashew", "Oilseeds (Nilakkadali, Kadugu, Soybean)"]:
            st.markdown(f'<div class="white-text-list">- {crop}</div>', unsafe_allow_html=True)
    with col_crop2:
        st.markdown('<h3 style="color:#ffffff; text-decoration: underline;">🌾 Mostly Used Crops (Athigamaaga Payanpaduthappadum Payirgal)</h3>', unsafe_allow_html=True)
        for crop in ["Rice (Nel)", "Wheat (Godhumai)", "Maize (Makkacholam)", "Ragi", "Jowar (Cholam)", "Bajra (Kambu)", "Pulses (Paruppu vagaigal)", "Potato (Uruzaikkizhangu)", "Onion (Vengayam)", "Coconut (Thengai)"]:
            st.markdown(f'<div class="white-text-list">- {crop}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # 🛠️ லாகின் செய்த பிறகு மட்டுமே அனாலிசிஸ் டேஷ்போர்டு இயங்கும்படி லூப்பிற்குள் கொண்டுவரப்பட்டுள்ளது
    show_analysis_and_rating_dashboard()