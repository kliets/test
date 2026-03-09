import streamlit as st
from supabase import create_client
import numpy as np
from PIL import Image, ImageOps
import tensorflow as tf
import uuid

# ==========================
# 1. SETUP & VERBINDUNG
# ==========================
# Holen der Keys aus den Streamlit Secrets
SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

BUCKET_NAME = "fundbuero"

# ==========================
# 2. KI FUNKTION (Teachable Machine)
# ==========================
@st.cache_resource
def load_prediction_model():
    # Lädt Modell und Labels aus deinem GitHub Ordner
    model = tf.keras.models.load_model("keras_model.h5", compile=False)
    with open("labels.txt", "r") as f:
        class_names = f.readlines()
    return model, class_names

def classify_image(img):
    model, class_names = load_prediction_model()
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    
    # Bildvorbereitung
    size = (224, 224)
    image = ImageOps.fit(img, size, Image.Resampling.LANCZOS)
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    data[0] = normalized_image_array

    # Vorhersage
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index].strip()
    
    # Entfernt Zahlen am Anfang (z.B. "0 Brotdose" -> "Brotdose")
    return "".join([i for i in class_name if not i.isdigit()]).strip()

# ==========================
# 3. BENUTZEROBERFLÄCHE (EINGABE)
# ==========================
st.set_page_config(page_title="Schul-Fundbüro", layout="centered")
st.title("🏫 Digitales Schul-Fundbüro")

st.header("📤 Neues Fundstück melden")
uploaded_file = st.file_uploader("Foto des Gegenstands", type=["jpg", "png", "jpeg"])
fundort = st.text_input("Fundort (z.B. Pausenhof, Raum 204)")
beschreibung = st.text_area("Zusätzliche Infos (Farbe, Marke...)")

if st.button("🚀 Fundstück speichern"):
    if uploaded_file:
        try:
            # Bild verarbeiten & KI Analyse
            uploaded_file.seek(0)
            image = Image.open(uploaded_file)
            kategorie = classify_image(image)
            st.info(f"KI erkennt: **{kategorie}**")

            # Bild-Upload zu Supabase Storage
            filename = f"{uuid.uuid4()}.jpg"
            uploaded_file.seek(0)
            supabase.storage.from_(BUCKET_NAME).upload(filename, uploaded_file.read())
            
            # Link zum Bild erstellen
            img_url = f"{SUPABASE_URL}/storage/v1/object/public/{BUCKET_NAME}/{filename}"

            # Daten in Tabelle schreiben
            supabase.table("fundbuero").insert({
                "kategorie": kategorie,
                "beschreibung": beschreibung,
                "fundort": fundort,
                "bild_url": img_url,
                "status": "Offen"
            }).execute()

            st.success("Erfolgreich in der Datenbank gespeichert!")
            st.balloons()
            st.rerun() # Seite neu laden, um Liste zu aktualisieren

        except Exception as e:
            st.error(f"Fehler: {e}")
    else:
        st.warning("Bitte lade ein Bild hoch.")

# ==========================
# 4. DATENBANK-ANZEIGE (FÜR SUCHER)
# ==========================
st.divider()
st.header("🔍 Gefundene Gegenstände durchsuchen")

# Daten aus Supabase abrufen
try:
    response = supabase.table("fundbuero").select("*").order("created_at", desc=True).execute()
    fundstuecke = response.data

    if not fundstuecke:
        st.write("Aktuell sind keine Gegenstände gemeldet.")
    else:
        # Erstellt ein Raster mit 2 Spalten für die Bilder
        cols = st.columns(2)
        for i, item in enumerate(fundstuecke):
            with cols[i % 2]:
                st.image(item["bild_url"], use_container_width=True)
                st.subheader(f"{item['kategorie']}")
                st.write(f"📍 **Ort:** {item['fundort']}")
                if item['beschreibung']:
                    st.write(f"📝 **Details:** {item['beschreibung']}")
                st.caption(f"Status: {item['status']} | Datum: {item['created_at'][:10]}")
                st.markdown("---")
except Exception as e:
    st.error(f"Fehler beim Laden der Liste: {e}")
