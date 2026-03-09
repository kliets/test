import streamlit as st
from supabase import create_client
import numpy as np
from PIL import Image, ImageOps
import tensorflow as tf
import uuid

# ==========================
# KONFIGURATION & VERBINDUNG
# ==========================
# In st.secrets müssen SUPABASE_URL und SUPABASE_KEY stehen
supabase = create_client(st.secrets["SUPABASE_URL"], st.secrets["SUPABASE_KEY"])
BUCKET_NAME = "fundbuero"

# Modell laden (Dateien müssen im selben Ordner liegen)
@st.cache_resource
def load_prediction_model():
    model = tf.keras.models.load_model("keras_model.h5", compile=False)
    class_names = open("labels.txt", "r").readlines()
    return model, class_names

def classify_image(img):
    model, class_names = load_prediction_model()
    # Bild für Modell vorbereiten
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    size = (224, 224)
    image = ImageOps.fit(img, size, Image.Resampling.LANCZOS)
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    data[0] = normalized_image_array

    # Vorhersage
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index]
    return class_name.strip()

# ==========================
# UI
# ==========================
st.title("🏫 Schul-Fundbüro")

uploaded_file = st.file_uploader("Foto des Fundstücks", type=["jpg", "jpeg", "png"])
beschreibung = st.text_input("Was wurde gefunden?")
fundort = st.text_input("Wo wurde es gefunden?")

if st.button("Fundstück speichern"):
    if uploaded_file and beschreibung:
        # KI Klassifizierung
        image = Image.open(uploaded_file)
        kategorie = classify_image(image)
        st.info(f"KI erkennt: {kategorie}")

        # Upload Logik
        filename = f"{uuid.uuid4()}.jpg"
        
        # Cursor an Anfang der Datei setzen für Upload
        uploaded_file.seek(0)
        supabase.storage.from_(BUCKET_NAME).upload(filename, uploaded_file.read())

        image_url = f"{st.secrets['SUPABASE_URL']}/storage/v1/object/public/{BUCKET_NAME}/{filename}"

        # In DB speichern
        supabase.table("fundbuero").insert({
            "kategorie": kategorie,
            "beschreibung": beschreibung,
            "fundort": fundort,
            "bild_url": image_url,
            "status": "Offen"
        }).execute()

        st.success("Erfolgreich gespeichert!")
    else:
        st.warning("Bitte Bild und Beschreibung ausfüllen.")

st.divider()
# ... (Anzeige-Teil bleibt ähnlich wie in deinem Code)
