import streamlit as st
import time
import tensorflow as tf
import torch
import torch.nn as nn
import numpy as np

# Registra la clase Sequential en los objetos personalizados de Keras
tf.keras.utils.get_custom_objects()['Sequential'] = tf.keras.models.Sequential

# Parámetros
ACCIDENT_IMG_SIZE = 160   # Tamaño para modelo de accidentes
FIRE_IMG_SIZE = 128       # Tamaño para modelo de incendios
SEQUENCE_LENGTH = 5

# Definir y registrar la clase personalizada de TimeDistributed
from tensorflow.keras.layers import TimeDistributed as OriginalTimeDistributed

class FixedTimeDistributed(OriginalTimeDistributed):
    def __init__(self, *args, **kwargs):
        super(FixedTimeDistributed, self).__init__(*args, **kwargs)
        self._self_tracked_trackables = {}

tf.keras.utils.get_custom_objects()['TimeDistributed'] = FixedTimeDistributed

# Definir custom_objects usando la clase arreglada directamente
custom_objects = {
    'DTypePolicy': tf.keras.mixed_precision.Policy,
    'TimeDistributed': FixedTimeDistributed
}

# ----- Prueba 1: Cargar modelo de Accidentes (TensorFlow) -----
def test_load_accident_model():
    start_time = time.time()
    try:
        accident_model = tf.keras.models.load_model('models/model_car.h5', compile=False, custom_objects=custom_objects)
        accident_model.compile(jit_compile=True)
        dummy_accident = tf.zeros((1, SEQUENCE_LENGTH, ACCIDENT_IMG_SIZE, ACCIDENT_IMG_SIZE, 3))
        accident_model(dummy_accident)
        elapsed = time.time() - start_time
        st.success(f"Modelo de Accidentes cargado en {elapsed:.2f} segundos")
    except Exception as e:
        st.error(f"Error al cargar el modelo de Accidentes: {e}")

# ----- Prueba 2: Cargar modelo de Incendios (TensorFlow) -----
def test_load_fire_model():
    start_time = time.time()
    try:
        fire_model = tf.keras.models.load_model('models/model_fire.h5', compile=False, custom_objects=custom_objects)
        fire_model.compile(jit_compile=True)
        dummy_fire = tf.zeros((1, FIRE_IMG_SIZE, FIRE_IMG_SIZE, 3))
        fire_model(dummy_fire)
        elapsed = time.time() - start_time
        st.success(f"Modelo de Incendios cargado en {elapsed:.2f} segundos")
    except Exception as e:
        st.error(f"Error al cargar el modelo de Incendios: {e}")

# ----- Prueba 3: Cargar modelo de Peleas (PyTorch) -----
def test_load_fight_model():
    start_time = time.time()
    try:
        class SimpleVideoClassifier(nn.Module):
            def __init__(self, num_classes=1):
                super(SimpleVideoClassifier, self).__init__()
                self.resnet = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=False)
                self.resnet.fc = nn.Identity()
                self.fc = nn.Linear(512, num_classes)
            
            def forward(self, x):
                B, C, T, H, W = x.shape
                x = x.permute(0, 2, 1, 3, 4).contiguous().view(B * T, C, H, W)
                features = self.resnet(x)
                features = features.view(B, T, -1)
                features = features.mean(dim=1)
                out = self.fc(features)
                return out
        
        fight_model = SimpleVideoClassifier()
        fight_model.load_state_dict(torch.load('models/model_fight.pth', map_location=torch.device('cpu')))
        fight_model.eval()
        elapsed = time.time() - start_time
        st.success(f"Modelo de Peleas (PyTorch) cargado en {elapsed:.2f} segundos")
    except Exception as e:
        st.error(f"Error al cargar el modelo de Peleas: {e}")

st.title("Pruebas de carga de modelos")

st.header("1. Modelo de Accidentes (TensorFlow)")
if st.button("Cargar Modelo de Accidentes"):
    test_load_accident_model()

st.header("2. Modelo de Incendios (TensorFlow)")
if st.button("Cargar Modelo de Incendios"):
    test_load_fire_model()

st.header("3. Modelo de Peleas (PyTorch)")
if st.button("Cargar Modelo de Peleas"):
    test_load_fight_model()
