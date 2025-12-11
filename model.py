import numpy as np
import pandas as pd
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from scipy.signal import welch
from scipy.integrate import simpson
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import gc
import keras

# # --- 1. REPRODUCIBILIDAD
SEED = 9876
keras.utils.set_random_seed(42)
tf.random.set_seed(SEED)

# --- CONFIGURACIÓN ---
FS = 250
BATCH_SIZE = 500 # Procesar de 500 en 500 para cuidar la RAM
BANDS = {
    'Delta': (0.5, 4),
    'Theta': (4, 8),
    'Alpha': (8, 13),
    'Beta':  (13, 30),
    'Gamma': (30, 45)
}

def extraer_features_low_ram(path_npz):
    """
    Extrae características leyendo directo del disco duro (mmap) 
    para no saturar la RAM.
    """
    print(f"--- INICIANDO EXTRACCIÓN LOW-RAM ---")
    
    # 1. Cargar como mapa de memoria (No consume RAM instantánea)
    data_disk = np.load(path_npz, mmap_mode='r')
    X_disk = data_disk['X'] 
    y = data_disk['y']      
    groups = data_disk['groups']
    
    n_ventanas = X_disk.shape[0]
    n_canales = X_disk.shape[2]
    
    print(f"Dataset en disco: {X_disk.shape}")
    
    # Reservar espacio para features (95 columnas)
    X_features = np.zeros((n_ventanas, 95), dtype=np.float32)
    
    # 2. Procesar por lotes
    print(f"Procesando en lotes de {BATCH_SIZE}...")
    for i in range(0, n_ventanas, BATCH_SIZE):
        fin = min(i + BATCH_SIZE, n_ventanas)
        
        # Cargar solo el pedacito necesario a RAM
        batch_raw = np.array(X_disk[i:fin]) 
        
        # Calcular Welch (PSD)
        freqs, psd = welch(batch_raw, fs=FS, nperseg=FS, axis=1)
        
        # Calcular Potencia Relativa por Bandas
        for j in range(len(batch_raw)):
            idx_global = i + j
            row = []
            
            for c in range(n_canales):
                psd_c = psd[j, :, c]
                # Potencia total (para normalizar)
                idx_total = np.logical_and(freqs >= 0.5, freqs <= 45)
                total_power = simpson(psd_c[idx_total], dx=freqs[1]-freqs[0])
                if total_power == 0: total_power = 1.0
                
                for banda, (low, high) in BANDS.items():
                    idx_band = np.logical_and(freqs >= low, freqs <= high)
                    power = simpson(psd_c[idx_band], dx=freqs[1]-freqs[0])
                    # Feature Relativa
                    row.append(power / total_power)
            
            X_features[idx_global] = np.array(row, dtype=np.float32)
        
        # Limpiar RAM del lote inmediatamente
        del batch_raw
        del psd
        gc.collect() # Forzar limpieza
        
        if i % 2000 == 0:
            print(f"   > Procesado {i}/{n_ventanas}...")

    print("Extracción completada. Features listas en RAM (muy ligeras).")
    return X_features, y, groups

def construir_mlp(input_dim):
    # Red Neuronal Densa
    model = Sequential()
    model.add(Dense(64, input_dim=input_dim, activation='relu', kernel_initializer='he_normal'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    
    model.add(Dense(32, activation='relu', kernel_initializer='he_normal'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    
    model.add(Dense(1, activation='sigmoid'))
    
    opt = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
    return model

def main():
    # 1. Extracción Protegida
    X_feats, y, groups = extraer_features_low_ram('datos_eeg_procesados.npz')
    
    # 2. Validación LOSO + Votación
    logo = LeaveOneGroupOut()
    patient_results = []
    
    print("\n--- INICIANDO DIAGNÓSTICO CLÍNICO (DEEP LEARNING + VOTING) ---")
    
    fold = 1
    # Iterar por Paciente
    for train_idx, test_idx in logo.split(X_feats, y, groups=groups):
        
        subject_id = np.unique(groups[test_idx])[0]
        true_label = np.unique(y[test_idx])[0]
        
        # Split
        X_train, X_test = X_feats[train_idx], X_feats[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Escalado
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        # Modelo
        model = construir_mlp(X_train.shape[1])
        
        # Callbacks
        early_stop = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=5)
        
        # Entrenar (Silencioso para no ensuciar la consola)
        model.fit(X_train, y_train, epochs=50, batch_size=32, 
                  callbacks=[early_stop, reduce_lr], verbose=0)
        
        # --- VOTACIÓN ---
        # Predecimos todas las ventanas de ESTE paciente
        y_pred_windows = (model.predict(X_test, verbose=0) > 0.5).astype(int).flatten()
        
        votos_sz = np.sum(y_pred_windows == 1)
        votos_ct = np.sum(y_pred_windows == 0)
        
        # El paciente es lo que diga la mayoría de sus ventanas
        final_prediction = 1 if votos_sz > votos_ct else 0
        is_correct = (final_prediction == true_label)
        
        # Guardar métricas
        confidence = max(votos_sz, votos_ct) / len(y_pred_windows)
        
        patient_results.append({
            'Sujeto': subject_id,
            'Real': true_label,
            'Predicho': final_prediction,
            'Acierto': is_correct,
            'Confianza': confidence
        })
        
        res_icon = "✅" if is_correct else "❌"
        print(f"Paciente {subject_id}: {res_icon} (Confianza: {confidence:.2f}) - Votos SZ: {votos_sz}, CT: {votos_ct}")
        
        fold += 1
        # Limpiar memoria de Keras entre iteraciones
        tf.keras.backend.clear_session()
        gc.collect()

    # --- REPORTE FINAL ---
    df = pd.DataFrame(patient_results)
    acc_final = df['Acierto'].mean()
    
    print("\n" + "="*50)
    print(f"ACCURACY CLÍNICO FINAL: {acc_final*100:.2f}%")
    print("="*50)
    print(df)
    
    # Matriz Final
    cm = confusion_matrix(df['Real'], df['Predicho'])
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='RdYlGn',
                xticklabels=['Control', 'Esquizofrenia'],
                yticklabels=['Control', 'Esquizofrenia'])
    plt.title(f'Diagnóstico por Paciente\nAcc: {acc_final*100:.1f}%')
    plt.ylabel('Realidad')
    plt.xlabel('Predicción')
    plt.savefig('resultado_final_completo.png')
    print("Gráfica guardada: resultado_final_completo_IMPROVED.png")

if __name__ == "__main__":
    main()