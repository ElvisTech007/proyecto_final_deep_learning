import numpy as np
import pandas as pd
from scipy.signal import welch
from scipy.integrate import simps as simpson
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os

# # --- 1. REPRODUCIBILIDAD (Vital para tu reporte) ---
SEED = 9876
np.random.seed(SEED)


# --- CONFIGURACIÓN ---
NPZ_PATH = 'datos_mejorados.npz' 
FS = 250
BATCH_SIZE = 500 

BANDS = {
    'Delta': (0.5, 4), 'Theta': (4, 8),
    'Alpha': (8, 13), 'Beta': (13, 30), 'Gamma': (30, 45)
}

def extraer_features_low_ram(path_npz):
    print(f"--- EXTRACCIÓN DE FEATURES (LOGISTIC REGRESSION) ---")
    data = np.load(path_npz, mmap_mode='r')
    X_disk = data['X'] 
    y = data['y']      
    groups = data['groups']
    
    n_ventanas = X_disk.shape[0]
    n_canales = X_disk.shape[2]
    X_features = np.zeros((n_ventanas, 95), dtype=np.float32)
    
    for i in range(0, n_ventanas, BATCH_SIZE):
        fin = min(i + BATCH_SIZE, n_ventanas)
        batch = np.array(X_disk[i:fin])
        freqs, psd = welch(batch, fs=FS, nperseg=FS, axis=1)
        
        for j in range(len(batch)):
            idx_global = i + j
            row = []
            for c in range(n_canales):
                psd_c = psd[j, :, c]
                idx_total = np.logical_and(freqs >= 0.5, freqs <= 45)
                total_power = simpson(psd_c[idx_total], dx=freqs[1]-freqs[0])
                if total_power == 0: total_power = 1.0
                
                for banda, (low, high) in BANDS.items():
                    idx_band = np.logical_and(freqs >= low, freqs <= high)
                    power = simpson(psd_c[idx_band], dx=freqs[1]-freqs[0])
                    row.append(power / total_power)
            X_features[idx_global] = np.array(row, dtype=np.float32)
        
        if i % 2000 == 0: print(f"   > Procesado {i}/{n_ventanas}...")
    return X_features, y, groups

def main():
    # 1. Obtener Features
    X, y, groups = extraer_features_low_ram(NPZ_PATH)
    
    # 2. Validación LOSO
    logo = LeaveOneGroupOut()
    patient_results = []
    
    print("\n--- INICIANDO BASELINE: REGRESIÓN LOGÍSTICA ---")
    
    # Modelo Lineal (Aumentamos max_iter para asegurar convergencia)
    clf = LogisticRegression(max_iter=1000, solver='lbfgs', class_weight='balanced', random_state=SEED)
    
    fold = 1
    for train_idx, test_idx in logo.split(X, y, groups=groups):
        subject = np.unique(groups[test_idx])[0]
        true_lbl = np.unique(y[test_idx])[0]
        
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Scaling (CRUCIAL para Regresión Logística)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        # Entrenar
        clf.fit(X_train, y_train)
        
        # Predecir Ventanas
        preds = clf.predict(X_test)
        
        # Votación
        votes_sz = np.sum(preds == 1)
        votes_ct = np.sum(preds == 0)
        final_pred = 1 if votes_sz > votes_ct else 0
        is_correct = (final_pred == true_lbl)
        
        # Confianza (Probabilidad basada en votos)
        confidence = max(votes_sz, votes_ct) / len(preds)
        
        patient_results.append({
            'Sujeto': subject, 'Real': true_lbl, 'Pred': final_pred, 
            'Ok': is_correct, 'Votos_SZ': votes_sz, 'Votos_CT': votes_ct
        })
        
        icon = "✅" if is_correct else "❌"
        print(f"Sujeto {subject}: {icon} (Conf: {confidence:.2f})")

    # --- REPORTES ---
    df = pd.DataFrame(patient_results)
    acc = df['Ok'].mean()
    
    print("\n" + "="*40)
    print(f"ACCURACY REGRESIÓN LOGÍSTICA: {acc*100:.2f}%")
    print("="*40)
    
    # Matriz
    cm = confusion_matrix(df['Real'], df['Pred'])
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='YlGnBu')
    plt.title(f'Baseline Logística (Acc: {acc*100:.1f}%)')
    plt.xlabel('Predicción')
    plt.ylabel('Realidad')
    plt.savefig('resultado_baseline_logistica_IMPROVED.png')
    print("Gráfica guardada: resultado_baseline_logistica.png")

if __name__ == "__main__":
    main()