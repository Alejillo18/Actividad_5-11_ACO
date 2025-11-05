# 1. Definición de la Nueva Arquitectura: [4 entradas, 6 ocultas, 5 salidas]
    nn_expanded = SimpleNeuralNetwork([4, 6, 5]) 

    # 2. Generación de la Nueva Tabla de Verdad (Datos de Entrenamiento)
    
    # ENTRADAS (4): [Dist, Pos, Luz (1=Oscuro), Línea (1=Detecta)]
    # SALIDAS (5): [M1, M2, M3, M4, LED]

    # Casos de entrenamiento combinados:
    X_expanded = np.array([
        [-1, 0, -1, -1], # Avance Normal 
        [0, 1, -1, -1],  # Giro Izquierda (Cerca Obstáculo Izquierda)
        [1, 0, 1, -1],   # M. A: Retroceso Agresivo (Muy Cerca, Oscuro)
        [-1, 0, -1, 1],  # M. B: Giro Izquierda para Seguir Línea 
        [-1, 0, 1, -1],  # M. C: Advertencia Constante (Avanzar, LED ON)
        [1, 0, 1, 1]     # M. B: FRENADO de Emergencia
    ])

    # Salidas esperadas: [M1, M2, M3, M4, LED]
    y_expanded = np.array([
        [1, 0, 0, 1, 0], # Avanzar, LED OFF
        [1, 0, 1, 0, 0], # Giro Izquierda, LED OFF
        [0, 1, 1, 0, 1], # Retroceder, LED ON
        [1, 0, 1, 0, 0], # Giro Izquierda (para centrar), LED OFF
        [1, 0, 0, 1, 1], # Avanzar, LED ON
        [0, 0, 0, 0, 1]  # STOP/FRENAR, LED ON
    ])

    # 3. Modificar y Ajustar la Red Neuronal (Simulación del Entrenamiento)
    nn_expanded.fit(X_expanded, y_expanded, layers=[4, 6, 5])
    
    # -----------------------------------------------------------
    # 4. Generar y Desplegar Nueva Tabla de Verdad por Miembro
    # -----------------------------------------------------------
    print("\n" + "="*80)
    print("           TABLAS DE VERDAD EXPANDIDAS POR MIEMBRO (Punto 4)            ")
    print("  (El resultado 'Obtenido' es una predicción simulada con pesos aleatorios) ")
    print("="*80)

    # --- Estrategia de Miembro A: Evasión Agresiva ---
    X_A = np.array([[-1, 0, -1, -1], [0, 1, -1, -1], [1, 0, 1, -1]])
    y_A_exp = np.array([[1, 0, 0, 1, 0], [1, 0, 1, 0, 0], [0, 1, 1, 0, 1]])

    print("\n--- Estrategia de M. A: Evasión Agresiva y Advertencia en Oscuridad ---")
    print("| Dist | Pos | Luz | Línea | Esperado: [M1-M4, LED] | Predicción Binarizada |")
    print("|------|-----|-----|-------|------------------------|-----------------------|")
    for i, e in enumerate(X_A):
        prediccion_float = nn_expanded.forward_propagation(e)
        out = [valNN(p) for p in prediccion_float]
        print(f"| {e[0]:4} | {e[1]:3} | {e[2]:3} | {e[3]:5} | {y_A_exp[i]} | {out} |")

    
    # --- Estrategia de Miembro B: Seguimiento de Línea y Frenado ---
    X_B = np.array([[-1, 0, -1, -1], [-1, 0, -1, 1], [1, 0, 1, 1]])
    y_B_exp = np.array([[1, 0, 0, 1, 0], [1, 0, 1, 0, 0], [0, 0, 0, 0, 1]])

    print("\n--- Estrategia de M. B: Seguimiento de Línea y Frenado en Emergencia ---")
    print("| Dist | Pos | Luz | Línea | Esperado: [M1-M4, LED] | Predicción Binarizada |")
    print("|------|-----|-----|-------|------------------------|-----------------------|")
    for i, e in enumerate(X_B):
        prediccion_float = nn_expanded.forward_propagation(e)
        out = [valNN(p) for p in prediccion_float]
        print(f"| {e[0]:4} | {e[1]:3} | {e[2]:3} | {e[3]:5} | {y_B_exp[i]} | {out} |")

    
    # --- Estrategia de Miembro C: Advertencia Constante ---
    X_C = np.array([[-1, 0, -1, -1], [0, 1, -1, -1], [-1, 0, 1, -1], [1, 1, 1, 1]])
    y_C_exp = np.array([[1, 0, 0, 1, 0], [0, 1, 1, 0, 0], [1, 0, 0, 1, 1], [0, 1, 1, 0, 1]])

    print("\n--- Estrategia de M. C: Advertencia Constante y Retroceso Simple ---")
    print("| Dist | Pos | Luz | Línea | Esperado: [M1-M4, LED] | Predicción Binarizada |")
    print("|------|-----|-----|-------|------------------------|-----------------------|")
    for i, e in enumerate(X_C):
        prediccion_float = nn_expanded.forward_propagation(e)
        out = [valNN(p) for p in prediccion_float]
        print(f"| {e[0]:4} | {e[1]:3} | {e[2]:3} | {e[3]:5} | {y_C_exp[i]} | {out} |")


if __name__ == "__main__":
    run_expanded_training_simulation()