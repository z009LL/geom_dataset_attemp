import json
import numpy as np

base_path = ''

# Función para cargar datos JSON
def load_json(file_path):
    with open(file_path) as f:
        data = json.load(f)
    return data

# Cargar los datos de los archivos JSON
training_challenges = load_json(base_path + 'arc-agi_training_challenges.json')
training_solutions = load_json(base_path + 'arc-agi_training_solutions.json')
evaluation_challenges = load_json(base_path + 'arc-agi_evaluation_challenges.json')
evaluation_solutions = load_json(base_path + 'arc-agi_evaluation_solutions.json')

# Función para convertir una lista a una matriz utilizando NumPy
def convertir_a_matriz(lista):
    return np.array(lista)

# Obtener las matrices de todas las tareas y agregar el output faltante
def obtener_matrices_con_output_faltante(task_id, challenges, solutions):
    task = challenges[task_id]
    pares = []
    
    # Procesar pares de entrenamiento
    for pair in task['train']:
        input_matrix = convertir_a_matriz(pair['input'])
        output_matrix = convertir_a_matriz(pair['output'])
        pares.append({'input': input_matrix, 'output': output_matrix, 'source': 'training_challenges'})
    
    # Procesar pares de prueba
    test_inputs = []
    for pair in task['test']:
        input_matrix = convertir_a_matriz(pair['input'])
        test_inputs.append(input_matrix)
    
    # Agregar el output faltante de las soluciones al final del par correspondiente
    if task_id in solutions:
        for output in solutions[task_id]:
            output_faltante = convertir_a_matriz(output)
            if test_inputs:
                input_matrix = test_inputs.pop(0)
                pares.append({'input': input_matrix, 'output': output_faltante, 'source': 'training_solutions'})
    
    # Agregar los inputs huérfanos sin output correspondiente
    for input_matrix in test_inputs:
        pares.append({'input': input_matrix, 'source': 'training_challenges'})
    
    return pares

# Función para imprimir las matrices de manera clara
def imprimir_matrices(task_id, pares):
    for i, pair in enumerate(pares):
        if 'input' in pair:
            print(f"{task_id} pair {i+1} input matrix from {pair['source']}:\n", pair['input'])
        if 'output' in pair:
            print(f"{task_id} pair {i+1} output matrix from {pair['source']}:\n", pair['output'])

# Función para almacenar las matrices en un diccionario
def almacenar_matrices(task_id, pares, almacen):
    if task_id not in almacen:
        almacen[task_id] = {'pairs': []}
    
    for pair in pares:
        entry = {}
        if 'input' in pair:
            entry['input'] = pair['input'].tolist()
        if 'output' in pair:
            entry['output'] = pair['output'].tolist()
        entry['source'] = pair['source']
        almacen[task_id]['pairs'].append(entry)

# Diccionario para almacenar todas las matrices
almacen_matrices = {}

# Procesar todas las tareas de entrenamiento
for task_id in training_challenges:
    pares = obtener_matrices_con_output_faltante(task_id, training_challenges, training_solutions)
    almacenar_matrices(task_id, pares, almacen_matrices)

# Procesar todas las tareas de evaluación
for task_id in evaluation_challenges:
    pares = obtener_matrices_con_output_faltante(task_id, evaluation_challenges, evaluation_solutions)
    almacenar_matrices(task_id, pares, almacen_matrices)

# Imprimir las matrices almacenadas de un ejemplo específico para verificar
task_id = '007bbfb7'
imprimir_matrices(task_id, almacen_matrices[task_id]['pairs'])