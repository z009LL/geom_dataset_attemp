using Lux
using Zygote
using Optimisers
using Random
using CUDA  # Asegurarse de que CUDA esté instalado y configurado
using Printf  # Importar Printf para usar @printf

# Definir la estructura y constructor del LSTMCell
struct MyLSTMCell{L, C} <: Lux.AbstractExplicitContainerLayer{(:lstm, :classifier)}
    lstm::L
    classifier::C
end

function MyLSTMCell(in_dims, hidden_dims, out_dims)
    lstm_layer = Lux.LSTMCell(in_dims => hidden_dims)
    classifier_layer = Lux.Dense(hidden_dims => out_dims, sigmoid)
    return MyLSTMCell(lstm_layer, classifier_layer)
end

# Definir la función apply para MyLSTMCell
function Lux.apply(cell::MyLSTMCell, ps, st, x)
    x, st_lstm = Lux.apply(cell.lstm, ps.lstm, st.lstm, x)
    y, st_classifier = Lux.apply(cell.classifier, ps.classifier, st.classifier, x)
    return y, (lstm = st_lstm, classifier = st_classifier)
end

# Definir la función de pérdida
function compute_loss(model, ps, st, data)
    x, y = data
    y_pred, st = Lux.apply(model, ps, st, x)
    loss = sum((y .- y_pred).^2) / length(y)  # Usar MSE
    return loss, st, (y_pred = y_pred)
end

# Definir la función de precisión (accuracy)
function accuracy(y_pred, y)
    return sum((y_pred .> 0.5) .== y) / length(y)
end

# Generar datos aleatorios para prueba
function generate_data(n_samples, input_dim)
    X = rand(Float32, n_samples, input_dim)
    y = rand(Bool, n_samples)
    return [(X[i, :], y[i]) for i in 1:n_samples]
end

# Main
function main(model_type)
    # Crear el modelo
    model = model_type(2, 8, 1)

    # Inicializar el modelo
    rng = Xoshiro(0)
    dev = CUDA.functional() ? gpu : cpu
    train_state = Lux.Experimental.TrainState(
        rng, model, Adam(0.01f0); transform_variables=dev
    )

    # Generar datos aleatorios para entrenamiento y validación
    train_loader = generate_data(100, 2)
    val_loader = generate_data(20, 2)

    # Entrenar el modelo
    for epoch in 1:25
        for (x, y) in train_loader
            x = x |> dev
            y = y |> dev
            gs, loss, _, train_state = Lux.Experimental.compute_gradients(
                AutoZygote(), compute_loss, (x, y), train_state
            )
            train_state = Lux.Experimental.apply_gradients(train_state, gs)
            @printf "Epoch [%3d]: Loss %4.5f\n" epoch loss
        end
    end

    # Evaluar el modelo
    st_ = Lux.testmode(train_state.states)
    for (x, y) in val_loader
        x = x |> dev
        y = y |> dev
        loss, st_, ret = compute_loss(model, train_state.parameters, st_, (x, y))
        acc = accuracy(ret.y_pred, y)
        @printf "Validation: Loss %4.5f Accuracy %4.5f\n" loss acc
    end

    return (train_state.parameters, train_state.states) |> cpu
end

# Ejecutar el main con MyLSTMCell como el tipo de modelo
ps_trained, st_trained = main(MyLSTMCell)


