
// Red neuronal 4x10x3 con retropropagación - basada en retrop-Red4x10x3.pdf

// Función de activación sigmoide
function y = sigmoid(x)
    y = 1 ./ (1 + exp(-x));
endfunction

// Derivada de la sigmoide
function y = sigmoid_derivada(x)
    y = sigmoid(x) .* (1 - sigmoid(x));
endfunction

// Parámetros de la red
n_entradas = 4;
n_ocultas = 10;
n_salidas = 3;
n_num_dat_ent = 30;
tasa_aprendizaje = 0.1;
epocas = 1000;

// Datos de entrenamiento aleatorios para prueba
X = rand(n_num_dat_ent, n_entradas);
Y = rand(n_num_dat_ent, n_salidas);

// Inicialización de pesos y sesgos
W1 = rand(n_entradas, n_ocultas);
b1 = rand(1, n_ocultas);
W2 = rand(n_ocultas, n_salidas);
b2 = rand(1, n_salidas);

// Entrenamiento
for epoca = 1:epocas

    // Propagación hacia adelante
    b1_expanded = repmat(b1, n_num_dat_ent, 1);
    Z1 = X * W1 + b1_expanded;
    A1 = sigmoid(Z1);

    b2_expanded = repmat(b2, n_num_dat_ent, 1);
    Z2 = A1 * W2 + b2_expanded;
    A2 = sigmoid(Z2);

    // Calcular error
    error = Y - A2;

    // Retropropagación
    dZ2 = error .* sigmoid_derivada(Z2);
    dW2 = A1' * dZ2;
    db2 = sum(dZ2, 1);

    dZ1 = (dZ2 * W2') .* sigmoid_derivada(Z1);
    dW1 = X' * dZ1;
    db1 = sum(dZ1, 1);

    // Actualización de pesos
    W2 = W2 + tasa_aprendizaje * dW2;
    b2 = b2 + tasa_aprendizaje * db2;
    W1 = W1 + tasa_aprendizaje * dW1;
    b1 = b1 + tasa_aprendizaje * db1;

end

// Mostrar salidas finales
disp("Salidas de la red:");
disp(A2);
