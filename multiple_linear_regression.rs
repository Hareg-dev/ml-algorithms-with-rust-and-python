fn predict(x: &[Vec<f64>], w: &[f64], b: f64) -> Vec<f64> {
    x.iter()
        .map(|row| {
            let sum: f64 = row.iter().zip(w.iter()).map(|(&xi, &wi)| xi * wi).sum();
            sum + b
        })
        .collect()
}

fn compute_mse_loss(predictions: &[f64], y: &[f64]) -> f64 {
    let n = predictions.len() as f64;
    predictions
        .iter()
        .zip(y.iter())
        .map(|(&pred, &target)| (pred - target).powi(2))
        .sum::<f64>()
        / n
}

fn compute_l2_regularization(w: &[f64], lambda: f64) -> f64 {
    lambda * w.iter().map(|&wi| wi.powi(2)).sum::<f64>()
}

fn compute_gradients(
    x: &[Vec<f64>],
    y: &[f64],
    predictions: &[f64],
    w: &[f64],
    lambda: f64,
) -> (Vec<f64>, f64) {
    let n = x.len() as f64;
    
    // Gradient for weights
    let w_gradient: Vec<f64> = (0..w.len())
        .map(|j| {
            let grad: f64 = x.iter()
                .zip(predictions.iter())
                .zip(y.iter())
                .map(|((row, &pred), &yi)| (pred - yi) * row[j])
                .sum::<f64>()
                / n;
            grad + (lambda * w[j]) // L2 regularization term
        })
        .collect();
    
    // Gradient for bias
    let b_gradient: f64 = predictions
        .iter()
        .zip(y.iter())
        .map(|(&pred, &yi)| pred - yi)
        .sum::<f64>()
        / n;
    
    (w_gradient, b_gradient)
}

fn update_parameters(
    w: &mut Vec<f64>,
    b: &mut f64,
    w_gradient: &[f64],
    b_gradient: f64,
    learning_rate: f64,
) {
    for i in 0..w.len() {
        w[i] -= learning_rate * w_gradient[i];
    }
    *b -= learning_rate * b_gradient;
}

fn train(
    x: &[Vec<f64>],
    y: &[f64],
    w: &mut Vec<f64>,
    b: &mut f64,
    learning_rate: f64,
    lambda: f64,
    epochs: usize,
) {
    for epoch in 0..epochs {
        // Forward pass
        let predictions = predict(x, w, *b);
        
        // Compute loss
        let mse_loss = compute_mse_loss(&predictions, y);
        let reg_loss = compute_l2_regularization(w, lambda);
        let total_loss = mse_loss + reg_loss;
        
        // Compute gradients
        let (w_gradient, b_gradient) = compute_gradients(x, y, &predictions, w, lambda);
        
        // Update parameters
        update_parameters(w, b, &w_gradient, b_gradient, learning_rate);
        
        // Print loss every 100 epochs
        if epoch % 100 == 0 {
            println!(
                "Epoch {}: MSE Loss = {}, Reg Loss = {}, Total Loss = {}",
                epoch, mse_loss, reg_loss, total_loss
            );
        }
    }
}

fn main() {
    // Sample data: 3 samples, 2 features
    let x = vec![
        vec![1.0, 2.0],
        vec![2.0, 4.0],
        vec![3.0, 6.0],
    ];
    let y = vec![3.0, 6.0, 9.0]; // y = 1*x_1 + 1*x_2
    let mut w = vec![0.0, 0.0]; // Initialize weights
    let mut b = 0.0; // Initialize bias
    let learning_rate = 0.01;
    let lambda = 0.1; // Regularization strength
    let epochs = 1000;
    
    train(&x, &y, &mut w, &mut b, learning_rate, lambda, epochs);
    
    println!("Learned weights: {:?}", w);
    println!("Learned bias: {}", b);
    // Expected: w ≈ [1.0, 1.0], b ≈ 0.0
}