use rand::Rng;

// Vector operations for math
#[derive(Debug, Clone)]
struct Vector {
    data: Vec<f64>,
}

impl Vector {
    fn new(data: Vec<f64>) -> Self {
        Vector { data }
    }

    fn dot(&self, other: &Vector) -> f64 {
        self.data.iter().zip(other.data.iter()).map(|(&a, &b)| a * b).sum()
    }

    fn norm(&self) -> f64 {
        self.dot(self).sqrt()
    }

    fn add(&self, other: &Vector) -> Vector {
        let data = self.data.iter().zip(other.data.iter()).map(|(&a, &b)| a + b).collect();
        Vector { data }
    }

    fn subtract(&self, other: &Vector) -> Vector {
        let data = self.data.iter().zip(other.data.iter()).map(|(&a, &b)| a - b).collect();
        Vector { data }
    }

    fn scale(&self, scalar: f64) -> Vector {
        let data = self.data.iter().map(|&x| x * scalar).collect();
        Vector { data }
    }
}

// Dataset structure
struct Dataset {
    features: Vec<Vector>,
    labels: Vec<f64>, // Labels are -1.0 or 1.0
}

impl Dataset {
    fn new(features: Vec<Vector>, labels: Vec<f64>) -> Self {
        Dataset { features, labels }
    }
}

// SVM model
struct SVM {
    weights: Vector,
    bias: f64,
    c: f64, // Regularization parameter
    learning_rate: f64,
    epochs: usize,
}

impl SVM {
    fn new(feature_dim: usize, c: f64, learning_rate: f64, epochs: usize) -> Self {
        let mut rng = rand::thread_rng();
        let weights = Vector::new((0..feature_dim).map(|_| rng.gen_range(-0.01..0.01)).collect());
        SVM {
            weights,
            bias: 0.0,
            c,
            learning_rate,
            epochs,
        }
    }

    fn train(&mut self, dataset: &Dataset) {
        for _ in 0..self.epochs {
            // Compute gradients
            let mut grad_w = self.weights.scale(1.0); // Regularization term: w
            let mut grad_b = 0.0;

            for (x, &y) in dataset.features.iter().zip(dataset.labels.iter()) {
                let margin = y * (self.weights.dot(x) + self.bias);
                if margin < 1.0 {
                    // Hinge loss contributes to gradient
                    grad_w = grad_w.add(&x.scale(-self.c * y));
                    grad_b -= self.c * y;
                }
            }

            // Update weights and bias
            self.weights = self.weights.subtract(&grad_w.scale(self.learning_rate));
            self.bias -= self.learning_rate * grad_b;
        }
    }

    fn predict(&self, x: &Vector) -> f64 {
        if self.weights.dot(x) + self.bias >= 0.0 {
            1.0
        } else {
            -1.0
        }
    }
}

// Example usage
fn main() {
    // Sample dataset: 2D points, linearly separable
    let features = vec![
        Vector::new(vec![2.0, 1.0]),  // Class 1
        Vector::new(vec![3.0, 2.0]),  // Class 1
        Vector::new(vec![0.0, 0.0]),  // Class -1
        Vector::new(vec![1.0, -1.0]), // Class -1
    ];
    let labels = vec![1.0, 1.0, -1.0, -1.0];
    let dataset = Dataset::new(features, labels);

    // Initialize and train SVM
    let mut svm = SVM::new(2, 1.0, 0.01, 1000);
    svm.train(&dataset);

    // Test predictions
    let test_points = vec![
        Vector::new(vec![2.5, 1.5]), // Should be Class 1
        Vector::new(vec![0.5, -0.5]), // Should be Class -1
    ];

    for point in test_points.iter() {
        let prediction = svm.predict(point);
        println!("Point {:?} -> Predicted class: {}", point.data, prediction);
    }
}