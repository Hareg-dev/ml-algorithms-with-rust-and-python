use std::collections::HashMap;

#[derive(Debug)]
struct Node {
    leaf: bool,
    class: Option<usize>,
    feature: Option<usize>,
    threshold: Option<f64>,
    left: Option<Box<Node>>,
    right: Option<Box<Node>>,
}

struct DecisionTree {
    max_depth: usize,
    min_samples_split: usize,
    tree: Option<Node>,
}

impl DecisionTree {
    fn new(max_depth: usize, min_samples_split: usize) -> Self {
        DecisionTree {
            max_depth,
            min_samples_split,
            tree: None,
        }
    }

    fn gini_impurity(&self, y: &[usize]) -> f64 {
        let mut counts = HashMap::new();
        for &label in y {
            *counts.entry(label).or_insert(0) += 1;
        }
        let n = y.len() as f64;
        1.0 - counts.values().map(|&c| (c as f64 / n).powi(2)).sum::<f64>()
    }

    fn find_best_split(&self, x: &[Vec<f64>], y: &[usize]) -> (Option<usize>, Option<f64>, f64) {
        let n_samples = x.len();
        let n_features = x[0].len();
        let mut best_gain = -1.0;
        let mut best_feature = None;
        let mut best_threshold = None;

        let parent_impurity = self.gini_impurity(y);

        for feature in 0..n_features {
            let mut thresholds: Vec<f64> = x.iter().map(|row| row[feature]).collect();
            thresholds.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let thresholds: Vec<f64> = thresholds.into_iter().collect::<Vec<_>>();

            for &threshold in &thresholds {
                let mut left_y = Vec::new();
                let mut right_y = Vec::new();

                for i in 0..n_samples {
                    if x[i][feature] <= threshold {
                        left_y.push(y[i]);
                    } else {
                        right_y.push(y[i]);
                    }
                }

                if left_y.len() < self.min_samples_split || right_y.len() < self.min_samples_split {
                    continue;
                }

                let left_impurity = self.gini_impurity(&left_y);
                let right_impurity = self.gini_impurity(&right_y);

                let weighted_impurity = (left_y.len() as f64 / n_samples as f64) * left_impurity +
                                       (right_y.len() as f64 / n_samples as f64) * right_impurity;
                let gain = parent_impurity - weighted_impurity;

                if gain > best_gain {
                    best_gain = gain;
                    best_feature = Some(feature);
                    best_threshold = Some(threshold);
                }
            }
        }

        (best_feature, best_threshold, best_gain)
    }

    fn fit(&mut self, x: Vec<Vec<f64>>, y: Vec<usize>, depth: usize) -> Node {
        let n_samples = y.len();
        let n_classes = y.iter().collect::<std::collections::HashSet<_>>().len();

        if depth >= self.max_depth || n_samples < self.min_samples_split || n_classes == 1 {
            let mut counts = HashMap::new();
            for &label in &y {
                *counts.entry(label).or_insert(0) += 1;
            }
            let class = counts.into_iter().max_by_key(|&(_, count)| count).map(|(label, _)| label);
            return Node {
                leaf: true,
                class,
                feature: None,
                threshold: None,
                left: None,
                right: None,
            };
        }

        let (feature, threshold, gain) = self.find_best_split(&x, &y);
        if feature.is_none() || gain <= 0.0 {
            let mut counts = HashMap::new();
            for &label in &y {
                *counts.entry(label).or_insert(0) += 1;
            }
            let class = counts.into_iter().max_by_key(|&(_, count)| count).map(|(label, _)| label);
            return Node {
                leaf: true,
                class,
                feature: None,
                threshold: None,
                left: None,
                right: None,
            };
        }

        let feature = feature.unwrap();
        let threshold = threshold.unwrap();
        let mut left_x = Vec::new();
        let mut left_y = Vec::new();
        let mut right_x = Vec::new();
        let mut right_y = Vec::new();

        for i in 0..n_samples {
            if x[i][feature] <= threshold {
                left_x.push(x[i].clone());
                left_y.push(y[i]);
            } else {
                right_x.push(x[i].clone());
                right_y.push(y[i]);
            }
        }

        Node {
            leaf: false,
            class: None,
            feature: Some(feature),
            threshold: Some(threshold),
            left: Some(Box::new(self.fit(left_x, left_y, depth + 1))),
            right: Some(Box::new(self.fit(right_x, right_y, depth + 1))),
        }
    }

    fn predict(&self, x: &[Vec<f64>]) -> Vec<usize> {
        let mut predictions = Vec::new();
        for sample in x {
            let mut node = self.tree.as_ref().unwrap();
            while !node.leaf {
                if sample[node.feature.unwrap()] <= node.threshold.unwrap() {
                    node = node.left.as_ref().unwrap();
                } else {
                    node = node.right.as_ref().unwrap();
                }
            }
            predictions.push(node.class.unwrap());
        }
        predictions
    }
}

fn main() {
    let x = vec![
        vec![1.0, 2.0],
        vec![2.0, 3.0],
        vec![3.0, 1.0],
        vec![4.0, 4.0],
    ];
    let y = vec![0, 0, 1, 1];
    let mut dt = DecisionTree::new(3, 2);
    dt.tree = Some(dt.fit(x.clone(), y, 0));
    let predictions = dt.predict(&x);
    println!("{:?}", predictions);
}