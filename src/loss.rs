use crate::tensor::Tensor;

/// Cross-entropy loss with softmax (numerically stable).
///
/// Input:  logits [batch, num_classes], labels [batch] (integer class indices as f32)
/// Output: (scalar loss, gradient w.r.t. logits [batch, num_classes])
pub fn cross_entropy_loss(logits: &Tensor, labels: &[f32]) -> (f32, Tensor) {
    let batch = logits.shape[0];
    let classes = logits.shape[1];

    let probs = logits.softmax(); // [batch, classes]
    let mut loss = 0.0f32;
    let mut grad = probs.clone();

    for b in 0..batch {
        let target = labels[b] as usize;
        let p = probs.data[b * classes + target].max(1e-12);
        loss -= p.ln();
        // Gradient of CE+softmax: probs - one_hot(target)
        grad.data[b * classes + target] -= 1.0;
    }

    loss /= batch as f32;
    // Average gradient over batch
    let grad = grad.scale(1.0 / batch as f32);

    (loss, grad)
}
