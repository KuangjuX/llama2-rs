pub fn rms_norm(x: &[f32], weitht: &[f32], output: &mut [f32], size: usize) {
    let mut sum = 0.0;
    for &i in x.iter().take(size) {
        sum += i * i;
    }

    sum /= size as f32;
    sum += 1e-5_f32;
    sum = 1.0_f32 / sum.sqrt();

    for i in 0..size {
        output[i] = x[i] * sum * weitht[i];
    }
}

pub fn softmax(x: &mut [f32], size: usize) {
    let mut max = x[0];
    for &i in x.iter().take(size) {
        if i > max {
            max = i;
        }
    }

    let mut sum = 0.0;
    for item in x.iter_mut().take(size) {
        *item = (*item - max).exp();
        sum += *item;
    }

    for item in x.iter_mut().take(size) {
        *item /= sum;
    }
}

pub fn matmul(x: &[f32], w: &[f32], o: &mut [f32], n: usize, d: usize) {
    // W (d, n) @ x (n,) -> xout (d,)
    // bu far the most amount of time is spent inside this little function
    for i in 0..d {
        let mut val = 0.0_f32;
        for j in 0..n {
            val += w[i * n + j] * x[j];
        }
        o[i] = val;
    }
}

pub fn SwiGLU(x: &mut [f32], y: &[f32], size: usize) {
    // silu(x)=x*σ(x), where σ(x) is the logistic sigmoid
    for i in 0..size {
        x[i] *= 1.0_f32 / (1.0_f32 + (-x[i]).exp());
        x[i] *= y[i];
    }
}
