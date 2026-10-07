//! CUDA PTX Kernels for TARA Neural Engine.
//!
//! Handcrafted, release-grade PTX assembly targeting Maxwell architecture (sm_50)
//! and upward (Pascal, Volta, Turing, Ampere, Ada, Hopper).
//! Pure CUDA execution with zero external compilation dependencies.

pub const PTX_TARA_KERNELS: &str = r#"
.version 6.5
.target sm_50
.address_size 64

// 1. In-place Gradient Accumulation: accum[i] += grad[i]
.visible .entry accum_grads_kernel(
    .param .u64 p_accum,
    .param .u64 p_grad,
    .param .u32 p_n
) {
    .reg .pred %p;
    .reg .b32 %r<5>;
    .reg .b64 %rd<5>;
    .reg .f32 %f<3>;

    mov.u32 %r0, %ctaid.x;
    mov.u32 %r1, %ntid.x;
    mov.u32 %r2, %tid.x;
    mad.lo.s32 %r3, %r0, %r1, %r2;

    ld.param.u32 %r4, [p_n];
    setp.ge.u32 %p, %r3, %r4;
    @%p bra $ACCUM_DONE;

    mul.wide.u32 %rd0, %r3, 4;
    ld.param.u64 %rd1, [p_accum];
    ld.param.u64 %rd2, [p_grad];
    add.u64 %rd3, %rd1, %rd0;
    add.u64 %rd4, %rd2, %rd0;

    ld.global.f32 %f0, [%rd3];
    ld.global.f32 %f1, [%rd4];
    add.f32 %f2, %f0, %f1;
    st.global.f32 [%rd3], %f2;

$ACCUM_DONE:
    ret;
}

// 2. In-place Gradient Scaling: grad[i] *= scale
.visible .entry scale_grads_kernel(
    .param .u64 p_grad,
    .param .f32 p_scale,
    .param .u32 p_n
) {
    .reg .pred %p;
    .reg .b32 %r<5>;
    .reg .b64 %rd<3>;
    .reg .f32 %f<3>;

    mov.u32 %r0, %ctaid.x;
    mov.u32 %r1, %ntid.x;
    mov.u32 %r2, %tid.x;
    mad.lo.s32 %r3, %r0, %r1, %r2;

    ld.param.u32 %r4, [p_n];
    setp.ge.u32 %p, %r3, %r4;
    @%p bra $SCALE_DONE;

    mul.wide.u32 %rd0, %r3, 4;
    ld.param.u64 %rd1, [p_grad];
    add.u64 %rd2, %rd1, %rd0;

    ld.global.f32 %f0, [%rd2];
    ld.param.f32 %f1, [p_scale];
    mul.f32 %f2, %f0, %f1;
    st.global.f32 [%rd2], %f2;

$SCALE_DONE:
    ret;
}

// 3. AdamW Optimizer Step with Gradient Clipping and Weight Decay
.visible .entry adamw_step_kernel(
    .param .u64 p_weight,
    .param .u64 p_grad,
    .param .u64 p_m,
    .param .u64 p_v,
    .param .f32 p_lr,
    .param .f32 p_beta1,
    .param .f32 p_beta2,
    .param .f32 p_eps,
    .param .f32 p_weight_decay,
    .param .f32 p_clip_scale,
    .param .f32 p_bc1,
    .param .f32 p_bc2,
    .param .u32 p_n
) {
    .reg .pred %p;
    .reg .b32 %r<5>;
    .reg .b64 %rd<9>;
    .reg .f32 %f<15>;

    mov.u32 %r0, %ctaid.x;
    mov.u32 %r1, %ntid.x;
    mov.u32 %r2, %tid.x;
    mad.lo.s32 %r3, %r0, %r1, %r2;

    ld.param.u32 %r4, [p_n];
    setp.ge.u32 %p, %r3, %r4;
    @%p bra $ADAMW_DONE;

    mul.wide.u32 %rd0, %r3, 4;

    ld.param.u64 %rd1, [p_weight];
    ld.param.u64 %rd2, [p_grad];
    ld.param.u64 %rd3, [p_m];
    ld.param.u64 %rd4, [p_v];

    add.u64 %rd5, %rd1, %rd0;
    add.u64 %rd6, %rd2, %rd0;
    add.u64 %rd7, %rd3, %rd0;
    add.u64 %rd8, %rd4, %rd0;

    ld.global.f32 %f0, [%rd5]; // weight
    ld.global.f32 %f1, [%rd6]; // grad
    ld.global.f32 %f2, [%rd7]; // m
    ld.global.f32 %f3, [%rd8]; // v

    ld.param.f32 %f4, [p_lr];
    ld.param.f32 %f5, [p_beta1];
    ld.param.f32 %f6, [p_beta2];
    ld.param.f32 %f7, [p_eps];
    ld.param.f32 %f8, [p_weight_decay];
    ld.param.f32 %f9, [p_clip_scale];
    ld.param.f32 %f10, [p_bc1];
    ld.param.f32 %f11, [p_bc2];

    // g = grad * clip_scale (decoupled from weight decay)
    mul.f32 %f1, %f1, %f9; // f1 is g

    // m = beta1 * m + (1.0 - beta1) * g
    sub.f32 %f13, 1.0, %f5;
    mul.f32 %f2, %f2, %f5;
    fma.rn.f32 %f2, %f13, %f1, %f2;

    // v = beta2 * v + (1.0 - beta2) * g * g
    sub.f32 %f14, 1.0, %f6;
    mul.f32 %f3, %f3, %f6;
    mul.f32 %f12, %f1, %f1;
    fma.rn.f32 %f3, %f14, %f12, %f3;

    // m_hat = m / bc1; v_hat = v / bc2
    div.approx.f32 %f13, %f2, %f10;
    div.approx.f32 %f14, %f3, %f11;

    // denom = sqrt(v_hat) + eps
    sqrt.approx.f32 %f12, %f14;
    add.f32 %f12, %f12, %f7;

    // delta = lr * m_hat / denom
    mul.f32 %f13, %f4, %f13;
    div.approx.f32 %f13, %f13, %f12;

    // Decoupled weight decay: weight -= lr * weight_decay * weight
    mul.f32 %f12, %f4, %f8;
    mul.f32 %f12, %f12, %f0;
    sub.f32 %f0, %f0, %f12;

    // weight -= delta
    sub.f32 %f0, %f0, %f13;

    st.global.f32 [%rd5], %f0;
    st.global.f32 [%rd7], %f2;
    st.global.f32 [%rd8], %f3;

$ADAMW_DONE:
    ret;
}

// 4. LM Head Forward Projection: logits[t, v] = dot(hidden[t, :], lm_head[v, :])
.visible .entry lm_head_fwd_kernel(
    .param .u64 p_hidden,
    .param .u64 p_lm_head,
    .param .u64 p_logits,
    .param .u32 p_seq_len,
    .param .u32 p_hs,
    .param .u32 p_vs
) {
    .reg .pred %p;
    .reg .b32 %r<10>;
    .reg .b64 %rd<10>;
    .reg .f32 %f<5>;

    mov.u32 %r0, %ctaid.x;
    mov.u32 %r1, %ntid.x;
    mov.u32 %r2, %tid.x;
    mad.lo.s32 %r3, %r0, %r1, %r2;

    ld.param.u32 %r4, [p_seq_len];
    ld.param.u32 %r5, [p_vs];
    mul.lo.u32 %r6, %r4, %r5;

    setp.ge.u32 %p, %r3, %r6;
    @%p bra $LM_FWD_DONE;

    div.u32 %r7, %r3, %r5; // t
    rem.u32 %r8, %r3, %r5; // v

    ld.param.u32 %r9, [p_hs];
    ld.param.u64 %rd1, [p_hidden];
    ld.param.u64 %rd2, [p_lm_head];

    mul.lo.u32 %r0, %r7, %r9;
    mul.wide.u32 %rd3, %r0, 4;
    add.u64 %rd4, %rd1, %rd3;

    mul.lo.u32 %r1, %r8, %r9;
    mul.wide.u32 %rd5, %r1, 4;
    add.u64 %rd6, %rd2, %rd5;

    mov.f32 %f0, 0.0;
    mov.u32 %r2, 0;

$LM_FWD_LOOP:
    setp.ge.u32 %p, %r2, %r9;
    @%p bra $LM_FWD_STORE;

    mul.wide.u32 %rd7, %r2, 4;
    add.u64 %rd8, %rd4, %rd7;
    add.u64 %rd9, %rd6, %rd7;
    ld.global.f32 %f1, [%rd8];
    ld.global.f32 %f2, [%rd9];
    fma.rn.f32 %f0, %f1, %f2, %f0;

    add.u32 %r2, %r2, 1;
    bra $LM_FWD_LOOP;

$LM_FWD_STORE:
    ld.param.u64 %rd1, [p_logits];
    mul.wide.u32 %rd3, %r3, 4;
    add.u64 %rd4, %rd1, %rd3;
    st.global.f32 [%rd4], %f0;

$LM_FWD_DONE:
    ret;
}

// 5. LM Head Backward Weight Gradient: d_lm_head[v, h] = sum_t (d_logits[t, v] * final_normed[t, h])
.visible .entry lm_head_bwd_weight_kernel(
    .param .u64 p_d_logits,
    .param .u64 p_final_normed,
    .param .u64 p_d_lm_head,
    .param .u32 p_seq_len,
    .param .u32 p_hs,
    .param .u32 p_vs
) {
    .reg .pred %p;
    .reg .b32 %r<10>;
    .reg .b64 %rd<10>;
    .reg .f32 %f<5>;

    mov.u32 %r0, %ctaid.x;
    mov.u32 %r1, %ntid.x;
    mov.u32 %r2, %tid.x;
    mad.lo.s32 %r3, %r0, %r1, %r2;

    ld.param.u32 %r4, [p_vs];
    ld.param.u32 %r5, [p_hs];
    mul.lo.u32 %r6, %r4, %r5;

    setp.ge.u32 %p, %r3, %r6;
    @%p bra $BWD_W_DONE;

    div.u32 %r7, %r3, %r5; // v
    rem.u32 %r8, %r3, %r5; // h

    ld.param.u32 %r9, [p_seq_len];
    ld.param.u64 %rd1, [p_d_logits];
    ld.param.u64 %rd2, [p_final_normed];

    mov.f32 %f0, 0.0;
    mov.u32 %r2, 0;

$BWD_W_LOOP:
    setp.ge.u32 %p, %r2, %r9;
    @%p bra $BWD_W_STORE;

    mul.lo.u32 %r0, %r2, %r4;
    add.u32 %r0, %r0, %r7;
    mul.wide.u32 %rd3, %r0, 4;
    add.u64 %rd4, %rd1, %rd3;
    ld.global.f32 %f1, [%rd4];

    mul.lo.u32 %r1, %r2, %r5;
    add.u32 %r1, %r1, %r8;
    mul.wide.u32 %rd5, %r1, 4;
    add.u64 %rd6, %rd2, %rd5;
    ld.global.f32 %f2, [%rd6];

    fma.rn.f32 %f0, %f1, %f2, %f0;

    add.u32 %r2, %r2, 1;
    bra $BWD_W_LOOP;

$BWD_W_STORE:
    ld.param.u64 %rd1, [p_d_lm_head];
    mul.wide.u32 %rd3, %r3, 4;
    add.u64 %rd4, %rd1, %rd3;
    st.global.f32 [%rd4], %f0;

$BWD_W_DONE:
    ret;
}

// 6. LM Head Backward Input Gradient: d_final_normed[t, h] = sum_v (d_logits[t, v] * lm_head[v, h])
.visible .entry lm_head_bwd_input_kernel(
    .param .u64 p_d_logits,
    .param .u64 p_lm_head,
    .param .u64 p_d_final_normed,
    .param .u32 p_seq_len,
    .param .u32 p_hs,
    .param .u32 p_vs
) {
    .reg .pred %p;
    .reg .b32 %r<10>;
    .reg .b64 %rd<10>;
    .reg .f32 %f<5>;

    mov.u32 %r0, %ctaid.x;
    mov.u32 %r1, %ntid.x;
    mov.u32 %r2, %tid.x;
    mad.lo.s32 %r3, %r0, %r1, %r2;

    ld.param.u32 %r4, [p_seq_len];
    ld.param.u32 %r5, [p_hs];
    mul.lo.u32 %r6, %r4, %r5;

    setp.ge.u32 %p, %r3, %r6;
    @%p bra $BWD_IN_DONE;

    div.u32 %r7, %r3, %r5; // t
    rem.u32 %r8, %r3, %r5; // h

    ld.param.u32 %r9, [p_vs];
    ld.param.u64 %rd1, [p_d_logits];
    ld.param.u64 %rd2, [p_lm_head];

    mov.f32 %f0, 0.0;
    mov.u32 %r2, 0;

$BWD_IN_LOOP:
    setp.ge.u32 %p, %r2, %r9;
    @%p bra $BWD_IN_STORE;

    mul.lo.u32 %r0, %r7, %r9;
    add.u32 %r0, %r0, %r2;
    mul.wide.u32 %rd3, %r0, 4;
    add.u64 %rd4, %rd1, %rd3;
    ld.global.f32 %f1, [%rd4];

    mul.lo.u32 %r1, %r2, %r5;
    add.u32 %r1, %r1, %r8;
    mul.wide.u32 %rd5, %r1, 4;
    add.u64 %rd6, %rd2, %rd5;
    ld.global.f32 %f2, [%rd6];

    fma.rn.f32 %f0, %f1, %f2, %f0;

    add.u32 %r2, %r2, 1;
    bra $BWD_IN_LOOP;

$BWD_IN_STORE:
    ld.param.u64 %rd1, [p_d_final_normed];
    mul.wide.u32 %rd3, %r3, 4;
    add.u64 %rd4, %rd1, %rd3;
    st.global.f32 [%rd4], %f0;

$BWD_IN_DONE:
    ret;
}

// 7. FP16 LM Head Forward Projection: logits[t, v] = dot(hidden[t, :], lm_head[v, :])
// p_hidden: .b16, p_lm_head: .b16, p_logits: .f32
.visible .entry lm_head_fwd_f16_kernel(
    .param .u64 p_hidden,
    .param .u64 p_lm_head,
    .param .u64 p_logits,
    .param .u32 p_seq_len,
    .param .u32 p_hs,
    .param .u32 p_vs
) {
    .reg .pred %p;
    .reg .b16 %h<3>;
    .reg .b32 %r<10>;
    .reg .b64 %rd<10>;
    .reg .f32 %f<5>;

    mov.u32 %r0, %ctaid.x;
    mov.u32 %r1, %ntid.x;
    mov.u32 %r2, %tid.x;
    mad.lo.s32 %r3, %r0, %r1, %r2;

    ld.param.u32 %r4, [p_seq_len];
    ld.param.u32 %r5, [p_vs];
    mul.lo.u32 %r6, %r4, %r5;

    setp.ge.u32 %p, %r3, %r6;
    @%p bra $LM_FWD_F16_DONE;

    div.u32 %r7, %r3, %r5;
    rem.u32 %r8, %r3, %r5;

    ld.param.u32 %r9, [p_hs];
    ld.param.u64 %rd1, [p_hidden];
    ld.param.u64 %rd2, [p_lm_head];

    mul.lo.u32 %r0, %r7, %r9;
    mul.wide.u32 %rd3, %r0, 2;
    add.u64 %rd4, %rd1, %rd3;

    mul.lo.u32 %r1, %r8, %r9;
    mul.wide.u32 %rd5, %r1, 2;
    add.u64 %rd6, %rd2, %rd5;

    mov.f32 %f0, 0.0;
    mov.u32 %r2, 0;

$LM_FWD_F16_LOOP:
    setp.ge.u32 %p, %r2, %r9;
    @%p bra $LM_FWD_F16_STORE;

    mul.wide.u32 %rd7, %r2, 2;
    add.u64 %rd8, %rd4, %rd7;
    add.u64 %rd9, %rd6, %rd7;
    ld.global.b16 %h1, [%rd8];
    ld.global.b16 %h2, [%rd9];
    cvt.f32.f16 %f1, %h1;
    cvt.f32.f16 %f2, %h2;
    fma.rn.f32 %f0, %f1, %f2, %f0;

    add.u32 %r2, %r2, 1;
    bra $LM_FWD_F16_LOOP;

$LM_FWD_F16_STORE:
    ld.param.u64 %rd1, [p_logits];
    mul.wide.u32 %rd3, %r3, 4;
    add.u64 %rd4, %rd1, %rd3;
    st.global.f32 [%rd4], %f0;

$LM_FWD_F16_DONE:
    ret;
}

// 8. FP16 LM Head Backward Weight Gradient: d_lm_head[v, h] = sum_t (d_logits[t, v] * final_normed[t, h])
// p_d_logits: .b16, p_final_normed: .b16, p_d_lm_head: .b16
.visible .entry lm_head_bwd_weight_f16_kernel(
    .param .u64 p_d_logits,
    .param .u64 p_final_normed,
    .param .u64 p_d_lm_head,
    .param .u32 p_seq_len,
    .param .u32 p_hs,
    .param .u32 p_vs
) {
    .reg .pred %p;
    .reg .b16 %h<3>;
    .reg .b32 %r<10>;
    .reg .b64 %rd<10>;
    .reg .f32 %f<5>;

    mov.u32 %r0, %ctaid.x;
    mov.u32 %r1, %ntid.x;
    mov.u32 %r2, %tid.x;
    mad.lo.s32 %r3, %r0, %r1, %r2;

    ld.param.u32 %r4, [p_vs];
    ld.param.u32 %r5, [p_hs];
    mul.lo.u32 %r6, %r4, %r5;

    setp.ge.u32 %p, %r3, %r6;
    @%p bra $BWD_W_F16_DONE;

    div.u32 %r7, %r3, %r5;
    rem.u32 %r8, %r3, %r5;

    ld.param.u32 %r9, [p_seq_len];
    ld.param.u64 %rd1, [p_d_logits];
    ld.param.u64 %rd2, [p_final_normed];

    mov.f32 %f0, 0.0;
    mov.u32 %r2, 0;

$BWD_W_F16_LOOP:
    setp.ge.u32 %p, %r2, %r9;
    @%p bra $BWD_W_F16_STORE;

    mul.lo.u32 %r0, %r2, %r4;
    add.u32 %r0, %r0, %r7;
    mul.wide.u32 %rd3, %r0, 2;
    add.u64 %rd4, %rd1, %rd3;
    ld.global.b16 %h1, [%rd4];
    cvt.f32.f16 %f1, %h1;

    mul.lo.u32 %r1, %r2, %r5;
    add.u32 %r1, %r1, %r8;
    mul.wide.u32 %rd5, %r1, 2;
    add.u64 %rd6, %rd2, %rd5;
    ld.global.b16 %h2, [%rd6];
    cvt.f32.f16 %f2, %h2;

    fma.rn.f32 %f0, %f1, %f2, %f0;

    add.u32 %r2, %r2, 1;
    bra $BWD_W_F16_LOOP;

$BWD_W_F16_STORE:
    ld.param.u64 %rd1, [p_d_lm_head];
    mul.wide.u32 %rd3, %r3, 2;
    add.u64 %rd4, %rd1, %rd3;
    cvt.rn.f16.f32 %h0, %f0;
    st.global.b16 [%rd4], %h0;

$BWD_W_F16_DONE:
    ret;
}

// 9. FP16 LM Head Backward Input Gradient: d_final_normed[t, h] = sum_v (d_logits[t, v] * lm_head[v, h])
// p_d_logits: .b16, p_lm_head: .b16, p_d_final_normed: .b16
.visible .entry lm_head_bwd_input_f16_kernel(
    .param .u64 p_d_logits,
    .param .u64 p_lm_head,
    .param .u64 p_d_final_normed,
    .param .u32 p_seq_len,
    .param .u32 p_hs,
    .param .u32 p_vs
) {
    .reg .pred %p;
    .reg .b16 %h<3>;
    .reg .b32 %r<10>;
    .reg .b64 %rd<10>;
    .reg .f32 %f<5>;

    mov.u32 %r0, %ctaid.x;
    mov.u32 %r1, %ntid.x;
    mov.u32 %r2, %tid.x;
    mad.lo.s32 %r3, %r0, %r1, %r2;

    ld.param.u32 %r4, [p_seq_len];
    ld.param.u32 %r5, [p_hs];
    mul.lo.u32 %r6, %r4, %r5;

    setp.ge.u32 %p, %r3, %r6;
    @%p bra $BWD_IN_F16_DONE;

    div.u32 %r7, %r3, %r5;
    rem.u32 %r8, %r3, %r5;

    ld.param.u32 %r9, [p_vs];
    ld.param.u64 %rd1, [p_d_logits];
    ld.param.u64 %rd2, [p_lm_head];

    mov.f32 %f0, 0.0;
    mov.u32 %r2, 0;

$BWD_IN_F16_LOOP:
    setp.ge.u32 %p, %r2, %r9;
    @%p bra $BWD_IN_F16_STORE;

    mul.lo.u32 %r0, %r7, %r9;
    add.u32 %r0, %r0, %r2;
    mul.wide.u32 %rd3, %r0, 2;
    add.u64 %rd4, %rd1, %rd3;
    ld.global.b16 %h1, [%rd4];
    cvt.f32.f16 %f1, %h1;

    mul.lo.u32 %r1, %r2, %r5;
    add.u32 %r1, %r1, %r8;
    mul.wide.u32 %rd5, %r1, 2;
    add.u64 %rd6, %rd2, %rd5;
    ld.global.b16 %h2, [%rd6];
    cvt.f32.f16 %f2, %h2;

    fma.rn.f32 %f0, %f1, %f2, %f0;

    add.u32 %r2, %r2, 1;
    bra $BWD_IN_F16_LOOP;

$BWD_IN_F16_STORE:
    ld.param.u64 %rd1, [p_d_final_normed];
    mul.wide.u32 %rd3, %r3, 2;
    add.u64 %rd4, %rd1, %rd3;
    cvt.rn.f16.f32 %h0, %f0;
    st.global.b16 [%rd4], %h0;

$BWD_IN_F16_DONE:
    ret;
}

// 10. FP16 in-place gradient accumulation: accum[i] += grad[i]
.visible .entry accum_grads_f16_kernel(
    .param .u64 p_accum,
    .param .u64 p_grad,
    .param .u32 p_n
) {
    .reg .pred %p;
    .reg .b16 %h<3>;
    .reg .b32 %r<5>;
    .reg .b64 %rd<5>;
    .reg .f32 %f<3>;

    mov.u32 %r0, %ctaid.x;
    mov.u32 %r1, %ntid.x;
    mov.u32 %r2, %tid.x;
    mad.lo.s32 %r3, %r0, %r1, %r2;

    ld.param.u32 %r4, [p_n];
    setp.ge.u32 %p, %r3, %r4;
    @%p bra $ACCUM_F16_DONE;

    mul.wide.u32 %rd0, %r3, 2;
    ld.param.u64 %rd1, [p_accum];
    ld.param.u64 %rd2, [p_grad];
    add.u64 %rd3, %rd1, %rd0;
    add.u64 %rd4, %rd2, %rd0;

    ld.global.b16 %h0, [%rd3];
    ld.global.b16 %h1, [%rd4];
    cvt.f32.f16 %f0, %h0;
    cvt.f32.f16 %f1, %h1;
    add.f32 %f2, %f0, %f1;
    cvt.rn.f16.f32 %h2, %f2;
    st.global.b16 [%rd3], %h2;

$ACCUM_F16_DONE:
    ret;
}

// 11. FP16 in-place gradient scaling: grad[i] *= scale
.visible .entry scale_grads_f16_kernel(
    .param .u64 p_grad,
    .param .f32 p_scale,
    .param .u32 p_n
) {
    .reg .pred %p;
    .reg .b16 %h<2>;
    .reg .b32 %r<5>;
    .reg .b64 %rd<3>;
    .reg .f32 %f<3>;

    mov.u32 %r0, %ctaid.x;
    mov.u32 %r1, %ntid.x;
    mov.u32 %r2, %tid.x;
    mad.lo.s32 %r3, %r0, %r1, %r2;

    ld.param.u32 %r4, [p_n];
    setp.ge.u32 %p, %r3, %r4;
    @%p bra $SCALE_F16_DONE;

    mul.wide.u32 %rd0, %r3, 2;
    ld.param.u64 %rd1, [p_grad];
    add.u64 %rd2, %rd1, %rd0;

    ld.global.b16 %h0, [%rd2];
    cvt.f32.f16 %f0, %h0;
    ld.param.f32 %f1, [p_scale];
    mul.f32 %f2, %f0, %f1;
    cvt.rn.f16.f32 %h1, %f2;
    st.global.b16 [%rd2], %h1;

$SCALE_F16_DONE:
    ret;
}

// 12. Mixed-Precision AdamW Optimizer Step with FP32 Master Weights and FP16 Model Weights
.visible .entry adamw_step_mixed_f16_kernel(
    .param .u64 p_master_weight,
    .param .u64 p_model_weight_f16,
    .param .u64 p_grad_f16,
    .param .u64 p_m,
    .param .u64 p_v,
    .param .f32 p_lr,
    .param .f32 p_beta1,
    .param .f32 p_beta2,
    .param .f32 p_eps,
    .param .f32 p_weight_decay,
    .param .f32 p_clip_scale,
    .param .f32 p_bc1,
    .param .f32 p_bc2,
    .param .u32 p_n
) {
    .reg .pred %p;
    .reg .b16 %h<3>;
    .reg .b32 %r<5>;
    .reg .b64 %rd<11>;
    .reg .f32 %f<15>;

    mov.u32 %r0, %ctaid.x;
    mov.u32 %r1, %ntid.x;
    mov.u32 %r2, %tid.x;
    mad.lo.s32 %r3, %r0, %r1, %r2;

    ld.param.u32 %r4, [p_n];
    setp.ge.u32 %p, %r3, %r4;
    @%p bra $ADAMW_F16_DONE;

    mul.wide.u32 %rd0, %r3, 4;
    mul.wide.u32 %rd1, %r3, 2;

    ld.param.u64 %rd2, [p_master_weight];
    ld.param.u64 %rd3, [p_model_weight_f16];
    ld.param.u64 %rd4, [p_grad_f16];
    ld.param.u64 %rd5, [p_m];
    ld.param.u64 %rd6, [p_v];

    add.u64 %rd7, %rd2, %rd0;
    add.u64 %rd8, %rd3, %rd1;
    add.u64 %rd9, %rd4, %rd1;
    add.u64 %rd10, %rd5, %rd0;
    add.u64 %rd2, %rd6, %rd0;

    ld.global.f32 %f0, [%rd7];
    ld.global.b16 %h0, [%rd9];
    cvt.f32.f16 %f1, %h0;
    ld.global.f32 %f2, [%rd10];
    ld.global.f32 %f3, [%rd2];

    ld.param.f32 %f4, [p_lr];
    ld.param.f32 %f5, [p_beta1];
    ld.param.f32 %f6, [p_beta2];
    ld.param.f32 %f7, [p_eps];
    ld.param.f32 %f8, [p_weight_decay];
    ld.param.f32 %f9, [p_clip_scale];
    ld.param.f32 %f10, [p_bc1];
    ld.param.f32 %f11, [p_bc2];

    // g = grad * clip_scale (decoupled from weight decay)
    mul.f32 %f1, %f1, %f9;

    // m = beta1 * m + (1.0 - beta1) * g
    sub.f32 %f13, 1.0, %f5;
    mul.f32 %f2, %f2, %f5;
    fma.rn.f32 %f2, %f13, %f1, %f2;

    // v = beta2 * v + (1.0 - beta2) * g * g
    sub.f32 %f14, 1.0, %f6;
    mul.f32 %f3, %f3, %f6;
    mul.f32 %f12, %f1, %f1;
    fma.rn.f32 %f3, %f14, %f12, %f3;

    // m_hat = m / bc1; v_hat = v / bc2
    div.approx.f32 %f13, %f2, %f10;
    div.approx.f32 %f14, %f3, %f11;

    // denom = sqrt(v_hat) + eps
    sqrt.approx.f32 %f12, %f14;
    add.f32 %f12, %f12, %f7;

    // delta = lr * m_hat / denom
    mul.f32 %f13, %f4, %f13;
    div.approx.f32 %f13, %f13, %f12;

    // Decoupled weight decay on master weights: master_weight -= lr * weight_decay * master_weight
    mul.f32 %f12, %f4, %f8;
    mul.f32 %f12, %f12, %f0;
    sub.f32 %f0, %f0, %f12;

    // master_weight -= delta
    sub.f32 %f0, %f0, %f13;

    // Convert updated master weight to FP16 and store into model weight buffer
    cvt.rn.f16.f32 %h1, %f0;

    st.global.f32 [%rd7], %f0;
    st.global.b16 [%rd8], %h1;
    st.global.f32 [%rd10], %f2;
    st.global.f32 [%rd2], %f3;

    // Reset gradient buffer to zero
    mov.b16 %h2, 0;
    st.global.b16 [%rd9], %h2;

$ADAMW_F16_DONE:
    ret;
}

// ─────────────────────────────────────────────────────────────────────────────
// Full-Network Transformer Decoder Kernels (Embedding, RMSNorm, SwiGLU, Residual, RoPE, Attention)
// ─────────────────────────────────────────────────────────────────────────────

// 13. Embedding Forward: out[t * hs + h] = embed_weight[tokens[t] * hs + h]
.visible .entry embedding_fwd_kernel(
    .param .u64 p_tokens,
    .param .u64 p_embed,
    .param .u64 p_out,
    .param .u32 p_seq_len,
    .param .u32 p_hs
) {
    .reg .pred %p;
    .reg .b32 %r<10>;
    .reg .b64 %rd<10>;
    .reg .f32 %f0;

    mov.u32 %r0, %ctaid.x;
    mov.u32 %r1, %ntid.x;
    mov.u32 %r2, %tid.x;
    mad.lo.s32 %r3, %r0, %r1, %r2;

    ld.param.u32 %r4, [p_seq_len];
    ld.param.u32 %r5, [p_hs];
    mul.lo.u32 %r6, %r4, %r5;
    setp.ge.u32 %p, %r3, %r6;
    @%p bra $EMB_FWD_DONE;

    div.u32 %r7, %r3, %r5;
    rem.u32 %r8, %r3, %r5;

    ld.param.u64 %rd0, [p_tokens];
    mul.wide.u32 %rd1, %r7, 4;
    add.u64 %rd2, %rd0, %rd1;
    ld.global.u32 %r9, [%rd2];

    mul.lo.u32 %r0, %r9, %r5;
    add.u32 %r0, %r0, %r8;
    mul.wide.u32 %rd3, %r0, 4;
    ld.param.u64 %rd4, [p_embed];
    add.u64 %rd5, %rd4, %rd3;
    ld.global.f32 %f0, [%rd5];

    ld.param.u64 %rd6, [p_out];
    mul.wide.u32 %rd7, %r3, 4;
    add.u64 %rd8, %rd6, %rd7;
    st.global.f32 [%rd8], %f0;

$EMB_FWD_DONE:
    ret;
}

// 14. Embedding Backward: d_embed[tokens[t] * hs + h] += d_hidden[t * hs + h]
.visible .entry embedding_bwd_kernel(
    .param .u64 p_tokens,
    .param .u64 p_d_hidden,
    .param .u64 p_d_embed,
    .param .u32 p_seq_len,
    .param .u32 p_hs
) {
    .reg .pred %p;
    .reg .b32 %r<10>;
    .reg .b64 %rd<10>;
    .reg .f32 %f0;

    mov.u32 %r0, %ctaid.x;
    mov.u32 %r1, %ntid.x;
    mov.u32 %r2, %tid.x;
    mad.lo.s32 %r3, %r0, %r1, %r2;

    ld.param.u32 %r4, [p_seq_len];
    ld.param.u32 %r5, [p_hs];
    mul.lo.u32 %r6, %r4, %r5;
    setp.ge.u32 %p, %r3, %r6;
    @%p bra $EMB_BWD_DONE;

    div.u32 %r7, %r3, %r5;
    rem.u32 %r8, %r3, %r5;

    ld.param.u64 %rd0, [p_tokens];
    mul.wide.u32 %rd1, %r7, 4;
    add.u64 %rd2, %rd0, %rd1;
    ld.global.u32 %r9, [%rd2];

    ld.param.u64 %rd3, [p_d_hidden];
    mul.wide.u32 %rd4, %r3, 4;
    add.u64 %rd5, %rd3, %rd4;
    ld.global.f32 %f0, [%rd5];

    mul.lo.u32 %r0, %r9, %r5;
    add.u32 %r0, %r0, %r8;
    mul.wide.u32 %rd6, %r0, 4;
    ld.param.u64 %rd7, [p_d_embed];
    add.u64 %rd8, %rd7, %rd6;
    atom.global.add.f32 %f0, [%rd8], %f0;

$EMB_BWD_DONE:
    ret;
}

// 15. RMSNorm Forward: y = (x / rms) * gamma
.visible .entry rmsnorm_fwd_kernel(
    .param .u64 p_input,
    .param .u64 p_weight,
    .param .u64 p_out,
    .param .f32 p_eps,
    .param .u32 p_seq_len,
    .param .u32 p_hs
) {
    .reg .pred %p;
    .reg .b32 %r<10>;
    .reg .b64 %rd<10>;
    .reg .f32 %f<10>;

    mov.u32 %r0, %ctaid.x;
    mov.u32 %r1, %ntid.x;
    mov.u32 %r2, %tid.x;
    mad.lo.s32 %r3, %r0, %r1, %r2;

    ld.param.u32 %r4, [p_seq_len];
    setp.ge.u32 %p, %r3, %r4;
    @%p bra $RMS_FWD_DONE;

    ld.param.u32 %r5, [p_hs];
    ld.param.u64 %rd0, [p_input];
    ld.param.u64 %rd1, [p_weight];
    ld.param.u64 %rd2, [p_out];

    mul.lo.u32 %r6, %r3, %r5;
    mul.wide.u32 %rd3, %r6, 4;
    add.u64 %rd4, %rd0, %rd3;
    add.u64 %rd5, %rd2, %rd3;

    mov.f32 %f0, 0.0;
    mov.u32 %r7, 0;

$RMS_FWD_SUM:
    setp.ge.u32 %p, %r7, %r5;
    @%p bra $RMS_FWD_NORM;

    mul.wide.u32 %rd6, %r7, 4;
    add.u64 %rd7, %rd4, %rd6;
    ld.global.f32 %f1, [%rd7];
    fma.rn.f32 %f0, %f1, %f1, %f0;

    add.u32 %r7, %r7, 1;
    bra $RMS_FWD_SUM;

$RMS_FWD_NORM:
    cvt.rn.f32.u32 %f2, %r5;
    div.approx.f32 %f3, %f0, %f2;
    ld.param.f32 %f4, [p_eps];
    add.f32 %f3, %f3, %f4;
    rsqrt.approx.f32 %f5, %f3;

    mov.u32 %r7, 0;

$RMS_FWD_STORE:
    setp.ge.u32 %p, %r7, %r5;
    @%p bra $RMS_FWD_DONE;

    mul.wide.u32 %rd6, %r7, 4;
    add.u64 %rd7, %rd4, %rd6;
    add.u64 %rd8, %rd1, %rd6;
    add.u64 %rd9, %rd5, %rd6;

    ld.global.f32 %f1, [%rd7];
    ld.global.f32 %f6, [%rd8];
    mul.f32 %f7, %f1, %f5;
    mul.f32 %f7, %f7, %f6;
    st.global.f32 [%rd9], %f7;

    add.u32 %r7, %r7, 1;
    bra $RMS_FWD_STORE;

$RMS_FWD_DONE:
    ret;
}

// 16. SwiGLU Forward: out = silu(gate) * up
.visible .entry swiglu_fwd_kernel(
    .param .u64 p_gate,
    .param .u64 p_up,
    .param .u64 p_out,
    .param .u32 p_n
) {
    .reg .pred %p;
    .reg .b32 %r<5>;
    .reg .b64 %rd<8>;
    .reg .f32 %f<8>;

    mov.u32 %r0, %ctaid.x;
    mov.u32 %r1, %ntid.x;
    mov.u32 %r2, %tid.x;
    mad.lo.s32 %r3, %r0, %r1, %r2;

    ld.param.u32 %r4, [p_n];
    setp.ge.u32 %p, %r3, %r4;
    @%p bra $SWIGLU_FWD_DONE;

    mul.wide.u32 %rd0, %r3, 4;
    ld.param.u64 %rd1, [p_gate];
    ld.param.u64 %rd2, [p_up];
    ld.param.u64 %rd3, [p_out];

    add.u64 %rd4, %rd1, %rd0;
    add.u64 %rd5, %rd2, %rd0;
    add.u64 %rd6, %rd3, %rd0;

    ld.global.f32 %f0, [%rd4];
    ld.global.f32 %f1, [%rd5];

    // silu(g) = g / (1.0 + exp(-g)) = g * sig(g)
    neg.f32 %f2, %f0;
    mul.f32 %f2, %f2, 1.44269504;
    ex2.approx.f32 %f3, %f2;
    add.f32 %f3, %f3, 1.0;
    div.approx.f32 %f4, %f0, %f3;

    mul.f32 %f5, %f4, %f1;
    st.global.f32 [%rd6], %f5;

$SWIGLU_FWD_DONE:
    ret;
}

// 17. SwiGLU Backward: d_gate and d_up
.visible .entry swiglu_bwd_kernel(
    .param .u64 p_dout,
    .param .u64 p_gate,
    .param .u64 p_up,
    .param .u64 p_dgate,
    .param .u64 p_dup,
    .param .u32 p_n
) {
    .reg .pred %p;
    .reg .b32 %r<5>;
    .reg .b64 %rd<10>;
    .reg .f32 %f<12>;

    mov.u32 %r0, %ctaid.x;
    mov.u32 %r1, %ntid.x;
    mov.u32 %r2, %tid.x;
    mad.lo.s32 %r3, %r0, %r1, %r2;

    ld.param.u32 %r4, [p_n];
    setp.ge.u32 %p, %r3, %r4;
    @%p bra $SWIGLU_BWD_DONE;

    mul.wide.u32 %rd0, %r3, 4;
    ld.param.u64 %rd1, [p_dout];
    ld.param.u64 %rd2, [p_gate];
    ld.param.u64 %rd3, [p_up];
    ld.param.u64 %rd4, [p_dgate];
    ld.param.u64 %rd5, [p_dup];

    add.u64 %rd6, %rd1, %rd0;
    add.u64 %rd7, %rd2, %rd0;
    add.u64 %rd8, %rd3, %rd0;
    add.u64 %rd9, %rd4, %rd0;
    add.u64 %rd1, %rd5, %rd0;

    ld.global.f32 %f0, [%rd6]; // dy
    ld.global.f32 %f1, [%rd7]; // g
    ld.global.f32 %f2, [%rd8]; // u

    // sig(g) = 1.0 / (1.0 + exp(-g))
    neg.f32 %f3, %f1;
    mul.f32 %f3, %f3, 1.44269504;
    ex2.approx.f32 %f4, %f3;
    add.f32 %f4, %f4, 1.0;
    rcp.approx.f32 %f5, %f4; // sig

    // silu(g) = g * sig
    mul.f32 %f6, %f1, %f5;

    // dup = dy * silu(g)
    mul.f32 %f7, %f0, %f6;
    st.global.f32 [%rd1], %f7;

    // dgate = dy * u * sig * (1.0 + g * (1.0 - sig))
    sub.f32 %f8, 1.0, %f5;
    mul.f32 %f8, %f1, %f8;
    add.f32 %f8, %f8, 1.0;
    mul.f32 %f9, %f5, %f8;
    mul.f32 %f10, %f2, %f9;
    mul.f32 %f11, %f0, %f10;
    st.global.f32 [%rd9], %f11;

$SWIGLU_BWD_DONE:
    ret;
}

// 18. Residual Add: out = a + b
.visible .entry residual_add_kernel(
    .param .u64 p_a,
    .param .u64 p_b,
    .param .u64 p_out,
    .param .u32 p_n
) {
    .reg .pred %p;
    .reg .b32 %r<5>;
    .reg .b64 %rd<8>;
    .reg .f32 %f<3>;

    mov.u32 %r0, %ctaid.x;
    mov.u32 %r1, %ntid.x;
    mov.u32 %r2, %tid.x;
    mad.lo.s32 %r3, %r0, %r1, %r2;

    ld.param.u32 %r4, [p_n];
    setp.ge.u32 %p, %r3, %r4;
    @%p bra $RES_ADD_DONE;

    mul.wide.u32 %rd0, %r3, 4;
    ld.param.u64 %rd1, [p_a];
    ld.param.u64 %rd2, [p_b];
    ld.param.u64 %rd3, [p_out];

    add.u64 %rd4, %rd1, %rd0;
    add.u64 %rd5, %rd2, %rd0;
    add.u64 %rd6, %rd3, %rd0;

    ld.global.f32 %f0, [%rd4];
    ld.global.f32 %f1, [%rd5];
    add.f32 %f2, %f0, %f1;
    st.global.f32 [%rd6], %f2;

$RES_ADD_DONE:
    ret;
}
"#;
