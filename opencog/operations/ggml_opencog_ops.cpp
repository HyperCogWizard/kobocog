/**
 * @file ggml_opencog_ops.cpp
 * @brief Implementation of custom GGML operations for OpenCog
 */

#include "ggml_opencog_ops.h"
#include <cmath>
#include <algorithm>
#include <cstring>

// Helper function for computing tensor similarity
static float compute_similarity(const float* a, const float* b, size_t size) {
    float dot_product = 0.0f;
    float norm_a = 0.0f;
    float norm_b = 0.0f;
    
    for (size_t i = 0; i < size; ++i) {
        dot_product += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }
    
    if (norm_a == 0.0f || norm_b == 0.0f) return 0.0f;
    return dot_product / (sqrtf(norm_a) * sqrtf(norm_b));
}

// Forward declarations for operation implementations
static void ggml_compute_forward_atomspace_lookup_f32(
    const struct ggml_compute_params* params,
    struct ggml_tensor* dst);

static void ggml_compute_forward_pattern_match_f32(
    const struct ggml_compute_params* params,
    struct ggml_tensor* dst);

static void ggml_compute_forward_activation_spread_f32(
    const struct ggml_compute_params* params,
    struct ggml_tensor* dst);

static void ggml_compute_forward_attention_update_f32(
    const struct ggml_compute_params* params,
    struct ggml_tensor* dst);

static void ggml_compute_forward_pln_inference_f32(
    const struct ggml_compute_params* params,
    struct ggml_tensor* dst);

struct ggml_tensor* ggml_atomspace_lookup(
    struct ggml_context* ctx,
    struct ggml_tensor* atomspace,
    struct ggml_tensor* query) {
    
    GGML_ASSERT(atomspace->type == GGML_TYPE_F32);
    GGML_ASSERT(query->type == GGML_TYPE_F32);
    
    // Result tensor with same dimensions as query
    struct ggml_tensor* result = ggml_new_tensor(ctx, GGML_TYPE_F32, 
                                                query->n_dims, query->ne);
    
    result->op = (enum ggml_op)GGML_OP_ATOMSPACE_LOOKUP;
    result->src[0] = atomspace;
    result->src[1] = query;
    
    return result;
}

struct ggml_tensor* ggml_pattern_match(
    struct ggml_context* ctx,
    struct ggml_tensor* pattern,
    struct ggml_tensor* target,
    float threshold) {
    
    GGML_ASSERT(pattern->type == GGML_TYPE_F32);
    GGML_ASSERT(target->type == GGML_TYPE_F32);
    
    // Result is a scalar similarity score
    struct ggml_tensor* result = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
    
    result->op = (enum ggml_op)GGML_OP_PATTERN_MATCH;
    result->src[0] = pattern;
    result->src[1] = target;
    
    // Store threshold in op_params
    *(float*)&result->op_params[0] = threshold;
    
    return result;
}

struct ggml_tensor* ggml_activation_spread(
    struct ggml_context* ctx,
    struct ggml_tensor* graph,
    struct ggml_tensor* activation,
    float decay_factor) {
    
    GGML_ASSERT(graph->type == GGML_TYPE_F32);
    GGML_ASSERT(activation->type == GGML_TYPE_F32);
    
    // Result has same shape as activation
    struct ggml_tensor* result = ggml_new_tensor(ctx, GGML_TYPE_F32,
                                                activation->n_dims, activation->ne);
    
    result->op = (enum ggml_op)GGML_OP_ACTIVATION_SPREAD;
    result->src[0] = graph;
    result->src[1] = activation;
    
    // Store decay factor
    *(float*)&result->op_params[0] = decay_factor;
    
    return result;
}

struct ggml_tensor* ggml_attention_update(
    struct ggml_context* ctx,
    struct ggml_tensor* attention_values,
    struct ggml_tensor* importance_tensor,
    struct ggml_tensor* wage_tensor) {
    
    GGML_ASSERT(attention_values->type == GGML_TYPE_F32);
    GGML_ASSERT(importance_tensor->type == GGML_TYPE_F32);
    GGML_ASSERT(wage_tensor->type == GGML_TYPE_F32);
    
    struct ggml_tensor* result = ggml_new_tensor(ctx, GGML_TYPE_F32,
                                                attention_values->n_dims, 
                                                attention_values->ne);
    
    result->op = (enum ggml_op)GGML_OP_ATTENTION_UPDATE;
    result->src[0] = attention_values;
    result->src[1] = importance_tensor;
    result->src[2] = wage_tensor;
    
    return result;
}

struct ggml_tensor* ggml_pln_inference(
    struct ggml_context* ctx,
    struct ggml_tensor* premise1,
    struct ggml_tensor* premise2,
    int rule_type) {
    
    GGML_ASSERT(premise1->type == GGML_TYPE_F32);
    GGML_ASSERT(premise2->type == GGML_TYPE_F32);
    
    // Result tensor for inference output
    struct ggml_tensor* result = ggml_new_tensor(ctx, GGML_TYPE_F32,
                                                premise1->n_dims, premise1->ne);
    
    result->op = (enum ggml_op)GGML_OP_PLN_INFERENCE;
    result->src[0] = premise1;
    result->src[1] = premise2;
    
    // Store rule type
    result->op_params[0] = rule_type;
    
    return result;
}

struct ggml_tensor* ggml_truth_value_merge(
    struct ggml_context* ctx,
    struct ggml_tensor* truth1,
    struct ggml_tensor* truth2,
    int merge_type) {
    
    GGML_ASSERT(truth1->type == GGML_TYPE_F32);
    GGML_ASSERT(truth2->type == GGML_TYPE_F32);
    
    // Truth values are [strength, confidence] pairs
    struct ggml_tensor* result = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 2);
    
    result->op = (enum ggml_op)GGML_OP_TRUTH_VALUE_MERGE;
    result->src[0] = truth1;
    result->src[1] = truth2;
    
    result->op_params[0] = merge_type;
    
    return result;
}

// Implementation of compute functions

static void ggml_compute_forward_atomspace_lookup_f32(
    const struct ggml_compute_params* params,
    struct ggml_tensor* dst) {
    
    const struct ggml_tensor* atomspace = dst->src[0];
    const struct ggml_tensor* query = dst->src[1];
    
    if (params->type == GGML_TASK_TYPE_INIT || params->type == GGML_TASK_TYPE_FINALIZE) {
        return;
    }
    
    const float* atomspace_data = (const float*)atomspace->data;
    const float* query_data = (const float*)query->data;
    float* result_data = (float*)dst->data;
    
    // Simple lookup implementation - find most similar atom
    size_t num_atoms = ggml_nelements(atomspace) / ggml_nelements(query);
    size_t atom_size = ggml_nelements(query);
    
    float best_similarity = -1.0f;
    size_t best_idx = 0;
    
    for (size_t i = 0; i < num_atoms; ++i) {
        float sim = compute_similarity(&atomspace_data[i * atom_size], 
                                     query_data, atom_size);
        if (sim > best_similarity) {
            best_similarity = sim;
            best_idx = i;
        }
    }
    
    // Copy best matching atom to result
    memcpy(result_data, &atomspace_data[best_idx * atom_size], 
           atom_size * sizeof(float));
}

static void ggml_compute_forward_pattern_match_f32(
    const struct ggml_compute_params* params,
    struct ggml_tensor* dst) {
    
    const struct ggml_tensor* pattern = dst->src[0];
    const struct ggml_tensor* target = dst->src[1];
    float threshold = *(const float*)&dst->op_params[0];
    
    if (params->type == GGML_TASK_TYPE_INIT || params->type == GGML_TASK_TYPE_FINALIZE) {
        return;
    }
    
    const float* pattern_data = (const float*)pattern->data;
    const float* target_data = (const float*)target->data;
    float* result_data = (float*)dst->data;
    
    size_t size = ggml_nelements(pattern);
    float similarity = compute_similarity(pattern_data, target_data, size);
    
    // Apply threshold
    result_data[0] = (similarity >= threshold) ? similarity : 0.0f;
}

static void ggml_compute_forward_activation_spread_f32(
    const struct ggml_compute_params* params,
    struct ggml_tensor* dst) {
    
    const struct ggml_tensor* graph = dst->src[0];
    const struct ggml_tensor* activation = dst->src[1];
    float decay = *(const float*)&dst->op_params[0];
    
    if (params->type == GGML_TASK_TYPE_INIT || params->type == GGML_TASK_TYPE_FINALIZE) {
        return;
    }
    
    const float* graph_data = (const float*)graph->data;
    const float* activation_data = (const float*)activation->data;
    float* result_data = (float*)dst->data;
    
    size_t n_nodes = ggml_nelements(activation);
    
    // Copy current activation with decay
    for (size_t i = 0; i < n_nodes; ++i) {
        result_data[i] = activation_data[i] * decay;
    }
    
    // Spread activation through graph connections
    for (size_t i = 0; i < n_nodes; ++i) {
        for (size_t j = 0; j < n_nodes; ++j) {
            float connection_strength = graph_data[i * n_nodes + j];
            result_data[j] += activation_data[i] * connection_strength * 0.1f;
        }
    }
    
    // Normalize to prevent explosion
    float sum = 0.0f;
    for (size_t i = 0; i < n_nodes; ++i) {
        sum += result_data[i];
    }
    if (sum > 0.0f) {
        for (size_t i = 0; i < n_nodes; ++i) {
            result_data[i] /= sum;
        }
    }
}

static void ggml_compute_forward_attention_update_f32(
    const struct ggml_compute_params* params,
    struct ggml_tensor* dst) {
    
    const struct ggml_tensor* attention = dst->src[0];
    const struct ggml_tensor* importance = dst->src[1];
    const struct ggml_tensor* wage = dst->src[2];
    
    if (params->type == GGML_TASK_TYPE_INIT || params->type == GGML_TASK_TYPE_FINALIZE) {
        return;
    }
    
    const float* attention_data = (const float*)attention->data;
    const float* importance_data = (const float*)importance->data;
    const float* wage_data = (const float*)wage->data;
    float* result_data = (float*)dst->data;
    
    size_t n_atoms = ggml_nelements(attention) / 3; // [STI, LTI, VLTI]
    
    // ECAN attention update algorithm
    for (size_t i = 0; i < n_atoms; ++i) {
        size_t base_idx = i * 3;
        
        float sti = attention_data[base_idx];
        float lti = attention_data[base_idx + 1];
        float vlti = attention_data[base_idx + 2];
        float imp = importance_data[i];
        float w = wage_data[i];
        
        // Update STI based on importance and economic wage
        float new_sti = sti + imp * w * 0.1f;
        
        // Update LTI based on sustained attention
        float new_lti = lti + (sti > 0.5f ? 0.01f : -0.01f);
        
        // Update VLTI slowly
        float new_vlti = vlti + (lti > 0.5f ? 0.001f : -0.001f);
        
        // Apply bounds
        result_data[base_idx] = fmaxf(0.0f, fminf(1.0f, new_sti));
        result_data[base_idx + 1] = fmaxf(0.0f, fminf(1.0f, new_lti));
        result_data[base_idx + 2] = fmaxf(0.0f, fminf(1.0f, new_vlti));
    }
}

static void ggml_compute_forward_pln_inference_f32(
    const struct ggml_compute_params* params,
    struct ggml_tensor* dst) {
    
    const struct ggml_tensor* premise1 = dst->src[0];
    const struct ggml_tensor* premise2 = dst->src[1];
    int rule_type = dst->op_params[0];
    
    if (params->type == GGML_TASK_TYPE_INIT || params->type == GGML_TASK_TYPE_FINALIZE) {
        return;
    }
    
    const float* p1_data = (const float*)premise1->data;
    const float* p2_data = (const float*)premise2->data;
    float* result_data = (float*)dst->data;
    
    // Assume premises are [strength, confidence] truth values
    float s1 = p1_data[0], c1 = p1_data[1];
    float s2 = p2_data[0], c2 = p2_data[1];
    
    float result_strength = 0.0f;
    float result_confidence = 0.0f;
    
    switch (rule_type) {
        case 0: // Deduction: P->Q, Q->R implies P->R
            result_strength = s1 * s2;
            result_confidence = c1 * c2 * fmaxf(s1, s2);
            break;
            
        case 1: // Induction: P->Q, P->R implies Q->R
            result_strength = s1 * s2;
            result_confidence = c1 * c2 * (1.0f - fmaxf(s1, s2));
            break;
            
        case 2: // Abduction: P->Q, R->Q implies P->R
            result_strength = s1 * s2;
            result_confidence = c1 * c2 * (1.0f - fminf(s1, s2));
            break;
            
        default: // Simple conjunction
            result_strength = s1 * s2;
            result_confidence = c1 * c2;
            break;
    }
    
    result_data[0] = fmaxf(0.0f, fminf(1.0f, result_strength));
    result_data[1] = fmaxf(0.0f, fminf(1.0f, result_confidence));
}

void ggml_opencog_init() {
    // Register custom operations with GGML
    // This would require modifying GGML core to support registration
    // For now, operations are implemented as standalone functions
}