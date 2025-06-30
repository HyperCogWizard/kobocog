#pragma once

/**
 * @file ggml_opencog_ops.h
 * @brief Custom GGML operations for OpenCog symbolic reasoning
 * 
 * This file defines GGML custom operations specifically designed for
 * neural-symbolic reasoning in the OpenCog cognitive architecture.
 */

#include "ggml.h"
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Custom operation types for OpenCog
enum ggml_opencog_op {
    GGML_OP_ATOMSPACE_LOOKUP = GGML_OP_COUNT + 1000,
    GGML_OP_PATTERN_MATCH,
    GGML_OP_ACTIVATION_SPREAD,
    GGML_OP_ATTENTION_UPDATE,
    GGML_OP_PLN_INFERENCE,
    GGML_OP_HYPERGRAPH_CONV,
    GGML_OP_SYMBOLIC_UNIFY,
    GGML_OP_TRUTH_VALUE_MERGE,
};

/**
 * @brief Lookup atoms in tensor atomspace
 * 
 * @param ctx GGML context
 * @param atomspace Tensor containing atomspace representation
 * @param query Query tensor for lookup
 * @return Result tensor containing matched atoms
 */
struct ggml_tensor* ggml_atomspace_lookup(
    struct ggml_context* ctx,
    struct ggml_tensor* atomspace,
    struct ggml_tensor* query);

/**
 * @brief Pattern matching operation for symbolic structures
 * 
 * @param ctx GGML context  
 * @param pattern Pattern tensor to match
 * @param target Target tensor to match against
 * @param threshold Matching threshold
 * @return Match score tensor
 */
struct ggml_tensor* ggml_pattern_match(
    struct ggml_context* ctx,
    struct ggml_tensor* pattern,
    struct ggml_tensor* target,
    float threshold);

/**
 * @brief Activation spreading in hypergraph structure
 * 
 * @param ctx GGML context
 * @param graph Hypergraph adjacency tensor
 * @param activation Current activation levels
 * @param decay_factor Activation decay rate
 * @return Updated activation tensor
 */
struct ggml_tensor* ggml_activation_spread(
    struct ggml_context* ctx,
    struct ggml_tensor* graph,
    struct ggml_tensor* activation,
    float decay_factor);

/**
 * @brief Update attention values using ECAN algorithm
 * 
 * @param ctx GGML context
 * @param attention_values Current attention tensor
 * @param importance_tensor Importance update tensor
 * @param wage_tensor Economic wage tensor
 * @return Updated attention values
 */
struct ggml_tensor* ggml_attention_update(
    struct ggml_context* ctx,
    struct ggml_tensor* attention_values,
    struct ggml_tensor* importance_tensor,
    struct ggml_tensor* wage_tensor);

/**
 * @brief PLN (Probabilistic Logic Networks) inference step
 * 
 * @param ctx GGML context
 * @param premise1 First premise tensor
 * @param premise2 Second premise tensor  
 * @param rule_type Type of inference rule to apply
 * @return Inference result tensor
 */
struct ggml_tensor* ggml_pln_inference(
    struct ggml_context* ctx,
    struct ggml_tensor* premise1,
    struct ggml_tensor* premise2,
    int rule_type);

/**
 * @brief Hypergraph convolution for structural reasoning
 * 
 * @param ctx GGML context
 * @param node_features Node feature tensor
 * @param hypergraph Hypergraph structure tensor
 * @param weights Convolution weights
 * @return Convolved features
 */
struct ggml_tensor* ggml_hypergraph_conv(
    struct ggml_context* ctx,
    struct ggml_tensor* node_features,
    struct ggml_tensor* hypergraph,
    struct ggml_tensor* weights);

/**
 * @brief Symbolic unification operation
 * 
 * @param ctx GGML context
 * @param term1 First term tensor
 * @param term2 Second term tensor
 * @param variable_mask Tensor indicating variables
 * @return Unification result and substitution
 */
struct ggml_tensor* ggml_symbolic_unify(
    struct ggml_context* ctx,
    struct ggml_tensor* term1,
    struct ggml_tensor* term2,
    struct ggml_tensor* variable_mask);

/**
 * @brief Merge truth values using PLN formulas
 * 
 * @param ctx GGML context
 * @param truth1 First truth value tensor [strength, confidence]
 * @param truth2 Second truth value tensor [strength, confidence]
 * @param merge_type Type of merge operation (0=AND, 1=OR, 2=REVISION)
 * @return Merged truth value tensor
 */
struct ggml_tensor* ggml_truth_value_merge(
    struct ggml_context* ctx,
    struct ggml_tensor* truth1,
    struct ggml_tensor* truth2,
    int merge_type);

/**
 * @brief Initialize OpenCog custom operations
 * 
 * This function registers all custom operations with GGML.
 * Should be called once during initialization.
 */
void ggml_opencog_init();

/**
 * @brief Parameters for attention update operation
 */
struct ggml_opencog_attention_params {
    float sti_decay;        // Short-term importance decay
    float lti_update;       // Long-term importance update rate
    float wage_factor;      // Economic wage factor
    int max_atoms;          // Maximum atoms in focus
};

/**
 * @brief Parameters for PLN inference
 */
struct ggml_opencog_pln_params {
    int rule_type;          // Inference rule type
    float confidence_threshold; // Minimum confidence for inference
    int max_depth;          // Maximum inference depth
};

/**
 * @brief Parameters for pattern matching
 */
struct ggml_opencog_pattern_params {
    float similarity_threshold;  // Minimum similarity for match
    int max_matches;            // Maximum number of matches to return
    bool fuzzy_matching;        // Enable fuzzy pattern matching
};

#ifdef __cplusplus
}
#endif