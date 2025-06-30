#pragma once

/**
 * @file pln_tensor_engine.h
 * @brief Tensorized Probabilistic Logic Networks (PLN) reasoning engine
 * 
 * This file implements PLN inference using GGML tensors for efficient
 * neural-symbolic reasoning in the OpenCog cognitive architecture.
 */

#include "../atomspace/tensor_atomspace.h"
#include "../operations/ggml_opencog_ops.h"
#include <vector>
#include <memory>

namespace opencog {

/**
 * @brief PLN inference rule types
 */
enum class PLNRuleType {
    DEDUCTION = 0,      // P->Q, Q->R ⊢ P->R
    INDUCTION = 1,      // P->Q, P->R ⊢ Q->R  
    ABDUCTION = 2,      // P->Q, R->Q ⊢ P->R
    REVISION = 3,       // P, P' ⊢ P''
    CHOICE = 4,         // P, Q ⊢ max(P,Q)
    CONJUNCTION = 5,    // P, Q ⊢ P∧Q
    DISJUNCTION = 6,    // P, Q ⊢ P∨Q
    NEGATION = 7,       // P ⊢ ¬P
    INHERITANCE_TO_SIMILARITY = 8,  // P→Q, Q→P ⊢ P↔Q
    SIMILARITY_TO_INHERITANCE = 9   // P↔Q ⊢ P→Q
};

/**
 * @brief PLN inference query
 */
struct PLNQuery {
    Handle target;          // Target atom to prove/derive
    std::vector<Handle> premises;  // Known premises
    float confidence_threshold;    // Minimum confidence required
    int max_steps;          // Maximum inference steps
    bool backward_chaining; // Use backward vs forward chaining
};

/**
 * @brief PLN inference result
 */
struct PLNResult {
    Handle conclusion;      // Derived conclusion
    TruthValue truth;       // Truth value of conclusion
    std::vector<Handle> proof_trail;  // Inference steps
    float confidence;       // Overall confidence
    bool success;           // Whether inference succeeded
};

/**
 * @brief Tensorized PLN reasoning engine
 */
class PLNTensorEngine {
private:
    TensorAtomSpace* atomspace_;
    struct ggml_context* ctx_;
    
    // Inference parameters
    float default_confidence_threshold_;
    int max_inference_depth_;
    
    // Cached tensor operations
    std::vector<struct ggml_tensor*> inference_cache_;
    
public:
    /**
     * @brief Constructor
     */
    explicit PLNTensorEngine(TensorAtomSpace* atomspace);
    
    /**
     * @brief Destructor
     */
    ~PLNTensorEngine();
    
    /**
     * @brief Set inference parameters
     */
    void set_parameters(float confidence_threshold = 0.5f, 
                       int max_depth = 10);
    
    /**
     * @brief Forward chaining inference
     */
    PLNResult forward_chain(const std::vector<Handle>& premises,
                           int max_steps = 10);
    
    /**
     * @brief Backward chaining inference
     */
    PLNResult backward_chain(Handle target,
                            int max_steps = 10);
    
    /**
     * @brief Apply specific PLN rule
     */
    PLNResult apply_rule(PLNRuleType rule,
                        const std::vector<Handle>& premises);
    
    /**
     * @brief Batch inference on multiple queries
     */
    std::vector<PLNResult> batch_inference(
        const std::vector<PLNQuery>& queries);
    
    /**
     * @brief Find applicable rules for given premises
     */
    std::vector<PLNRuleType> find_applicable_rules(
        const std::vector<Handle>& premises);
    
    /**
     * @brief Evaluate truth value of atom using tensor operations
     */
    TruthValue evaluate_truth(Handle atom);
    
    /**
     * @brief Merge truth values using PLN formulas
     */
    TruthValue merge_truth_values(const TruthValue& tv1,
                                 const TruthValue& tv2,
                                 PLNRuleType rule);
    
    /**
     * @brief Create inference graph for visualization
     */
    struct InferenceGraph {
        std::vector<Handle> nodes;
        std::vector<std::pair<Handle, Handle>> edges;
        std::vector<PLNRuleType> edge_rules;
    };
    
    InferenceGraph build_inference_graph(const PLNResult& result);
    
    /**
     * @brief Get inference statistics
     */
    struct PLNStats {
        size_t total_inferences;
        size_t successful_inferences;
        float avg_confidence;
        float avg_steps;
        std::map<PLNRuleType, size_t> rule_usage;
    };
    
    PLNStats get_statistics() const;

private:
    // Internal inference methods
    PLNResult try_deduction(Handle premise1, Handle premise2);
    PLNResult try_induction(Handle premise1, Handle premise2);
    PLNResult try_abduction(Handle premise1, Handle premise2);
    PLNResult try_revision(Handle premise1, Handle premise2);
    
    // Tensor-based truth value operations
    struct ggml_tensor* create_truth_tensor(const TruthValue& tv);
    TruthValue extract_truth_value(struct ggml_tensor* tensor);
    
    // Pattern matching for rule application
    bool matches_deduction_pattern(Handle h1, Handle h2);
    bool matches_induction_pattern(Handle h1, Handle h2);
    bool matches_abduction_pattern(Handle h1, Handle h2);
    
    // Utility functions
    bool is_inheritance_link(Handle handle);
    bool is_similarity_link(Handle handle);
    std::pair<Handle, Handle> get_link_endpoints(Handle link);
    
    // Statistics tracking
    mutable PLNStats stats_;
};

/**
 * @brief PLN formula implementations using GGML tensors
 */
class PLNFormulas {
public:
    /**
     * @brief Deduction formula: P(C|A) * P(B|C) -> P(B|A)
     */
    static TruthValue deduction(const TruthValue& premise1,
                               const TruthValue& premise2);
    
    /**
     * @brief Induction formula: P(B|A) * P(C|A) -> P(C|B)
     */
    static TruthValue induction(const TruthValue& premise1,
                               const TruthValue& premise2);
    
    /**
     * @brief Abduction formula: P(C|A) * P(C|B) -> P(B|A)
     */
    static TruthValue abduction(const TruthValue& premise1,
                               const TruthValue& premise2);
    
    /**
     * @brief Revision formula: combine two estimates of same statement
     */
    static TruthValue revision(const TruthValue& tv1,
                              const TruthValue& tv2);
    
    /**
     * @brief Choice formula: select higher confidence estimate
     */
    static TruthValue choice(const TruthValue& tv1,
                            const TruthValue& tv2);
    
    /**
     * @brief Conjunction formula: P(A∧B) from P(A) and P(B)
     */
    static TruthValue conjunction(const TruthValue& tv1,
                                 const TruthValue& tv2);
    
    /**
     * @brief Disjunction formula: P(A∨B) from P(A) and P(B)
     */
    static TruthValue disjunction(const TruthValue& tv1,
                                 const TruthValue& tv2);
    
    /**
     * @brief Negation formula: P(¬A) from P(A)
     */
    static TruthValue negation(const TruthValue& tv);

private:
    // Helper functions for truth value arithmetic
    static float and_function(float s1, float s2);
    static float or_function(float s1, float s2);
    static float w2c(float w);  // Convert weight to confidence
    static float c2w(float c);  // Convert confidence to weight
};

} // namespace opencog