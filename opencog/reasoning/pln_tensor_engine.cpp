/**
 * @file pln_tensor_engine.cpp
 * @brief Implementation of tensorized PLN reasoning engine
 */

#include "pln_tensor_engine.h"
#include <algorithm>
#include <cmath>
#include <iostream>

namespace opencog {

// PLNTensorEngine implementation

PLNTensorEngine::PLNTensorEngine(TensorAtomSpace* atomspace)
    : atomspace_(atomspace), ctx_(atomspace->get_context()),
      default_confidence_threshold_(0.5f), max_inference_depth_(10) {
    stats_ = {};
}

PLNTensorEngine::~PLNTensorEngine() {
    // Clean up cached tensors
    for (auto* tensor : inference_cache_) {
        // Tensors are managed by GGML context, so no explicit cleanup needed
    }
}

void PLNTensorEngine::set_parameters(float confidence_threshold, int max_depth) {
    default_confidence_threshold_ = confidence_threshold;
    max_inference_depth_ = max_depth;
}

PLNResult PLNTensorEngine::forward_chain(const std::vector<Handle>& premises,
                                         int max_steps) {
    PLNResult result = {};
    result.success = false;
    
    std::vector<Handle> working_set = premises;
    std::vector<Handle> derived_atoms;
    
    for (int step = 0; step < max_steps; ++step) {
        bool made_progress = false;
        
        // Try all pairs of atoms in working set
        for (size_t i = 0; i < working_set.size() && !made_progress; ++i) {
            for (size_t j = i + 1; j < working_set.size() && !made_progress; ++j) {
                Handle h1 = working_set[i];
                Handle h2 = working_set[j];
                
                // Try different inference rules
                auto rules = find_applicable_rules({h1, h2});
                for (auto rule : rules) {
                    PLNResult step_result = apply_rule(rule, {h1, h2});
                    if (step_result.success && 
                        step_result.confidence >= default_confidence_threshold_) {
                        
                        // Add derived atom to working set
                        working_set.push_back(step_result.conclusion);
                        derived_atoms.push_back(step_result.conclusion);
                        result.proof_trail.push_back(step_result.conclusion);
                        made_progress = true;
                        
                        // Update result
                        result.conclusion = step_result.conclusion;
                        result.truth = step_result.truth;
                        result.confidence = step_result.confidence;
                        result.success = true;
                        break;
                    }
                }
            }
        }
        
        if (!made_progress) break;
    }
    
    stats_.total_inferences++;
    if (result.success) stats_.successful_inferences++;
    
    return result;
}

PLNResult PLNTensorEngine::backward_chain(Handle target, int max_steps) {
    PLNResult result = {};
    result.success = false;
    result.conclusion = target;
    
    // Check if target already has sufficient truth value
    TruthValue target_tv = evaluate_truth(target);
    if (target_tv.confidence >= default_confidence_threshold_) {
        result.truth = target_tv;
        result.confidence = target_tv.confidence;
        result.success = true;
        return result;
    }
    
    // Try to find premises that could lead to target
    TensorAtom* target_atom = atomspace_->get_atom(target);
    if (!target_atom) return result;
    
    // Look for atoms that could produce target through inference
    for (int step = 0; step < max_steps; ++step) {
        bool made_progress = false;
        
        // Find potential premises in atomspace
        auto focus_atoms = atomspace_->get_attentional_focus(0.1f);
        
        for (size_t i = 0; i < focus_atoms.size() && !made_progress; ++i) {
            for (size_t j = i + 1; j < focus_atoms.size() && !made_progress; ++j) {
                Handle h1 = focus_atoms[i];
                Handle h2 = focus_atoms[j];
                
                // Try rules that could derive target
                if (matches_deduction_pattern(h1, h2)) {
                    PLNResult step_result = try_deduction(h1, h2);
                    if (step_result.success && step_result.conclusion == target) {
                        result = step_result;
                        made_progress = true;
                    }
                }
            }
        }
        
        if (!made_progress) break;
    }
    
    stats_.total_inferences++;
    if (result.success) stats_.successful_inferences++;
    
    return result;
}

PLNResult PLNTensorEngine::apply_rule(PLNRuleType rule,
                                      const std::vector<Handle>& premises) {
    if (premises.size() < 2) {
        return {INVALID_HANDLE, TruthValue(), {}, 0.0f, false};
    }
    
    Handle h1 = premises[0];
    Handle h2 = premises[1];
    
    switch (rule) {
        case PLNRuleType::DEDUCTION:
            return try_deduction(h1, h2);
        case PLNRuleType::INDUCTION:
            return try_induction(h1, h2);
        case PLNRuleType::ABDUCTION:
            return try_abduction(h1, h2);
        case PLNRuleType::REVISION:
            return try_revision(h1, h2);
        default:
            return {INVALID_HANDLE, TruthValue(), {}, 0.0f, false};
    }
}

std::vector<PLNRuleType> PLNTensorEngine::find_applicable_rules(
    const std::vector<Handle>& premises) {
    
    std::vector<PLNRuleType> applicable_rules;
    
    if (premises.size() >= 2) {
        Handle h1 = premises[0];
        Handle h2 = premises[1];
        
        if (matches_deduction_pattern(h1, h2)) {
            applicable_rules.push_back(PLNRuleType::DEDUCTION);
        }
        if (matches_induction_pattern(h1, h2)) {
            applicable_rules.push_back(PLNRuleType::INDUCTION);
        }
        if (matches_abduction_pattern(h1, h2)) {
            applicable_rules.push_back(PLNRuleType::ABDUCTION);
        }
        
        // Always try revision if atoms are of same type
        TensorAtom* atom1 = atomspace_->get_atom(h1);
        TensorAtom* atom2 = atomspace_->get_atom(h2);
        if (atom1 && atom2 && atom1->type == atom2->type) {
            applicable_rules.push_back(PLNRuleType::REVISION);
        }
    }
    
    return applicable_rules;
}

TruthValue PLNTensorEngine::evaluate_truth(Handle atom) {
    TensorAtom* tensor_atom = atomspace_->get_atom(atom);
    if (!tensor_atom) {
        return TruthValue(0.0f, 0.0f);
    }
    
    return tensor_atom->truth;
}

TruthValue PLNTensorEngine::merge_truth_values(const TruthValue& tv1,
                                               const TruthValue& tv2,
                                               PLNRuleType rule) {
    switch (rule) {
        case PLNRuleType::DEDUCTION:
            return PLNFormulas::deduction(tv1, tv2);
        case PLNRuleType::INDUCTION:
            return PLNFormulas::induction(tv1, tv2);
        case PLNRuleType::ABDUCTION:
            return PLNFormulas::abduction(tv1, tv2);
        case PLNRuleType::REVISION:
            return PLNFormulas::revision(tv1, tv2);
        case PLNRuleType::CONJUNCTION:
            return PLNFormulas::conjunction(tv1, tv2);
        case PLNRuleType::DISJUNCTION:
            return PLNFormulas::disjunction(tv1, tv2);
        default:
            return PLNFormulas::revision(tv1, tv2);
    }
}

PLNResult PLNTensorEngine::try_deduction(Handle premise1, Handle premise2) {
    // Pattern: A->B, B->C ⊢ A->C
    PLNResult result = {};
    result.success = false;
    
    if (!is_inheritance_link(premise1) || !is_inheritance_link(premise2)) {
        return result;
    }
    
    auto [a1, b1] = get_link_endpoints(premise1);
    auto [b2, c2] = get_link_endpoints(premise2);
    
    // Check if B1 == B2 (common middle term)
    if (b1 != b2) return result;
    
    // Get truth values
    TruthValue tv1 = evaluate_truth(premise1);
    TruthValue tv2 = evaluate_truth(premise2);
    
    // Apply deduction formula
    TruthValue result_tv = PLNFormulas::deduction(tv1, tv2);
    
    if (result_tv.confidence >= default_confidence_threshold_) {
        // Create new inheritance link A->C
        Handle conclusion = atomspace_->add_inheritance(a1, c2, result_tv);
        
        result.conclusion = conclusion;
        result.truth = result_tv;
        result.confidence = result_tv.confidence;
        result.proof_trail = {premise1, premise2};
        result.success = true;
        
        stats_.rule_usage[PLNRuleType::DEDUCTION]++;
    }
    
    return result;
}

PLNResult PLNTensorEngine::try_induction(Handle premise1, Handle premise2) {
    // Pattern: A->B, A->C ⊢ B->C
    PLNResult result = {};
    result.success = false;
    
    if (!is_inheritance_link(premise1) || !is_inheritance_link(premise2)) {
        return result;
    }
    
    auto [a1, b1] = get_link_endpoints(premise1);
    auto [a2, c2] = get_link_endpoints(premise2);
    
    // Check if A1 == A2 (common source)
    if (a1 != a2) return result;
    
    TruthValue tv1 = evaluate_truth(premise1);
    TruthValue tv2 = evaluate_truth(premise2);
    
    TruthValue result_tv = PLNFormulas::induction(tv1, tv2);
    
    if (result_tv.confidence >= default_confidence_threshold_) {
        Handle conclusion = atomspace_->add_inheritance(b1, c2, result_tv);
        
        result.conclusion = conclusion;
        result.truth = result_tv;
        result.confidence = result_tv.confidence;
        result.proof_trail = {premise1, premise2};
        result.success = true;
        
        stats_.rule_usage[PLNRuleType::INDUCTION]++;
    }
    
    return result;
}

PLNResult PLNTensorEngine::try_abduction(Handle premise1, Handle premise2) {
    // Pattern: A->C, B->C ⊢ A->B
    PLNResult result = {};
    result.success = false;
    
    if (!is_inheritance_link(premise1) || !is_inheritance_link(premise2)) {
        return result;
    }
    
    auto [a1, c1] = get_link_endpoints(premise1);
    auto [b2, c2] = get_link_endpoints(premise2);
    
    // Check if C1 == C2 (common target)
    if (c1 != c2) return result;
    
    TruthValue tv1 = evaluate_truth(premise1);
    TruthValue tv2 = evaluate_truth(premise2);
    
    TruthValue result_tv = PLNFormulas::abduction(tv1, tv2);
    
    if (result_tv.confidence >= default_confidence_threshold_) {
        Handle conclusion = atomspace_->add_inheritance(a1, b2, result_tv);
        
        result.conclusion = conclusion;
        result.truth = result_tv;
        result.confidence = result_tv.confidence;
        result.proof_trail = {premise1, premise2};
        result.success = true;
        
        stats_.rule_usage[PLNRuleType::ABDUCTION]++;
    }
    
    return result;
}

PLNResult PLNTensorEngine::try_revision(Handle premise1, Handle premise2) {
    // Combine two truth values for same statement
    PLNResult result = {};
    result.success = false;
    
    // For simplicity, assume premises refer to same statement
    TruthValue tv1 = evaluate_truth(premise1);
    TruthValue tv2 = evaluate_truth(premise2);
    
    TruthValue result_tv = PLNFormulas::revision(tv1, tv2);
    
    // Update truth value of one of the premises
    TensorAtom* atom = atomspace_->get_atom(premise1);
    if (atom) {
        atom->truth = result_tv;
        atom->update_tensor(ctx_);
        
        result.conclusion = premise1;
        result.truth = result_tv;
        result.confidence = result_tv.confidence;
        result.proof_trail = {premise1, premise2};
        result.success = true;
        
        stats_.rule_usage[PLNRuleType::REVISION]++;
    }
    
    return result;
}

// Helper functions

bool PLNTensorEngine::matches_deduction_pattern(Handle h1, Handle h2) {
    if (!is_inheritance_link(h1) || !is_inheritance_link(h2)) return false;
    
    auto [a1, b1] = get_link_endpoints(h1);
    auto [b2, c2] = get_link_endpoints(h2);
    
    return b1 == b2; // Common middle term
}

bool PLNTensorEngine::matches_induction_pattern(Handle h1, Handle h2) {
    if (!is_inheritance_link(h1) || !is_inheritance_link(h2)) return false;
    
    auto [a1, b1] = get_link_endpoints(h1);
    auto [a2, c2] = get_link_endpoints(h2);
    
    return a1 == a2; // Common source
}

bool PLNTensorEngine::matches_abduction_pattern(Handle h1, Handle h2) {
    if (!is_inheritance_link(h1) || !is_inheritance_link(h2)) return false;
    
    auto [a1, c1] = get_link_endpoints(h1);
    auto [b2, c2] = get_link_endpoints(h2);
    
    return c1 == c2; // Common target
}

bool PLNTensorEngine::is_inheritance_link(Handle handle) {
    TensorAtom* atom = atomspace_->get_atom(handle);
    return atom && atom->type == AtomType::INHERITANCE_LINK;
}

std::pair<Handle, Handle> PLNTensorEngine::get_link_endpoints(Handle link) {
    TensorAtom* atom = atomspace_->get_atom(link);
    if (atom && atom->type == AtomType::INHERITANCE_LINK) {
        auto* inheritance_link = static_cast<TensorInheritanceLink*>(atom);
        return {inheritance_link->source_handle, inheritance_link->target_handle};
    }
    return {INVALID_HANDLE, INVALID_HANDLE};
}

PLNTensorEngine::PLNStats PLNTensorEngine::get_statistics() const {
    PLNStats stats = stats_;
    if (stats.total_inferences > 0) {
        stats.avg_confidence = static_cast<float>(stats.successful_inferences) / stats.total_inferences;
    }
    return stats;
}

// PLNFormulas implementation

TruthValue PLNFormulas::deduction(const TruthValue& premise1, const TruthValue& premise2) {
    float s1 = premise1.strength;
    float c1 = premise1.confidence;
    float s2 = premise2.strength;
    float c2 = premise2.confidence;
    
    float result_strength = s1 * s2;
    float result_confidence = c1 * c2 * std::max(s1, s2);
    
    return TruthValue(result_strength, result_confidence);
}

TruthValue PLNFormulas::induction(const TruthValue& premise1, const TruthValue& premise2) {
    float s1 = premise1.strength;
    float c1 = premise1.confidence;
    float s2 = premise2.strength;
    float c2 = premise2.confidence;
    
    float result_strength = s1 * s2;
    float result_confidence = c1 * c2 * (1.0f - std::max(s1, s2));
    
    return TruthValue(result_strength, result_confidence);
}

TruthValue PLNFormulas::abduction(const TruthValue& premise1, const TruthValue& premise2) {
    float s1 = premise1.strength;
    float c1 = premise1.confidence;
    float s2 = premise2.strength;
    float c2 = premise2.confidence;
    
    float result_strength = s1 * s2;
    float result_confidence = c1 * c2 * (1.0f - std::min(s1, s2));
    
    return TruthValue(result_strength, result_confidence);
}

TruthValue PLNFormulas::revision(const TruthValue& tv1, const TruthValue& tv2) {
    float s1 = tv1.strength;
    float c1 = tv1.confidence;
    float s2 = tv2.strength;
    float c2 = tv2.confidence;
    
    float w1 = c2w(c1);
    float w2 = c2w(c2);
    float w = w1 + w2;
    
    if (w == 0.0f) return TruthValue(0.5f, 0.0f);
    
    float result_strength = (s1 * w1 + s2 * w2) / w;
    float result_confidence = w2c(w);
    
    return TruthValue(result_strength, result_confidence);
}

TruthValue PLNFormulas::conjunction(const TruthValue& tv1, const TruthValue& tv2) {
    float s1 = tv1.strength;
    float c1 = tv1.confidence;
    float s2 = tv2.strength;
    float c2 = tv2.confidence;
    
    float result_strength = and_function(s1, s2);
    float result_confidence = c1 * c2;
    
    return TruthValue(result_strength, result_confidence);
}

TruthValue PLNFormulas::disjunction(const TruthValue& tv1, const TruthValue& tv2) {
    float s1 = tv1.strength;
    float c1 = tv1.confidence;
    float s2 = tv2.strength;
    float c2 = tv2.confidence;
    
    float result_strength = or_function(s1, s2);
    float result_confidence = c1 * c2;
    
    return TruthValue(result_strength, result_confidence);
}

TruthValue PLNFormulas::negation(const TruthValue& tv) {
    return TruthValue(1.0f - tv.strength, tv.confidence);
}

// Helper functions

float PLNFormulas::and_function(float s1, float s2) {
    return s1 * s2;
}

float PLNFormulas::or_function(float s1, float s2) {
    return s1 + s2 - s1 * s2;
}

float PLNFormulas::w2c(float w) {
    return w / (w + 1.0f);
}

float PLNFormulas::c2w(float c) {
    if (c >= 1.0f) return 1000.0f; // Very high weight
    return c / (1.0f - c);
}

} // namespace opencog