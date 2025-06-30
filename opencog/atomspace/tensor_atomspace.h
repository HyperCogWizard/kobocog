#pragma once

/**
 * @file tensor_atomspace.h
 * @brief GGML-backed AtomSpace implementation for OpenCog cognitive architecture
 * 
 * This file implements a tensor-based AtomSpace that maps OpenCog atoms to
 * GGML tensor representations, enabling neural-symbolic reasoning.
 */

#include <memory>
#include <unordered_map>
#include <vector>
#include <string>
#include "ggml.h"

namespace opencog {

// Forward declarations
class TensorAtom;
class TensorAtomSpace;

/**
 * @brief Handle type for referencing atoms in tensor space
 */
using Handle = uint64_t;
constexpr Handle INVALID_HANDLE = 0;

/**
 * @brief Atom types supported in tensor representation
 */
enum class AtomType {
    CONCEPT_NODE = 1,
    PREDICATE_NODE,
    NUMBER_NODE,
    VARIABLE_NODE,
    INHERITANCE_LINK,
    EVALUATION_LINK,
    IMPLICATION_LINK,
    SIMILARITY_LINK,
    UNKNOWN
};

/**
 * @brief Attention values for ECAN system
 */
struct AttentionValue {
    float sti;    // Short-term importance
    float lti;    // Long-term importance  
    float vlti;   // Very long-term importance
    
    AttentionValue(float s = 0.0f, float l = 0.0f, float v = 0.0f)
        : sti(s), lti(l), vlti(v) {}
};

/**
 * @brief Truth value for probabilistic reasoning
 */
struct TruthValue {
    float strength;     // Confidence in truth
    float confidence;   // Meta-confidence
    
    TruthValue(float s = 0.5f, float c = 0.5f)
        : strength(s), confidence(c) {}
};

/**
 * @brief Base class for tensor-backed atoms
 */
class TensorAtom {
public:
    Handle handle;
    AtomType type;
    std::string name;
    struct ggml_tensor* tensor;
    AttentionValue attention;
    TruthValue truth;
    std::vector<Handle> incoming;  // Links pointing to this atom
    std::vector<Handle> outgoing;  // Links this atom points to
    
    TensorAtom(Handle h, AtomType t, const std::string& n, struct ggml_tensor* tens)
        : handle(h), type(t), name(n), tensor(tens) {}
    
    virtual ~TensorAtom() = default;
    
    // Tensor operations
    virtual void update_tensor(struct ggml_context* ctx) = 0;
    virtual size_t tensor_size() const = 0;
};

/**
 * @brief Tensor representation of concept nodes
 */
class TensorConceptNode : public TensorAtom {
public:
    static constexpr size_t EMBEDDING_DIM = 768;
    
    TensorConceptNode(Handle h, const std::string& name, struct ggml_tensor* tensor)
        : TensorAtom(h, AtomType::CONCEPT_NODE, name, tensor) {}
    
    void update_tensor(struct ggml_context* ctx) override;
    size_t tensor_size() const override { return EMBEDDING_DIM; }
};

/**
 * @brief Tensor representation of inheritance links
 */
class TensorInheritanceLink : public TensorAtom {
public:
    static constexpr size_t RELATION_DIM = 768 * 768;
    Handle source_handle;
    Handle target_handle;
    
    TensorInheritanceLink(Handle h, Handle src, Handle tgt, struct ggml_tensor* tensor)
        : TensorAtom(h, AtomType::INHERITANCE_LINK, "InheritanceLink", tensor),
          source_handle(src), target_handle(tgt) {}
    
    void update_tensor(struct ggml_context* ctx) override;
    size_t tensor_size() const override { return RELATION_DIM; }
};

/**
 * @brief GGML-backed AtomSpace for neural-symbolic reasoning
 */
class TensorAtomSpace {
private:
    struct ggml_context* ctx_;
    std::unordered_map<Handle, std::unique_ptr<TensorAtom>> atoms_;
    std::unordered_map<std::string, Handle> name_index_;
    Handle next_handle_;
    
    // Memory management
    size_t mem_size_;
    void* mem_buffer_;
    
public:
    /**
     * @brief Constructor with GGML context
     */
    explicit TensorAtomSpace(size_t memory_size = 256 * 1024 * 1024);
    
    /**
     * @brief Destructor
     */
    ~TensorAtomSpace();
    
    /**
     * @brief Get GGML context
     */
    struct ggml_context* get_context() const { return ctx_; }
    
    /**
     * @brief Add a concept node to the atomspace
     */
    Handle add_concept(const std::string& name, 
                      const float* embedding = nullptr);
    
    /**
     * @brief Add an inheritance link between two atoms
     */
    Handle add_inheritance(Handle source, Handle target, 
                          const TruthValue& tv = TruthValue());
    
    /**
     * @brief Get atom by handle
     */
    TensorAtom* get_atom(Handle handle) const;
    
    /**
     * @brief Get atom by name
     */
    Handle get_handle(const std::string& name) const;
    
    /**
     * @brief Update attention values using ECAN
     */
    void update_attention(Handle handle, const AttentionValue& av);
    
    /**
     * @brief Perform activation spreading
     */
    void spread_activation(Handle source, float amount);
    
    /**
     * @brief Get all atoms with attention above threshold
     */
    std::vector<Handle> get_attentional_focus(float threshold = 0.5f) const;
    
    /**
     * @brief Serialize atomspace to GGUF format
     */
    bool save_to_gguf(const std::string& filename) const;
    
    /**
     * @brief Load atomspace from GGUF format
     */
    bool load_from_gguf(const std::string& filename);
    
    /**
     * @brief Get statistics about the atomspace
     */
    struct Stats {
        size_t num_atoms;
        size_t num_nodes;
        size_t num_links;
        size_t memory_used;
        float avg_attention;
    };
    
    Stats get_stats() const;

private:
    Handle allocate_handle();
    struct ggml_tensor* create_concept_tensor(const std::string& name, 
                                            const float* embedding = nullptr);
    struct ggml_tensor* create_inheritance_tensor(Handle source, Handle target);
    void initialize_memory();
};

} // namespace opencog