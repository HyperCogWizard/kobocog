/**
 * @file tensor_atomspace.cpp
 * @brief Implementation of GGML-backed AtomSpace
 */

#include "tensor_atomspace.h"
#include <random>
#include <cstring>
#include <algorithm>
#include <iostream>

namespace opencog {

// TensorConceptNode implementation
void TensorConceptNode::update_tensor(struct ggml_context* ctx) {
    if (!tensor) return;
    
    // Update tensor based on attention and truth values
    float* data = (float*)tensor->data;
    if (data) {
        // Modulate embedding based on attention
        float attention_factor = 1.0f + attention.sti * 0.1f;
        for (size_t i = 0; i < EMBEDDING_DIM; ++i) {
            data[i] *= attention_factor;
        }
    }
}

// TensorInheritanceLink implementation  
void TensorInheritanceLink::update_tensor(struct ggml_context* ctx) {
    if (!tensor) return;
    
    // Update relation tensor based on truth value
    float* data = (float*)tensor->data;
    if (data) {
        float strength_factor = truth.strength * truth.confidence;
        for (size_t i = 0; i < RELATION_DIM; ++i) {
            data[i] *= strength_factor;
        }
    }
}

// TensorAtomSpace implementation
TensorAtomSpace::TensorAtomSpace(size_t memory_size) 
    : ctx_(nullptr), next_handle_(1), mem_size_(memory_size), mem_buffer_(nullptr) {
    initialize_memory();
}

TensorAtomSpace::~TensorAtomSpace() {
    if (ctx_) {
        ggml_free(ctx_);
    }
    if (mem_buffer_) {
        free(mem_buffer_);
    }
}

void TensorAtomSpace::initialize_memory() {
    // Allocate memory buffer
    mem_buffer_ = malloc(mem_size_);
    if (!mem_buffer_) {
        throw std::runtime_error("Failed to allocate memory for AtomSpace");
    }
    
    // Initialize GGML context
    struct ggml_init_params params = {
        .mem_size = mem_size_,
        .mem_buffer = mem_buffer_,
        .no_alloc = false,
    };
    
    ctx_ = ggml_init(params);
    if (!ctx_) {
        free(mem_buffer_);
        mem_buffer_ = nullptr;
        throw std::runtime_error("Failed to initialize GGML context");
    }
}

Handle TensorAtomSpace::allocate_handle() {
    return next_handle_++;
}

struct ggml_tensor* TensorAtomSpace::create_concept_tensor(const std::string& name,
                                                         const float* embedding) {
    struct ggml_tensor* tensor = ggml_new_tensor_1d(ctx_, GGML_TYPE_F32, 
                                                   TensorConceptNode::EMBEDDING_DIM);
    if (!tensor) {
        throw std::runtime_error("Failed to create concept tensor");
    }
    
    // Initialize tensor data
    float* data = (float*)tensor->data;
    if (embedding) {
        // Use provided embedding
        memcpy(data, embedding, TensorConceptNode::EMBEDDING_DIM * sizeof(float));
    } else {
        // Initialize with random values based on name hash
        std::hash<std::string> hasher;
        size_t seed = hasher(name);
        std::mt19937 gen(seed);
        std::normal_distribution<float> dist(0.0f, 0.1f);
        
        for (size_t i = 0; i < TensorConceptNode::EMBEDDING_DIM; ++i) {
            data[i] = dist(gen);
        }
    }
    
    return tensor;
}

struct ggml_tensor* TensorAtomSpace::create_inheritance_tensor(Handle source, Handle target) {
    struct ggml_tensor* tensor = ggml_new_tensor_2d(ctx_, GGML_TYPE_F32, 768, 768);
    if (!tensor) {
        throw std::runtime_error("Failed to create inheritance tensor");
    }
    
    // Initialize with identity-like matrix modified by source/target embeddings
    float* data = (float*)tensor->data;
    
    // Get source and target tensors
    TensorAtom* src_atom = get_atom(source);
    TensorAtom* tgt_atom = get_atom(target);
    
    if (src_atom && tgt_atom && src_atom->tensor && tgt_atom->tensor) {
        float* src_data = (float*)src_atom->tensor->data;
        float* tgt_data = (float*)tgt_atom->tensor->data;
        
        // Create relation matrix as outer product of embeddings
        for (int i = 0; i < 768; ++i) {
            for (int j = 0; j < 768; ++j) {
                data[i * 768 + j] = src_data[i] * tgt_data[j] * 0.01f;
            }
        }
    } else {
        // Initialize with small random values
        std::mt19937 gen(source ^ target);
        std::normal_distribution<float> dist(0.0f, 0.01f);
        
        for (size_t i = 0; i < 768 * 768; ++i) {
            data[i] = dist(gen);
        }
    }
    
    return tensor;
}

Handle TensorAtomSpace::add_concept(const std::string& name, const float* embedding) {
    // Check if concept already exists
    auto it = name_index_.find(name);
    if (it != name_index_.end()) {
        return it->second;
    }
    
    Handle handle = allocate_handle();
    struct ggml_tensor* tensor = create_concept_tensor(name, embedding);
    
    auto concept = std::make_unique<TensorConceptNode>(handle, name, tensor);
    name_index_[name] = handle;
    atoms_[handle] = std::move(concept);
    
    return handle;
}

Handle TensorAtomSpace::add_inheritance(Handle source, Handle target, const TruthValue& tv) {
    // Verify source and target exist
    if (!get_atom(source) || !get_atom(target)) {
        return INVALID_HANDLE;
    }
    
    Handle handle = allocate_handle();
    struct ggml_tensor* tensor = create_inheritance_tensor(source, target);
    
    auto link = std::make_unique<TensorInheritanceLink>(handle, source, target, tensor);
    link->truth = tv;
    
    // Update incoming/outgoing links
    atoms_[source]->outgoing.push_back(handle);
    atoms_[target]->incoming.push_back(handle);
    
    atoms_[handle] = std::move(link);
    
    return handle;
}

TensorAtom* TensorAtomSpace::get_atom(Handle handle) const {
    auto it = atoms_.find(handle);
    return (it != atoms_.end()) ? it->second.get() : nullptr;
}

Handle TensorAtomSpace::get_handle(const std::string& name) const {
    auto it = name_index_.find(name);
    return (it != name_index_.end()) ? it->second : INVALID_HANDLE;
}

void TensorAtomSpace::update_attention(Handle handle, const AttentionValue& av) {
    TensorAtom* atom = get_atom(handle);
    if (atom) {
        atom->attention = av;
        atom->update_tensor(ctx_);
    }
}

void TensorAtomSpace::spread_activation(Handle source, float amount) {
    TensorAtom* src_atom = get_atom(source);
    if (!src_atom) return;
    
    // Spread activation to connected atoms
    for (Handle target_handle : src_atom->outgoing) {
        TensorAtom* target = get_atom(target_handle);
        if (target && target->type == AtomType::INHERITANCE_LINK) {
            auto* link = static_cast<TensorInheritanceLink*>(target);
            TensorAtom* end_atom = get_atom(link->target_handle);
            if (end_atom) {
                float spread_amount = amount * link->truth.strength * 0.1f;
                end_atom->attention.sti += spread_amount;
                end_atom->update_tensor(ctx_);
            }
        }
    }
}

std::vector<Handle> TensorAtomSpace::get_attentional_focus(float threshold) const {
    std::vector<Handle> focused_atoms;
    
    for (const auto& pair : atoms_) {
        if (pair.second->attention.sti >= threshold) {
            focused_atoms.push_back(pair.first);
        }
    }
    
    // Sort by attention value (descending)
    std::sort(focused_atoms.begin(), focused_atoms.end(),
              [this](Handle a, Handle b) {
                  const TensorAtom* atom_a = get_atom(a);
                  const TensorAtom* atom_b = get_atom(b);
                  return atom_a->attention.sti > atom_b->attention.sti;
              });
    
    return focused_atoms;
}

TensorAtomSpace::Stats TensorAtomSpace::get_stats() const {
    Stats stats = {};
    stats.num_atoms = atoms_.size();
    
    float total_attention = 0.0f;
    for (const auto& pair : atoms_) {
        const TensorAtom* atom = pair.second.get();
        if (atom->type == AtomType::CONCEPT_NODE || 
            atom->type == AtomType::PREDICATE_NODE ||
            atom->type == AtomType::NUMBER_NODE ||
            atom->type == AtomType::VARIABLE_NODE) {
            stats.num_nodes++;
        } else {
            stats.num_links++;
        }
        total_attention += atom->attention.sti;
    }
    
    stats.avg_attention = stats.num_atoms > 0 ? total_attention / stats.num_atoms : 0.0f;
    stats.memory_used = ggml_used_mem(ctx_);
    
    return stats;
}

bool TensorAtomSpace::save_to_gguf(const std::string& filename) const {
    // TODO: Implement GGUF serialization
    // This would serialize the tensor data and atom structure
    std::cout << "GGUF save not yet implemented: " << filename << std::endl;
    return false;
}

bool TensorAtomSpace::load_from_gguf(const std::string& filename) {
    // TODO: Implement GGUF deserialization  
    // This would load tensor data and reconstruct atom structure
    std::cout << "GGUF load not yet implemented: " << filename << std::endl;
    return false;
}

} // namespace opencog