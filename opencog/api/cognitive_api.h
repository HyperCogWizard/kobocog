#pragma once

/**
 * @file cognitive_api.h  
 * @brief API integration for OpenCog cognitive operations in Kobocog
 * 
 * This file provides HTTP API endpoints for cognitive reasoning, memory operations,
 * and attention management integrated with the main Kobocog server.
 */

#include "../atomspace/tensor_atomspace.h"
#include "../reasoning/pln_tensor_engine.h"
#include "../attention/ecan_attention.h"
#include <string>
#include <memory>
#include <functional>

namespace opencog {

/**
 * @brief JSON request/response structures for API
 */
struct CognitiveRequest {
    std::string operation;      // Type of cognitive operation
    std::string params;         // JSON parameters
    std::string session_id;     // Session identifier
};

struct CognitiveResponse {
    bool success;
    std::string result;         // JSON result data
    std::string error_message;
    float confidence;
    int processing_time_ms;
};

/**
 * @brief Memory operation types
 */
enum class MemoryOp {
    ADD_CONCEPT,
    ADD_INHERITANCE,
    QUERY_CONCEPT,
    GET_RELATED,
    UPDATE_ATTENTION,
    GET_STATS
};

/**
 * @brief Reasoning operation types
 */
enum class ReasoningOp {
    FORWARD_CHAIN,
    BACKWARD_CHAIN,
    APPLY_RULE,
    BATCH_INFERENCE,
    EVALUATE_TRUTH
};

/**
 * @brief Main cognitive API handler
 */
class CognitiveAPI {
private:
    std::unique_ptr<TensorAtomSpace> atomspace_;
    std::unique_ptr<PLNTensorEngine> pln_engine_;
    std::unique_ptr<ECANAttentionSystem> attention_system_;
    
    // Session management
    std::unordered_map<std::string, std::chrono::time_point<std::chrono::steady_clock>> sessions_;
    int session_timeout_minutes_;
    
    // Performance monitoring
    struct APIStats {
        size_t total_requests;
        size_t successful_requests;
        float avg_processing_time;
        std::map<std::string, size_t> operation_counts;
    };
    
    mutable APIStats stats_;

public:
    /**
     * @brief Constructor
     */
    CognitiveAPI();
    
    /**
     * @brief Destructor
     */
    ~CognitiveAPI();
    
    /**
     * @brief Initialize cognitive system
     */
    bool initialize(size_t memory_size = 256 * 1024 * 1024);
    
    /**
     * @brief Handle cognitive API request
     */
    CognitiveResponse handle_request(const CognitiveRequest& request);
    
    /**
     * @brief Memory operations
     */
    CognitiveResponse handle_memory_operation(const std::string& params, 
                                             const std::string& session_id);
    
    /**
     * @brief Reasoning operations
     */
    CognitiveResponse handle_reasoning_operation(const std::string& params,
                                                const std::string& session_id);
    
    /**
     * @brief Attention operations
     */
    CognitiveResponse handle_attention_operation(const std::string& params,
                                                const std::string& session_id);
    
    /**
     * @brief Get system status
     */
    CognitiveResponse get_system_status(const std::string& session_id);
    
    /**
     * @brief Load knowledge from file
     */
    CognitiveResponse load_knowledge(const std::string& filename,
                                   const std::string& format = "json");
    
    /**
     * @brief Save knowledge to file
     */
    CognitiveResponse save_knowledge(const std::string& filename,
                                   const std::string& format = "json");
    
    /**
     * @brief Get API statistics
     */
    APIStats get_statistics() const;
    
    /**
     * @brief Register with Kobocog HTTP server
     */
    void register_endpoints(std::function<void(const std::string&, 
                                             std::function<std::string(const std::string&)>)> register_fn);

private:
    // Session management
    std::string create_session();
    bool validate_session(const std::string& session_id);
    void cleanup_expired_sessions();
    
    // JSON processing
    std::string parse_json_request(const std::string& json_str, const std::string& key);
    std::string create_json_response(const CognitiveResponse& response);
    
    // Utility functions
    std::string handle_to_string(Handle handle);
    Handle string_to_handle(const std::string& handle_str);
    std::string truth_value_to_json(const TruthValue& tv);
    TruthValue json_to_truth_value(const std::string& json_str);
    
    // Error handling
    CognitiveResponse create_error_response(const std::string& error_msg);
    CognitiveResponse create_success_response(const std::string& result, 
                                            float confidence = 1.0f);
    
    // Performance monitoring
    void update_stats(const std::string& operation, int processing_time, bool success);
};

/**
 * @brief Specific API endpoint handlers
 */
class MemoryEndpoints {
public:
    static std::string add_concept(TensorAtomSpace* atomspace, const std::string& params);
    static std::string add_inheritance(TensorAtomSpace* atomspace, const std::string& params);
    static std::string query_concept(TensorAtomSpace* atomspace, const std::string& params);
    static std::string get_related_atoms(TensorAtomSpace* atomspace, const std::string& params);
    static std::string update_attention(TensorAtomSpace* atomspace, 
                                       ECANAttentionSystem* attention, 
                                       const std::string& params);
    static std::string get_memory_stats(TensorAtomSpace* atomspace);
};

class ReasoningEndpoints {
public:
    static std::string forward_chain(PLNTensorEngine* pln, const std::string& params);
    static std::string backward_chain(PLNTensorEngine* pln, const std::string& params);
    static std::string apply_rule(PLNTensorEngine* pln, const std::string& params);
    static std::string batch_inference(PLNTensorEngine* pln, const std::string& params);
    static std::string evaluate_truth(PLNTensorEngine* pln, const std::string& params);
};

class AttentionEndpoints {
public:
    static std::string get_focus(ECANAttentionSystem* attention);
    static std::string update_focus(ECANAttentionSystem* attention, const std::string& params);
    static std::string spread_activation(ECANAttentionSystem* attention, const std::string& params);
    static std::string get_importance_ranking(ECANAttentionSystem* attention, const std::string& params);
    static std::string get_attention_stats(ECANAttentionSystem* attention);
};

/**
 * @brief Knowledge import/export utilities
 */
class KnowledgeIO {
public:
    /**
     * @brief Load knowledge from JSON format
     */
    static bool load_from_json(TensorAtomSpace* atomspace, const std::string& filename);
    
    /**
     * @brief Save knowledge to JSON format
     */
    static bool save_to_json(TensorAtomSpace* atomspace, const std::string& filename);
    
    /**
     * @brief Load knowledge from Scheme format (OpenCog native)
     */
    static bool load_from_scheme(TensorAtomSpace* atomspace, const std::string& filename);
    
    /**
     * @brief Convert AtomSpace to OpenCog Scheme format
     */
    static std::string to_scheme_format(TensorAtomSpace* atomspace);
    
    /**
     * @brief Import from external knowledge bases
     */
    static bool import_from_rdf(TensorAtomSpace* atomspace, const std::string& filename);
    static bool import_from_owl(TensorAtomSpace* atomspace, const std::string& filename);

private:
    static std::string atom_to_json(const TensorAtom* atom);
    static Handle json_to_atom(TensorAtomSpace* atomspace, const std::string& json_str);
};

/**
 * @brief Integration with Kobocog main server
 */
class KobocogIntegration {
public:
    /**
     * @brief Initialize cognitive API with Kobocog server
     */
    static bool initialize_cognitive_api();
    
    /**
     * @brief Register cognitive endpoints with HTTP server
     */
    static void register_cognitive_endpoints();
    
    /**
     * @brief Get singleton cognitive API instance
     */
    static CognitiveAPI* get_instance();
    
    /**
     * @brief Shutdown cognitive system
     */
    static void shutdown();

private:
    static std::unique_ptr<CognitiveAPI> instance_;
    static bool initialized_;
};

} // namespace opencog

// C interface for integration with main Kobocog codebase
extern "C" {
    /**
     * @brief Initialize OpenCog cognitive system
     */
    int opencog_init(size_t memory_size);
    
    /**
     * @brief Handle cognitive API request
     */
    const char* opencog_handle_request(const char* request_json);
    
    /**
     * @brief Get system status
     */
    const char* opencog_get_status();
    
    /**
     * @brief Shutdown cognitive system
     */
    void opencog_shutdown();
}